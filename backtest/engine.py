"""
Motor de backtesting multi-ticker, desacoplado de cualquier estrategia concreta.

El engine recibe señales de la estrategia (qué abrir, qué cerrar, en qué tickers)
y ejecuta la mecánica del portafolio: apertura, cierre, mark-to-market y métricas.
No sabe nada de cómo se generan las señales.

Modelo de portafolio
────────────────────
- El capital se asigna usando Kelly Criterion (configurable).
- Posiciones largas: se compra con efectivo, se vende y se recupera efectivo + P&L.
- Posiciones cortas: se reserva capital como margen; al cerrar se devuelve margen + P&L.
- Equity en cada barra:
    cash
    + Σ_longs  (shares × precio_actual)
    + Σ_shorts (margen + shares × (precio_entrada − precio_actual))

Kelly Criterion
───────────────
  f* = p − (1−p)/b
  donde p = win rate y b = avg_gain / avg_loss (en retornos porcentuales)
  estimados sobre los últimos kelly_window trades completados.

  Se aplica half-Kelly (f* × kelly_scale) para robustez ante errores de estimación.
  f* total se divide entre el número de señales nuevas del bar.
  Antes de tener kelly_window trades, se usa kelly_default como fracción inicial.

Costos
──────
- Slippage: compra sube el precio, venta lo baja.
- Comisión: fracción del valor operado, descontada en cada operación.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from strategy.base import Strategy


class BacktestEngine:
    """
    Simula la ejecución de una estrategia multi-ticker sobre datos históricos.

    Parámetros
    ----------
    prices          : pd.DataFrame — precios de cierre (DatetimeIndex × tickers).
    strategy        : Strategy     — cualquier objeto que implemente Strategy.
    initial_capital : float        — capital inicial en USD (default: 100 000).
    commission      : float        — fracción del valor operado (default: 0.001).
    slippage        : float        — fracción del precio de cierre (default: 0.0005).
    rf              : float        — tasa libre de riesgo anual para el Sharpe (default: 0.0).

    Kelly Criterion
    ---------------
    kelly_window    : int   — trades completados necesarios para activar Kelly (default: 30).
    kelly_scale     : float — fracción del Kelly completo a usar; 0.5 = half-Kelly (default: 0.5).
    kelly_default   : float — fracción del equity a usar antes de tener kelly_window trades (default: 0.05).
    kelly_max       : float — cap máximo de fracción del equity total por barra (default: 0.25).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        rf: float = 0.0,
        kelly_window: int = 30,
        kelly_scale: float = 0.5,
        kelly_default: float = 0.05,
        kelly_max: float = 0.25,
    ):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("El índice de prices debe ser DatetimeIndex.")
        self.prices = prices.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.rf = rf
        self.kelly_window = kelly_window
        self.kelly_scale = kelly_scale
        self.kelly_default = kelly_default
        self.kelly_max = kelly_max

        self.equity_curve: pd.Series | None = None
        self.trades: pd.DataFrame | None = None
        self.metrics: dict | None = None

    # ─── Loop principal ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Ejecuta la simulación barra a barra.

        En cada barra:
          1. Mark-to-market del portafolio al precio de cierre.
          2. Se pasan precios históricos + posiciones abiertas a la estrategia.
          3. Se cierran las posiciones indicadas en cerrar_long / cerrar_short.
          4. Se abren las nuevas posiciones indicadas en long / short,
             repartiendo el efectivo disponible en partes iguales.
        """
        cash = self.initial_capital
        positions = {}   # ticker → dict con datos de la posición
        equity_records = []
        trades = []

        for i in range(len(self.prices)):
            current = self.prices.iloc[i]
            date = self.prices.index[i]

            # 1. Mark-to-market
            equity = self._calc_equity(cash, positions, current)
            equity_records.append({"date": date, "equity": equity})

            # 2. Estado actual para pasarle a la estrategia
            open_positions = {
                "long":  [t for t, p in positions.items() if p["side"] == "long"],
                "short": [t for t, p in positions.items() if p["side"] == "short"],
            }

            # 3. Señales de la estrategia (sin lookahead)
            signals = self.strategy.generate_signals(
                self.prices.iloc[: i + 1], open_positions
            )

            # 4. Cerrar posiciones
            for ticker in signals.get("cerrar_long", []):
                if ticker in positions and positions[ticker]["side"] == "long":
                    trade, net = self._close_long(ticker, positions[ticker], current, date)
                    cash += net
                    trades.append(trade)
                    del positions[ticker]

            for ticker in signals.get("cerrar_short", []):
                if ticker in positions and positions[ticker]["side"] == "short":
                    trade, net = self._close_short(ticker, positions[ticker], current, date)
                    cash += net
                    trades.append(trade)
                    del positions[ticker]

            # 5. Abrir posiciones nuevas con sizing Kelly
            new_longs  = [t for t in signals.get("long",  [])
                          if t not in positions and t in current.index and not np.isnan(current[t])]
            new_shorts = [t for t in signals.get("short", [])
                          if t not in positions and t in current.index and not np.isnan(current[t])]
            n_new = len(new_longs) + len(new_shorts)

            if n_new > 0 and cash > 0:
                # Fracción Kelly del equity total a desplegar en estas señales
                kelly_f = self._kelly_fraction(trades)
                equity_now = self._calc_equity(cash, positions, current)

                # Inversión total deseada = kelly_f × equity, pero sin superar el cash
                ideal_total = equity_now * kelly_f
                alloc_total = min(ideal_total, cash)
                alloc = alloc_total / n_new   # igual entre todas las señales del bar

                # No abrir si la asignación es insignificante (Kelly cercano a 0)
                if alloc < 1.0:
                    continue

                for ticker in new_longs:
                    pos, cash_used = self._open_long(ticker, alloc, current, date)
                    positions[ticker] = pos
                    cash -= cash_used

                for ticker in new_shorts:
                    pos, cash_used = self._open_short(ticker, alloc, current, date)
                    positions[ticker] = pos
                    cash -= cash_used

        # Cerrar todo al final del período
        final = self.prices.iloc[-1]
        final_date = self.prices.index[-1]
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if pos["side"] == "long":
                trade, net = self._close_long(ticker, pos, final, final_date)
            else:
                trade, net = self._close_short(ticker, pos, final, final_date)
            cash += net
            trades.append(trade)

        self.equity_curve = (
            pd.DataFrame(equity_records).set_index("date")["equity"]
        )
        self.trades = (
            pd.DataFrame(trades) if trades
            else pd.DataFrame(columns=[
                "ticker", "side", "entry_date", "exit_date",
                "entry_price", "exit_price", "shares",
                "profit", "return_pct", "duration_days",
            ])
        )
        self.metrics = self._calculate_metrics()
        return self.metrics

    # ─── Mecánica de posiciones ───────────────────────────────────────────────

    def _open_long(self, ticker, allocation, current_prices, date):
        exec_price = float(current_prices[ticker]) * (1.0 + self.slippage)
        commission_cost = allocation * self.commission
        net_alloc = allocation - commission_cost
        shares = net_alloc / exec_price
        pos = {
            "side": "long",
            "shares": shares,
            "entry_price": exec_price,
            "entry_date": date,
            "alloc": allocation,
        }
        return pos, allocation   # cash_used

    def _open_short(self, ticker, allocation, current_prices, date):
        exec_price = float(current_prices[ticker]) * (1.0 - self.slippage)
        commission_cost = allocation * self.commission
        net_alloc = allocation - commission_cost
        shares = net_alloc / exec_price
        pos = {
            "side": "short",
            "shares": shares,
            "entry_price": exec_price,
            "entry_date": date,
            "alloc": allocation,
            "margin": allocation,   # capital reservado como garantía
        }
        return pos, allocation   # cash_used

    def _close_long(self, ticker, pos, current_prices, date):
        exec_price = float(current_prices[ticker]) * (1.0 - self.slippage)
        gross = pos["shares"] * exec_price
        commission_cost = gross * self.commission
        net_proceeds = gross - commission_cost
        profit = net_proceeds - pos["alloc"]
        trade = {
            "ticker": ticker,
            "side": "long",
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exec_price, 4),
            "shares": round(pos["shares"], 6),
            "profit": round(profit, 2),
            "return_pct": round(profit / pos["alloc"] * 100, 4) if pos["alloc"] > 0 else 0.0,
            "duration_days": (date - pos["entry_date"]).days,
        }
        return trade, net_proceeds

    def _close_short(self, ticker, pos, current_prices, date):
        exec_price = float(current_prices[ticker]) * (1.0 + self.slippage)
        buyback_gross = pos["shares"] * exec_price
        commission_cost = buyback_gross * self.commission
        pnl = pos["shares"] * (pos["entry_price"] - exec_price) - commission_cost
        cash_return = pos["margin"] + pnl
        trade = {
            "ticker": ticker,
            "side": "short",
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exec_price, 4),
            "shares": round(pos["shares"], 6),
            "profit": round(pnl, 2),
            "return_pct": round(pnl / pos["alloc"] * 100, 4) if pos["alloc"] > 0 else 0.0,
            "duration_days": (date - pos["entry_date"]).days,
        }
        return trade, cash_return

    def _kelly_fraction(self, completed_trades: list) -> float:
        """
        Calcula la fracción del equity a invertir usando Kelly Criterion.

        Fórmula: f* = p − (1−p)/b
          p = win rate estimado sobre los últimos kelly_window trades
          b = avg_return_ganador / avg_return_perdedor (en %)

        Se aplica kelly_scale (half-Kelly por defecto) y se capa en kelly_max.
        Antes de tener kelly_window trades usa kelly_default como arranque conservador.
        """
        if len(completed_trades) < self.kelly_window:
            return self.kelly_default

        sample = completed_trades[-self.kelly_window:]
        rets = [t["return_pct"] for t in sample]

        wins   = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]

        # Sin suficiente diversidad para estimar, usar default
        if not wins or not losses:
            return self.kelly_default

        p = len(wins) / len(sample)
        b = np.mean(wins) / abs(np.mean(losses))   # odds ratio

        kelly_full = p - (1.0 - p) / b
        kelly_full = max(0.0, kelly_full)           # nunca negativo

        # Half-Kelly (o el factor configurado) + cap máximo
        return min(kelly_full * self.kelly_scale, self.kelly_max)

    def _calc_equity(self, cash, positions, current_prices):
        equity = cash
        for ticker, pos in positions.items():
            if ticker not in current_prices.index:
                continue
            price = float(current_prices[ticker])
            if np.isnan(price):
                continue
            if pos["side"] == "long":
                equity += pos["shares"] * price
            else:  # short
                # margen recuperado + P&L no realizado
                equity += pos["margin"] + pos["shares"] * (pos["entry_price"] - price)
        return equity

    # ─── Métricas ─────────────────────────────────────────────────────────────

    def _calculate_metrics(self) -> dict:
        eq = self.equity_curve
        trades = self.trades

        total_return_pct = (eq.iloc[-1] / self.initial_capital - 1.0) * 100.0

        total_days = (eq.index[-1] - eq.index[0]).days
        years = total_days / 365.25
        annualized_return_pct = (
            ((eq.iloc[-1] / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0
            if years > 0 else 0.0
        )

        daily_returns = eq.pct_change().dropna()
        daily_rf = self.rf / 252.0
        excess = daily_returns - daily_rf
        sharpe = (
            (excess.mean() / excess.std()) * np.sqrt(252.0)
            if excess.std() > 0 else 0.0
        )

        running_max = np.maximum.accumulate(eq.values)
        drawdowns = (running_max - eq.values) / running_max
        max_drawdown_pct = float(np.max(drawdowns)) * 100.0

        calmar = (
            annualized_return_pct / max_drawdown_pct
            if max_drawdown_pct > 0 else 0.0
        )

        n_trades = len(trades)
        if n_trades > 0:
            wins = trades[trades["profit"] > 0]
            losses = trades[trades["profit"] <= 0]
            win_rate = len(wins) / n_trades * 100.0
            profit_factor = (
                wins["profit"].sum() / abs(losses["profit"].sum())
                if losses["profit"].sum() != 0 else float("inf")
            )
            avg_profit = trades["profit"].mean()
            avg_return = trades["return_pct"].mean()
            avg_duration = trades["duration_days"].mean()
            n_long = int((trades["side"] == "long").sum())
            n_short = int((trades["side"] == "short").sum())
        else:
            win_rate = profit_factor = avg_profit = avg_return = avg_duration = 0.0
            n_long = n_short = 0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized_return_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "calmar_ratio": round(calmar, 3),
            "n_trades": n_trades,
            "n_long_trades": n_long,
            "n_short_trades": n_short,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_profit_usd": round(avg_profit, 2),
            "avg_return_pct": round(avg_return, 4),
            "avg_duration_days": round(avg_duration, 1) if n_trades > 0 else 0.0,
            "final_equity_usd": round(float(eq.iloc[-1]), 2),
        }

    # ─── Output ───────────────────────────────────────────────────────────────

    def print_metrics(self) -> None:
        if self.metrics is None:
            print("Corré run() primero.")
            return
        m = self.metrics
        print("\n" + "═" * 44)
        print("  RESULTADOS DEL BACKTEST")
        print("═" * 44)
        print(f"  Capital inicial :  ${self.initial_capital:>12,.2f}")
        print(f"  Capital final   :  ${m['final_equity_usd']:>12,.2f}")
        print(f"  Retorno total   :  {m['total_return_pct']:>+10.2f} %")
        print(f"  Retorno anual.  :  {m['annualized_return_pct']:>+10.2f} %")
        print("─" * 44)
        print(f"  Sharpe Ratio    :  {m['sharpe_ratio']:>12.3f}")
        print(f"  Max Drawdown    :  {m['max_drawdown_pct']:>10.2f} %")
        print(f"  Calmar Ratio    :  {m['calmar_ratio']:>12.3f}")
        print("─" * 44)
        print(f"  Trades totales  :  {m['n_trades']:>12}  "
              f"(L: {m['n_long_trades']} / S: {m['n_short_trades']})")
        print(f"  Win Rate        :  {m['win_rate_pct']:>10.2f} %")
        print(f"  Profit Factor   :  {m['profit_factor']:>12.3f}")
        print(f"  Ganancia media  :  ${m['avg_profit_usd']:>11,.2f}")
        print(f"  Retorno medio   :  {m['avg_return_pct']:>+10.4f} %")
        print(f"  Duración media  :  {m['avg_duration_days']:>9.1f} días")
        print("═" * 44 + "\n")

    def plot_equity_curve(self, save_path: str | None = None, show: bool = True) -> None:
        if self.equity_curve is None:
            print("Corré run() primero.")
            return

        eq = self.equity_curve
        running_max = np.maximum.accumulate(eq.values)
        drawdown = (running_max - eq.values) / running_max * 100.0

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        ax1.plot(eq.index, eq.values, color="#1f77b4", linewidth=1.5, label="Portafolio")
        ax1.axhline(self.initial_capital, color="gray", linewidth=0.8, linestyle="--",
                    label=f"Capital inicial (${self.initial_capital:,.0f})")
        ax1.set_ylabel("Valor del portafolio (USD)")
        ax1.set_title(
            f"Equity Curve — retorno: {self.metrics['total_return_pct']:+.2f}%  |  "
            f"Sharpe: {self.metrics['sharpe_ratio']:.3f}  |  "
            f"Max DD: {self.metrics['max_drawdown_pct']:.2f}%"
        )
        ax1.legend(fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(eq.index, drawdown, color="#d62728", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Fecha")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Gráfico guardado en: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    def __repr__(self) -> str:
        status = "ejecutado" if self.metrics else "no ejecutado"
        return (
            f"BacktestEngine(strategy={self.strategy.__class__.__name__}, "
            f"tickers={len(self.prices.columns)}, "
            f"capital=${self.initial_capital:,.0f}, "
            f"kelly_scale={self.kelly_scale}, estado={status})"
        )
