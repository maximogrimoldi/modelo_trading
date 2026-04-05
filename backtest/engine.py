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
    kelly_max        : float — cap máximo de fracción del equity total por barra (default: 0.25).
    position_fraction: float — fracción del equity a desplegar por barra (default: 0.20).
                               Se distribuye proporcionalmente al |z-score| de cada señal.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        opens: pd.DataFrame | None = None,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        rf: float = 0.0,
        kelly_window: int = 30,
        kelly_scale: float = 0.5,
        kelly_default: float = 0.05,
        kelly_max: float = 0.25,
        position_fraction: float = 0.20,
    ):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("El índice de prices debe ser DatetimeIndex.")
        self.prices = prices.copy()
        self.opens  = opens.copy() if opens is not None else None
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.rf = rf
        self.kelly_window = kelly_window
        self.kelly_scale = kelly_scale
        self.kelly_default = kelly_default
        self.kelly_max = kelly_max
        self.position_fraction = position_fraction

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
        positions = {}
        equity_records = []
        trades = []
        pending_signals = {}   # señales del día anterior, a ejecutar al open de hoy

        use_opens = self.opens is not None

        for i in range(len(self.prices)):
            close  = self.prices.iloc[i]
            date   = self.prices.index[i]
            # Precio de ejecución: open de hoy si está disponible, sino close de hoy
            exec_bar = self.opens.iloc[i] if use_opens else close

            # 1. Ejecutar señales pendientes del día anterior al precio de apertura de hoy
            for ticker in pending_signals.get("cerrar_long", []):
                if ticker not in positions or positions[ticker]["side"] != "long":
                    continue
                if ticker not in exec_bar.index or np.isnan(exec_bar[ticker]):
                    continue
                trade, net = self._close_long(ticker, positions[ticker], exec_bar, date)
                cash += net
                trades.append(trade)
                del positions[ticker]

            for ticker in pending_signals.get("cerrar_short", []):
                if ticker not in positions or positions[ticker]["side"] != "short":
                    continue
                if ticker not in exec_bar.index or np.isnan(exec_bar[ticker]):
                    continue
                trade, net = self._close_short(ticker, positions[ticker], exec_bar, date)
                cash += net
                trades.append(trade)
                del positions[ticker]

            new_longs  = [t for t in pending_signals.get("long",  [])
                          if t not in positions and t in exec_bar.index and not np.isnan(exec_bar[t])]
            new_shorts = [t for t in pending_signals.get("short", [])
                          if t not in positions and t in exec_bar.index and not np.isnan(exec_bar[t])]
            n_new = len(new_longs) + len(new_shorts)

            if n_new > 0 and cash > 0:
                equity_now  = self._calc_equity(cash, positions, exec_bar)
                alloc_total = min(equity_now * self.position_fraction, cash)
                if alloc_total >= 1.0:
                    scores = pending_signals.get("scores", {})
                    allocs = self._zscore_alloc(new_longs + new_shorts, alloc_total, scores)
                    for ticker in new_longs:
                        pos, cash_used = self._open_long(ticker, allocs[ticker], exec_bar, date)
                        positions[ticker] = pos
                        cash -= cash_used
                    for ticker in new_shorts:
                        pos, cash_used = self._open_short(ticker, allocs[ticker], exec_bar, date)
                        positions[ticker] = pos
                        cash -= cash_used

            # 2. Mark-to-market al cierre de hoy
            equity = self._calc_equity(cash, positions, close)
            equity_records.append({"date": date, "equity": equity})

            # 3. Generar señales con el cierre de hoy → se ejecutarán mañana al open
            open_pos = {
                "long":  [t for t, p in positions.items() if p["side"] == "long"],
                "short": [t for t, p in positions.items() if p["side"] == "short"],
            }
            pending_signals = self.strategy.generate_signals(
                self.prices.iloc[: i + 1], open_pos
            )

        # Cerrar todo al final del período al último cierre disponible
        # Nota: el último bar ya fue registrado en equity_records dentro del loop.
        # Actualizamos ese último registro con el equity post-cierre (sin solapar fecha).
        final      = self.prices.iloc[-1]
        final_date = self.prices.index[-1]
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if pos["side"] == "long":
                trade, net = self._close_long(ticker, pos, final, final_date)
            else:
                trade, net = self._close_short(ticker, pos, final, final_date)
            cash += net
            trades.append(trade)

        # Reemplazar la última entrada (misma fecha) con el equity real post-cierre
        equity_records[-1] = {"date": final_date, "equity": self._calc_equity(cash, {}, final)}

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
        (Disponible pero no usado en el sizing actual — ver _zscore_alloc)

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

    def _zscore_alloc(self, tickers: list, total: float, scores: dict) -> dict:
        """
        Distribuye `total` entre los tickers proporcionalmente al |z-score|.
        Si no hay scores, distribución igual-peso como fallback.
        """
        if not scores:
            per = total / len(tickers)
            return {t: per for t in tickers}
        weights = {t: scores.get(t, 1.0) for t in tickers}
        w_sum = sum(weights.values())
        return {t: total * weights[t] / w_sum for t in tickers}

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
        sigma_daily = daily_returns.std()
        sigma_annual = sigma_daily * np.sqrt(252.0)

        # Retorno aritmético anualizado (lo que usa el Sharpe clásico)
        arithmetic_annual_pct = daily_returns.mean() * 252.0 * 100.0

        # Volatility drag: la "quita" que cobra la varianza por componer retornos
        # Geométrico ≈ Aritmético - σ²/2  (aproximación log-normal)
        volatility_drag_pct = (sigma_annual ** 2) / 2.0 * 100.0

        # Sharpe clásico (aritmético) — puede engañar en estrategias volátiles
        excess = daily_returns - daily_rf
        sharpe = (
            (excess.mean() / sigma_daily) * np.sqrt(252.0)
            if sigma_daily > 0 else 0.0
        )

        # Geometric Sharpe — usa el CAGR real como numerador
        # Captura el efecto no-ergódico: lo que realmente crecés con el tiempo
        geo_excess_annual = (annualized_return_pct - self.rf * 100.0) / 100.0
        geometric_sharpe = (
            geo_excess_annual / sigma_annual
            if sigma_annual > 0 else 0.0
        )

        # Sortino — penaliza solo la volatilidad negativa (upside vol no es riesgo)
        downside = daily_returns[daily_returns < daily_rf] - daily_rf
        downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else np.nan
        sortino = (
            (excess.mean() / downside_std) * np.sqrt(252.0)
            if downside_std and downside_std > 0 else 0.0
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
            avg_profit   = trades["profit"].mean()
            avg_return   = trades["return_pct"].mean()
            avg_duration = trades["duration_days"].mean()
            n_long  = int((trades["side"] == "long").sum())
            n_short = int((trades["side"] == "short").sum())

            # Expected log-return por trade: la métrica correcta para sistemas no-ergódicos.
            # E[log(1+r)] < E[r] siempre — la diferencia es el costo de la volatilidad.
            # Una estrategia con E[r] > 0 pero E[log(1+r)] < 0 te arruina con certeza.
            r = trades["return_pct"].values / 100.0
            valid = r[r > -1.0]   # log indefinido para pérdidas > 100%
            expected_log_return_pct = float(np.log1p(valid).mean()) * 100.0 if len(valid) > 0 else 0.0
        else:
            win_rate = profit_factor = avg_profit = avg_return = avg_duration = 0.0
            n_long = n_short = 0
            expected_log_return_pct = 0.0

        return {
            # ── Retornos ──────────────────────────────────────────────────────
            "total_return_pct":          round(total_return_pct, 2),
            "annualized_return_pct":     round(annualized_return_pct, 2),   # CAGR (geométrico)
            "arithmetic_return_pct":     round(arithmetic_annual_pct, 2),   # media aritmética × 252
            "volatility_drag_pct":       round(volatility_drag_pct, 2),     # aritmético − geométrico ≈ σ²/2
            # ── Riesgo ajustado ───────────────────────────────────────────────
            "sharpe_ratio":              round(sharpe, 3),           # aritmético — puede engañar
            "geometric_sharpe":          round(geometric_sharpe, 3), # CAGR-based — más honesto
            "sortino_ratio":             round(sortino, 3),          # solo downside vol
            "max_drawdown_pct":          round(max_drawdown_pct, 2),
            "calmar_ratio":              round(calmar, 3),
            # ── No-ergodicidad ────────────────────────────────────────────────
            "expected_log_return_pct":   round(expected_log_return_pct, 4), # E[log(1+r)] por trade
            # ── Trades ───────────────────────────────────────────────────────
            "n_trades":                  n_trades,
            "n_long_trades":             n_long,
            "n_short_trades":            n_short,
            "win_rate_pct":              round(win_rate, 2),
            "profit_factor":             round(profit_factor, 3),
            "avg_profit_usd":            round(avg_profit, 2),
            "avg_return_pct":            round(avg_return, 4),
            "avg_duration_days":         round(avg_duration, 1) if n_trades > 0 else 0.0,
            "final_equity_usd":          round(float(eq.iloc[-1]), 2),
        }

    # ─── Output ───────────────────────────────────────────────────────────────

    def print_metrics(self) -> None:
        if self.metrics is None:
            print("Corré run() primero.")
            return
        m = self.metrics
        print("\n" + "═" * 52)
        print("  RESULTADOS DEL BACKTEST")
        print("═" * 52)
        print(f"  Capital inicial   :  ${self.initial_capital:>12,.2f}")
        print(f"  Capital final     :  ${m['final_equity_usd']:>12,.2f}")
        print(f"  Retorno total     :  {m['total_return_pct']:>+10.2f} %")
        print("─" * 52)
        print(f"  CAGR (geométrico) :  {m['annualized_return_pct']:>+10.2f} %  ← lo que realmente crecés")
        print(f"  Retorno aritmético:  {m['arithmetic_return_pct']:>+10.2f} %  ← lo que 'esperás' en promedio")
        print(f"  Volatility drag   :  {m['volatility_drag_pct']:>10.2f} %  ← costo de la varianza (≈ σ²/2)")
        print("─" * 52)
        print(f"  Sharpe (aritmét.) :  {m['sharpe_ratio']:>12.3f}  ← puede engañar")
        print(f"  Sharpe geométrico :  {m['geometric_sharpe']:>12.3f}  ← más honesto (usa CAGR)")
        print(f"  Sortino Ratio     :  {m['sortino_ratio']:>12.3f}  ← solo downside vol")
        print(f"  Max Drawdown      :  {m['max_drawdown_pct']:>10.2f} %")
        print(f"  Calmar Ratio      :  {m['calmar_ratio']:>12.3f}")
        print("─" * 52)
        print(f"  E[log(1+r)] trade :  {m['expected_log_return_pct']:>+10.4f} %  ← métrica no-ergódica")
        print("─" * 52)
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
