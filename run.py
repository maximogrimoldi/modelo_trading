"""
Entry point del motor de backtesting.

Conecta datos, estrategia y engine. Para cambiar la estrategia,
solo modificás las líneas marcadas con ──►.

Uso:
    python run.py
"""

import os

from data.loader import DataLoader
from backtest.engine import BacktestEngine

# ─────────────────────────────────────────────────────────────────────────────
# ──► PASO 1: Elegí la estrategia
#
#     Opción A — RMT (estrategia principal):
from rmt_strategy import RMTBacktestStrategy
strategy = RMTBacktestStrategy(
    entry_threshold=2.0,
    exit_threshold=0.5,
    ventana_betas=252,
    ventana_zscore=252,
)
#
#     Opción B — Momentum (ejemplo básico):
#     from example_strategy import MomentumStrategy
#     strategy = MomentumStrategy(lookback=20, entry_threshold=0.05)
#
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ──► PASO 2: Universo y configuración del backtest
# ─────────────────────────────────────────────────────────────────────────────

START_DATE      = "2021-01-01"   # 4 años: ~1 año warmup + 3 años de señales
END_DATE        = "2024-12-31"
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 0.001          # 0.1%
SLIPPAGE        = 0.0005         # 0.05%
RISK_FREE_RATE  = 0.04

SAVE_RESULTS    = True
RESULTS_DIR     = "results/"

# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"\nDescargando precios del S&P 100 ({START_DATE} → {END_DATE})...")
    loader = DataLoader(
        start_date=START_DATE,
        end_date=END_DATE,
        frecuencia="diaria",
    )
    prices = loader.get_prices()
    opens  = loader.get_opens()
    print(f"  {prices.shape[1]} tickers  |  {len(prices)} barras  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    print(f"\nEstrategia: {strategy}")
    print(f"Comisión: {COMMISSION*100:.2f}%  |  Slippage: {SLIPPAGE*100:.3f}%")
    print("\nCorriendo backtest (puede tardar ~1 min)...")

    engine = BacktestEngine(
        prices=prices,
        opens=opens,
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        rf=RISK_FREE_RATE,
    )

    engine.run()
    engine.print_metrics()

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        trades_path = os.path.join(RESULTS_DIR, "trades.csv")
        equity_path = os.path.join(RESULTS_DIR, "equity_curve.csv")
        plot_path   = os.path.join(RESULTS_DIR, "equity_plot.png")

        engine.trades.to_csv(trades_path, index=False)
        engine.equity_curve.to_csv(equity_path)
        print(f"  Trades guardados en:       {trades_path}")
        print(f"  Equity curve guardada en:  {equity_path}")
        engine.plot_equity_curve(save_path=plot_path, show=False)
    else:
        engine.plot_equity_curve(show=True)


if __name__ == "__main__":
    main()
