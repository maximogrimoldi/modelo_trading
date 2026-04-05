"""
Estrategia de ejemplo: momentum cross-sectional multi-ticker.

Implementa la interfaz Strategy para verificar que el engine funciona.
No está optimizada para trading real — es solo un ejemplo didáctico
que muestra cómo conectar cualquier estrategia al motor.

Lógica:
  - Para cada ticker, calcula el retorno de los últimos `lookback` días.
  - Tickers con retorno > entry_threshold  → señal long.
  - Tickers con retorno < -entry_threshold → señal short.
  - Se cierra un long cuando el retorno baja de exit_threshold.
  - Se cierra un short cuando el retorno sube de -exit_threshold.

─────────────────────────────────────────────────────────────────────────────
PARA CREAR TU PROPIA ESTRATEGIA:

    from strategy.base import Strategy
    import pandas as pd

    class MiEstrategia(Strategy):
        def generate_signals(self, prices: pd.DataFrame, open_positions: dict) -> dict:
            # prices: DataFrame fechas × tickers (precios de cierre, hasta hoy)
            # open_positions: {"long": [...], "short": [...]}
            # Retornás qué abrir y qué cerrar.
            return {
                "long":         [...],
                "short":        [...],
                "cerrar_long":  [...],
                "cerrar_short": [...],
            }

Luego en run.py, reemplazá la línea marcada con ──►.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from strategy.base import Strategy


class MomentumStrategy(Strategy):
    """
    Estrategia de momentum cross-sectional.

    Abre longs en los tickers con mayor momentum positivo reciente
    y shorts en los de mayor momentum negativo.

    Parámetros
    ----------
    lookback        : int   — período para calcular el retorno (default: 20 días).
    entry_threshold : float — retorno mínimo para abrir posición (default: 0.05 = 5%).
    exit_threshold  : float — retorno por debajo del cual se cierra (default: 0.0).
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 0.05,
        exit_threshold: float = 0.0,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, prices: pd.DataFrame, open_positions: dict) -> dict:
        # Período de calentamiento
        if len(prices) < self.lookback + 1:
            return {"long": [], "short": [], "cerrar_long": [], "cerrar_short": []}

        # Retorno de los últimos `lookback` días para cada ticker
        ret = prices.iloc[-1] / prices.iloc[-self.lookback] - 1

        abiertas_long  = set(open_positions.get("long",  []))
        abiertas_short = set(open_positions.get("short", []))
        ya_en_cartera  = abiertas_long | abiertas_short

        # Nuevas entradas (solo tickers sin posición abierta)
        candidatos = ret.dropna().drop(index=list(ya_en_cartera), errors="ignore")
        new_longs  = list(candidatos[candidatos >  self.entry_threshold].index)
        new_shorts = list(candidatos[candidatos < -self.entry_threshold].index)

        # Cierres
        cerrar_long  = [t for t in abiertas_long  if ret.get(t, 0) <= self.exit_threshold]
        cerrar_short = [t for t in abiertas_short if ret.get(t, 0) >= -self.exit_threshold]

        return {
            "long":         new_longs,
            "short":        new_shorts,
            "cerrar_long":  cerrar_long,
            "cerrar_short": cerrar_short,
        }

    def __repr__(self) -> str:
        return (
            f"MomentumStrategy(lookback={self.lookback}, "
            f"entry={self.entry_threshold}, exit={self.exit_threshold})"
        )
