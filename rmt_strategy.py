"""
Adapter que conecta RMTStrategy al motor de backtesting.

Implementa la interfaz Strategy usando el pipeline RMT existente en
strategy/signals.py, sin modificarlo. En cada barra del backtest:

  1. Toma los últimos ventana_betas días de retornos como ventana de entrenamiento.
  2. Estima los factores RMT (Marchenko-Pastur) sobre esa ventana.
  3. Calcula el residuo idiosincrático SOLO del día actual (sin lookahead).
  4. Acumula ese residuo en un historial propio.
  5. Computa el z-score sobre el historial acumulado.
  6. Devuelve señales de apertura/cierre en el formato que espera BacktestEngine.

Por qué residuo incremental y no calcular_residuos_rolling en cada barra:
  calcular_residuos_rolling recorre toda la historia → O(n²) total.
  El enfoque incremental calcula UN solo residuo por barra → O(n) total.
  El resultado es matemáticamente equivalente.
"""

import numpy as np
import pandas as pd

from strategy.base import Strategy
from strategy.signals import RMTStrategy


class RMTBacktestStrategy(Strategy):
    """
    Wrappea RMTStrategy para correr sobre el motor de backtesting.

    Parámetros
    ----------
    entry_threshold : float — z-score para abrir posición (default: 2.0).
    exit_threshold  : float — z-score para cerrar posición (default: 0.5).
    ventana_betas   : int   — días usados para estimar los factores RMT (default: 252).
    ventana_zscore  : int   — días de historial de residuos para el z-score (default: 252).
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        ventana_betas: int = 252,
        ventana_zscore: int = 252,
    ):
        self.rmt = RMTStrategy(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        self.ventana_betas = ventana_betas
        self.ventana_zscore = ventana_zscore

        # Historial de residuos acumulados barra a barra (lista de arrays 1D)
        self._residuos_acum: list[np.ndarray] = []

    def generate_signals(self, prices: pd.DataFrame, open_positions: dict) -> dict:
        """
        prices        : Close prices (fechas × tickers) hasta la barra actual.
        open_positions: {"long": [...], "short": [...]} — estado actual del portafolio.
        """
        vacío = {"long": [], "short": [], "cerrar_long": [], "cerrar_short": []}

        # Trabajar solo sobre la ventana necesaria para eficiencia O(n)
        # +2 porque pct_change() pierde la primera fila
        prices_window = prices.iloc[-(self.ventana_betas + 2):]
        retornos = prices_window.pct_change().dropna(how="all")

        # Período de calentamiento: necesitamos ventana_betas días de entrenamiento
        # más el día actual → ventana_betas + 1 filas de retornos
        if len(retornos) < self.ventana_betas + 1:
            return vacío

        R = retornos.values          # (ventana_betas+1, N)
        T, N = R.shape
        tickers = list(retornos.columns)

        # R_train: ventana_betas días ANTES del día actual (pasado puro)
        # R_last:  retorno del día actual
        R_train = R[-(self.ventana_betas + 1):-1]   # (ventana_betas, N)
        R_last = R[-1]                               # (N,)

        # ── Pipeline RMT ──────────────────────────────────────────────────────
        corr = self.rmt.correlacion(R_train)
        autovalores, autovectores = self.rmt.pca(corr)
        V = self.rmt.filtrar_marchenko_pastur(
            autovalores, autovectores, self.ventana_betas, N
        )

        # Si no hay factores de señal (todo ruido), sin operaciones
        if V.shape[1] == 0:
            return vacío

        _, B = self.rmt.betas(R_train, V)

        # Residuo idiosincrático del día actual (estimado con betas del pasado)
        f_last = R_last @ V                  # proyección sobre factores
        residuo_hoy = R_last - f_last @ B   # (N,) — lo que no explican los factores

        # ── Acumular historial ────────────────────────────────────────────────
        self._residuos_acum.append(residuo_hoy)
        if len(self._residuos_acum) > self.ventana_zscore:
            self._residuos_acum.pop(0)

        # Z-score sobre los residuos acumulados (misma lógica que en signals.py)
        acum = np.cumsum(np.array(self._residuos_acum), axis=0)
        zs = pd.Series(self.rmt.zscore(acum), index=tickers)

        # ── Señales ───────────────────────────────────────────────────────────
        abiertas_long  = set(open_positions.get("long",  []))
        abiertas_short = set(open_positions.get("short", []))
        ya_en_cartera  = abiertas_long | abiertas_short

        return {
            "long":  list(
                zs[zs < -self.rmt.entry_threshold]
                .drop(index=ya_en_cartera, errors="ignore").index
            ),
            "short": list(
                zs[zs >  self.rmt.entry_threshold]
                .drop(index=ya_en_cartera, errors="ignore").index
            ),
            "cerrar_long":  [t for t in abiertas_long  if zs.get(t, 0) > -self.rmt.exit_threshold],
            "cerrar_short": [t for t in abiertas_short if zs.get(t, 0) <  self.rmt.exit_threshold],
        }

    def __repr__(self) -> str:
        return (
            f"RMTBacktestStrategy("
            f"entry={self.rmt.entry_threshold}, "
            f"exit={self.rmt.exit_threshold}, "
            f"ventana_betas={self.ventana_betas}, "
            f"ventana_zscore={self.ventana_zscore})"
        )
