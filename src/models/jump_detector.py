"""Detección de saltos (jumps) en series de retornos mediante el estadístico de Lee-Mykland.

El test de Lee-Mykland (2008) identifica saltos de precio a nivel intradiario
usando la variación bipower como estimador robusto de la varianza continua,
permitiendo distinguir movimientos bruscos genuinos de volatilidad estocástica normal.

Referencias:
    - Lee, S. S. & Mykland, P. A. (2008). Jumps in financial markets:
      A new nonparametric test and jump dynamics.
      Review of Financial Studies, 21(6), 2535–2563.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class JumpResult:
    """Resultado del análisis de saltos para un activo.

    Attributes:
        ticker: Símbolo del activo analizado.
        jump_dates: Lista de fechas donde se detectó un salto.
        statistics: Series con el estadístico L de Lee-Mykland por fecha.
        threshold: Valor crítico usado (nivel de significancia).
        jump_count: Total de saltos detectados.
    """

    ticker: str
    jump_dates: List[pd.Timestamp]
    statistics: pd.Series
    threshold: float
    jump_count: int


class JumpDetector:
    """Detecta saltos en retornos usando el estadístico de Lee-Mykland.

    El estadístico L(t) = r(t) / BPV_local mide cuántas desviaciones estándar
    (estimadas por variación bipower local) representa cada retorno. Un |L(t)|
    que excede el cuantil de la distribución de máximos de normales estándar
    se clasifica como salto.

    Attributes:
        significance: Nivel de significancia para el test (ej. 0.01 = 1%).
        window: Ventana deslizante para estimar la BPV local.
        mu_1: Constante E[|Z|] para distribución normal = sqrt(2/pi).
    """

    MU_1: float = np.sqrt(2.0 / np.pi)  # E[|Z|] para Z ~ N(0,1)

    def __init__(
        self,
        significance: float = 0.01,
        window: int = 252,
    ) -> None:
        """Inicializa el JumpDetector.

        Args:
            significance: Nivel de significancia del test (valor entre 0 y 1).
            window: Ventana (en observaciones) para calcular la BPV local.
        """
        if not 0 < significance < 1:
            raise ValueError(f"significance debe estar en (0, 1). Recibido: {significance}")
        self.significance = significance
        self.window = window
        self._threshold: float | None = None

    def detect(self, returns: pd.Series) -> JumpResult:
        """Aplica el test de Lee-Mykland a una serie de retornos.

        Args:
            returns: Serie de retornos con DatetimeIndex y nombre = ticker.

        Returns:
            JumpResult con las fechas de saltos y estadísticos.

        Raises:
            ValueError: Si la serie tiene menos observaciones que la ventana.
        """
        if len(returns) < self.window:
            raise ValueError(
                f"La serie tiene {len(returns)} obs. Se necesitan al menos {self.window}."
            )

        ticker = returns.name or "UNKNOWN"
        logger.info("Detectando saltos en %s (%d obs.)", ticker, len(returns))

        # TODO: Paso 1 — calcular BPV local (variación bipower en ventana rolling)
        # BPV(t) = (pi/2) * sum_{i=t-w}^{t} |r_i| * |r_{i-1}|
        # TODO: Paso 2 — calcular estadístico L(t) = r(t) / sqrt(BPV_local(t) / n_window)
        # TODO: Paso 3 — calcular threshold c_alpha basado en distribución de extremos
        # threshold ~ sqrt(2 * log(n)) - (log(pi) + log(log(n))) / (2 * sqrt(2 * log(n)))
        # TODO: Paso 4 — identificar fechas donde |L(t)| > threshold
        raise NotImplementedError("JumpDetector.detect() — pendiente de implementación.")

    def detect_universe(
        self, returns: pd.DataFrame
    ) -> Dict[str, JumpResult]:
        """Aplica la detección de saltos a todo el universo de activos.

        Args:
            returns: DataFrame de retornos con forma (T, N). Columnas = tickers.

        Returns:
            Diccionario {ticker: JumpResult} para cada activo.
        """
        results: Dict[str, JumpResult] = {}
        for ticker in returns.columns:
            try:
                results[ticker] = self.detect(returns[ticker].rename(ticker))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error procesando %s: %s", ticker, exc)
        return results

    def get_jump_mask(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Construye una máscara booleana de saltos para todo el universo.

        Args:
            returns: DataFrame de retornos con forma (T, N).

        Returns:
            DataFrame booleano (T, N). True indica un salto en esa fecha/activo.
        """
        # TODO: Usar detect_universe y construir el DataFrame de máscaras
        raise NotImplementedError("get_jump_mask() — pendiente de implementación.")

    def _bipower_variation(self, returns: pd.Series) -> pd.Series:
        """Calcula la variación bipower local en ventana rolling.

        Args:
            returns: Serie de retornos diarios.

        Returns:
            Serie con la BPV estimada para cada fecha (ventana deslizante).
        """
        # TODO: BPV = (pi/2) * rolling_sum(|r_t| * |r_{t-1}|, window=self.window)
        raise NotImplementedError("_bipower_variation() — pendiente de implementación.")
