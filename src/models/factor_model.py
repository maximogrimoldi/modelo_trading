"""Modelo de Factores basado en PCA sobre la matriz de correlación denoised.

Construye eigenportfolios ortogonales a partir de los eigenvectores de señal
devueltos por RMTCleaner, calcula los retornos de los factores y los residuos
(retornos idiosincráticos) para cada activo del universo.

Las señales de trading surgen de residuos que exhiben reversión a la media
estadísticamente significativa y no contaminados por saltos de precio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class FactorModelResult:
    """Resultado del ajuste del Factor Model.

    Attributes:
        factor_returns: DataFrame de retornos de eigenportfolios (T, K).
        residuals: DataFrame de residuos idiosincráticos (T, N).
        loadings: Matriz de cargas factoriales (N, K).
        explained_variance_ratio: Varianza explicada por cada factor.
        tickers: Lista de activos del universo.
    """

    factor_returns: pd.DataFrame
    residuals: pd.DataFrame
    loadings: np.ndarray
    explained_variance_ratio: np.ndarray
    tickers: List[str]


class FactorModel:
    """Construye eigenportfolios y calcula residuos idiosincráticos.

    Flujo:
        1. Recibe la matriz de correlación denoised de RMTCleaner.
        2. Extrae los K eigenvectores de mayor eigenvalor (señal).
        3. Proyecta los retornos sobre los eigenvectores → factor_returns.
        4. Calcula residuos: e_t = r_t - B @ f_t, donde B son las cargas.
        5. Expone residuos para el pipeline de señales (JumpDetector, z-score).

    Attributes:
        n_factors: Número de factores (eigenportfolios) a retener.
        demean: Si True, sustrae la media antes de proyectar.
    """

    def __init__(
        self,
        n_factors: int,
        demean: bool = True,
    ) -> None:
        """Inicializa el FactorModel.

        Args:
            n_factors: Cantidad de eigenportfolios a construir.
            demean: Si True, centra los retornos antes de la descomposición.
        """
        if n_factors < 1:
            raise ValueError(f"n_factors debe ser >= 1. Recibido: {n_factors}")
        self.n_factors = n_factors
        self.demean = demean
        self._pca: Optional[PCA] = None
        self._result: Optional[FactorModelResult] = None

    def fit(
        self,
        log_returns: pd.DataFrame,
        corr_matrix: Optional[np.ndarray] = None,
    ) -> "FactorModel":
        """Ajusta el modelo PCA sobre los retornos o la correlación denoised.

        Args:
            log_returns: DataFrame de log-retornos (T, N). Índice: DatetimeIndex.
            corr_matrix: Matriz de correlación denoised (N, N) de RMTCleaner.
                Si es None, se calcula la correlación empírica directamente.

        Returns:
            La instancia ajustada (para encadenamiento fluido).
        """
        logger.info(
            "Ajustando FactorModel: %d activos, %d factores",
            log_returns.shape[1],
            self.n_factors,
        )
        # TODO: Si corr_matrix es None, calcular correlación empírica
        # TODO: Descomponer corr_matrix con np.linalg.eigh, ordenar descendente
        # TODO: Extraer los K eigenvectores de mayor eigenvalor como loadings
        # TODO: Inicializar self._pca o trabajar con eigenvectores directamente
        raise NotImplementedError("FactorModel.fit() — pendiente de implementación.")

    def transform(self, log_returns: pd.DataFrame) -> FactorModelResult:
        """Proyecta retornos sobre factores y calcula residuos.

        Args:
            log_returns: DataFrame de log-retornos (T, N).

        Returns:
            FactorModelResult con factor_returns, residuals y loadings.

        Raises:
            RuntimeError: Si fit() no fue llamado previamente.
        """
        if self._pca is None:
            raise RuntimeError("Llamá a fit() antes de transform().")
        # TODO: factor_returns = log_returns @ loadings  → (T, K)
        # TODO: reconstruction = factor_returns @ loadings.T  → (T, N)
        # TODO: residuals = log_returns - reconstruction  → (T, N)
        raise NotImplementedError("FactorModel.transform() — pendiente de implementación.")

    def fit_transform(self, log_returns: pd.DataFrame, corr_matrix: Optional[np.ndarray] = None) -> FactorModelResult:
        """Ajusta y transforma en un solo paso.

        Args:
            log_returns: DataFrame de log-retornos (T, N).
            corr_matrix: Matriz de correlación denoised (N, N). Opcional.

        Returns:
            FactorModelResult completo.
        """
        return self.fit(log_returns, corr_matrix).transform(log_returns)

    def get_zscore_signals(
        self,
        residuals: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Calcula z-scores rolling de los residuos como señales de entrada.

        Un z-score extremo (|z| > threshold) indica reversión potencial.

        Args:
            residuals: DataFrame de residuos idiosincráticos (T, N).
            window: Ventana rolling para calcular media y desvío estándar.

        Returns:
            DataFrame de z-scores con la misma forma (T, N).
        """
        # TODO: z = (residual - rolling_mean) / rolling_std
        # TODO: Manejar casos de std ~ 0 con np.where o .replace
        raise NotImplementedError("get_zscore_signals() — pendiente de implementación.")

    @property
    def result(self) -> Optional[FactorModelResult]:
        """Último resultado del transform (disponible tras transform())."""
        return self._result
