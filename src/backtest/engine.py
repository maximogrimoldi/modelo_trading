"""Motor de backtesting vectorizado para la estrategia de Factor Arbitraje.

Implementa un backtest walk-forward con rebalanceo periódico, aplicando
el pipeline completo: RMTCleaner → FactorModel → JumpDetector → señales → PnL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.models import FactorModel, JumpDetector, RMTCleaner

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Parámetros de configuración del backtest.

    Attributes:
        train_window: Días de entrenamiento para ajustar el modelo (lookback).
        rebalance_freq: Frecuencia de reestimación del modelo ('M', 'Q', 'W').
        n_factors: Número de eigenportfolios a retener.
        zscore_entry: Z-score de entrada (apertura de posición).
        zscore_exit: Z-score de salida (cierre de posición).
        transaction_cost_bps: Costo de transacción en puntos básicos.
        max_leverage: Apalancamiento máximo del portafolio.
    """

    train_window: int = 252
    rebalance_freq: str = "M"
    n_factors: int = 5
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    transaction_cost_bps: float = 5.0
    max_leverage: float = 1.0


@dataclass
class BacktestResult:
    """Resultado completo del backtest.

    Attributes:
        portfolio_returns: Serie de retornos diarios del portafolio.
        positions: DataFrame de posiciones diarias por activo (T, N).
        turnover: Serie de turnover diario.
        metrics: Diccionario con métricas de performance (Sharpe, MaxDD, etc.).
        equity_curve: Serie de valor acumulado del portafolio.
    """

    portfolio_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        """Curva de equity acumulada (base 1)."""
        return (1 + self.portfolio_returns).cumprod()


class BacktestEngine:
    """Motor de backtesting walk-forward para Factor Arbitraje RMT.

    Itera sobre ventanas de tiempo solapadas, re-estimando el pipeline
    completo (RMT + Factor Model + Jump Detection) en cada rebalanceo
    y calculando retornos out-of-sample.

    Attributes:
        config: Parámetros del backtest (BacktestConfig).
        rmt_cleaner: Instancia configurada de RMTCleaner.
        factor_model: Instancia configurada de FactorModel.
        jump_detector: Instancia configurada de JumpDetector.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Inicializa el BacktestEngine con la configuración dada.

        Args:
            config: Objeto BacktestConfig con todos los parámetros.
        """
        self.config = config
        self.rmt_cleaner = RMTCleaner()
        self.factor_model = FactorModel(n_factors=config.n_factors)
        self.jump_detector = JumpDetector()

    def run(self, log_returns: pd.DataFrame) -> BacktestResult:
        """Ejecuta el backtest walk-forward completo.

        Args:
            log_returns: DataFrame de log-retornos del universo (T, N).

        Returns:
            BacktestResult con retornos, posiciones y métricas del período.

        Raises:
            ValueError: Si el período es insuficiente para al menos una ventana.
        """
        T, N = log_returns.shape
        if T < self.config.train_window + 21:
            raise ValueError(
                f"Período insuficiente. Necesitás al menos "
                f"{self.config.train_window + 21} días. Disponibles: {T}."
            )

        logger.info("Iniciando backtest: %d activos, %d días", N, T)
        # TODO: Paso 1 — generar fechas de rebalanceo con pd.date_range
        # TODO: Paso 2 — iterar ventanas: [i - train_window : i] para ajuste
        # TODO: Paso 3 — en cada ventana: rmt.fit_transform → factor.fit_transform
        # TODO: Paso 4 — aplicar jump_detector.get_jump_mask para filtrar señales
        # TODO: Paso 5 — calcular z-scores y generar posiciones (señal ÷ N activos)
        # TODO: Paso 6 — aplicar max_leverage y transaction_cost_bps
        # TODO: Paso 7 — calcular retornos out-of-sample y acumular
        raise NotImplementedError("BacktestEngine.run() — pendiente de implementación.")

    def compute_metrics(self, portfolio_returns: pd.Series) -> dict:
        """Calcula métricas estándar de performance.

        Args:
            portfolio_returns: Serie de retornos diarios del portafolio.

        Returns:
            Diccionario con métricas: Sharpe, Sortino, MaxDrawdown, CAGR, Calmar.
        """
        # TODO: Sharpe = mean(r) / std(r) * sqrt(252)
        # TODO: Sortino = mean(r) / std(r[r<0]) * sqrt(252)
        # TODO: MaxDrawdown = max peak-to-trough de la equity curve
        # TODO: CAGR = (equity_curve.iloc[-1]) ** (252 / T) - 1
        raise NotImplementedError("compute_metrics() — pendiente de implementación.")

    def _generate_positions(
        self,
        zscores: pd.DataFrame,
        jump_mask: pd.DataFrame,
        prev_positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Genera el vector de posiciones a partir de z-scores y máscara de saltos.

        Regla de trading:
            - Abrir long (short) cuando z < -zscore_entry (z > +zscore_entry).
            - Cerrar cuando |z| < zscore_exit.
            - Forzar posición = 0 si hay salto detectado en ese activo.

        Args:
            zscores: Z-scores de residuos idiosincráticos (T, N).
            jump_mask: Máscara booleana de saltos (T, N).
            prev_positions: Posiciones del período anterior (T, N).

        Returns:
            DataFrame de posiciones target (T, N) en el rango [-1, 1].
        """
        # TODO: Implementar lógica de señales long/short con hysteresis
        raise NotImplementedError("_generate_positions() — pendiente de implementación.")
