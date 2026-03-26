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


logger = logging.getLogger(__name__)



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


class BacktestResult:
    """Resultado completo del backtest.

    Attributes:
        portfolio_returns: Serie de retornos diarios del portafolio.
        positions: DataFrame de posiciones diarias por activo (T, N).
        turnover: Serie de turnover diario.
        metrics: Diccionario con métricas de performance (Sharpe, MaxDD, etc.).
        equity_curve: Serie de valor acumulado del portafolio.
    """




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

    