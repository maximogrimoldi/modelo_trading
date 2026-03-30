"""
Backtester: lleva registro de posiciones, calcula P&L por trade y métricas.
No sabe nada de la estrategia — recibe signals y ejecuta la mecánica de la cartera.
"""

import os
import numpy as np
import pandas as pd



class Backtester:
    """
    Gestiona el ciclo de vida de las posiciones y registra el historial de trades.
    Recibe signals de la estrategia y ejecuta apertura, cierre y persistencia.
    """

    def __init__( self, rf=0.0):
        self.rf = rf              

    def retorno_posicion(self, entry_price, exit_price, entry_date, exit_date):
        """Calcula el retorno anualizado de una posición."""
        if entry_price <= 0 or exit_price <= 0:
            raise ValueError("Los precios de entrada y salida deben ser mayores a cero.")
        if exit_date <= entry_date:
            raise ValueError("La fecha de salida debe ser posterior a la fecha de entrada.")

        # Retorno simple
        simple_return = (exit_price - entry_price) / entry_price

        # Duración en años
        duration_years = (exit_date - entry_date).days / 365.25

        # Retorno anualizado
        annualized_return = (1 + simple_return) ** (1 / duration_years) - 1

        return annualized_return
    
    def max_drawdown(self, equity_curve):
        """Calcula el máximo drawdown de una curva de equity."""
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdowns = (cumulative_max - equity_curve) / cumulative_max
        return np.max(drawdowns)
    


