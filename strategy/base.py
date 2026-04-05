"""
Interfaz base para estrategias de trading multi-ticker.

─────────────────────────────────────────────────────────────────────────────
CÓMO CREAR UNA NUEVA ESTRATEGIA
─────────────────────────────────────────────────────────────────────────────
1. Creá un archivo nuevo (ej. rmt_strategy.py) en la raíz del proyecto.
2. Importá Strategy desde este módulo:
       from strategy.base import Strategy
3. Subclasificá Strategy e implementá generate_signals().
4. Pasá una instancia al BacktestEngine en run.py.

No modificás ningún otro archivo del motor.

Ejemplo mínimo:
──────────────────────────────────────────────────────────────────────────────
    from strategy.base import Strategy
    import pandas as pd

    class MiEstrategia(Strategy):
        def generate_signals(self, prices, open_positions):
            # prices: DataFrame (fechas × tickers) con precios de cierre
            # open_positions: {"long": [...], "short": [...]}
            return {
                "long":         ["AAPL"],   # abrir estos longs
                "short":        ["MSFT"],   # abrir estos shorts
                "cerrar_long":  [],         # cerrar estos longs abiertos
                "cerrar_short": [],         # cerrar estos shorts abiertos
            }
──────────────────────────────────────────────────────────────────────────────
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Contrato que toda estrategia debe cumplir para operar con BacktestEngine.

    El engine llama a generate_signals() en cada barra, pasando el historial
    de precios disponible hasta ese momento y las posiciones actualmente
    abiertas. La estrategia decide qué abrir y qué cerrar.
    """

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame, open_positions: dict) -> dict:
        """
        Genera señales de apertura y cierre para la barra actual.

        Parámetros
        ----------
        prices : pd.DataFrame
            Precios de cierre (DatetimeIndex × tickers).
            Contiene solo las barras hasta la actual (inclusive).
            Sin lookahead: el día siguiente no está en este DataFrame.

        open_positions : dict
            Estado actual del portafolio, antes de ejecutar cualquier trade:
            {
                "long":  ["AAPL", "MSFT", ...],   # tickers con posición larga abierta
                "short": ["GOOGL", ...],           # tickers con posición corta abierta
            }

        Retorna
        -------
        dict con cuatro listas:
            "long"         → tickers a los que abrir posición larga.
            "short"        → tickers a los que abrir posición corta.
            "cerrar_long"  → tickers con long abierto que hay que cerrar.
            "cerrar_short" → tickers con short abierto que hay que cerrar.

        Reglas:
        - Un ticker en "long" o "short" que ya esté en open_positions se ignora.
        - Un ticker en "cerrar_long" o "cerrar_short" que no esté abierto se ignora.
        - Las listas pueden estar vacías — el engine lo maneja.
        """
        ...
