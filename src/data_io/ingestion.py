"""Módulo de ingesta de datos de mercado.

Responsable de descargar, validar y persistir precios ajustados
de los activos del universo de trading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataIngestion:
    """Descarga y gestiona series de precios para el universo de activos.

    Soporta descarga incremental (sólo datos faltantes) y caché en formato
    Parquet para minimizar llamadas a la API.

    Attributes:
        tickers: Lista de símbolos del universo de activos.
        cache_dir: Directorio donde se persisten los datos en Parquet.
        start_date: Fecha de inicio del período de análisis (YYYY-MM-DD).
        end_date: Fecha de fin del período de análisis (YYYY-MM-DD).
    """

    def __init__(
        self,
        tickers: List[str],
        cache_dir: Path,
        start_date: str,
        end_date: str,
    ) -> None:
        """Inicializa la instancia de DataIngestion.

        Args:
            tickers: Lista de tickers (ej. ['AAPL', 'MSFT', ...]).
            cache_dir: Ruta al directorio de caché (data/cache/).
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
            end_date: Fecha de fin en formato 'YYYY-MM-DD'.
        """
        self.tickers = tickers
        self.cache_dir = Path(cache_dir)
        self.start_date = start_date
        self.end_date = end_date
        self._cache_file = self.cache_dir / "prices.parquet"

    def fetch(self, force_refresh: bool = False) -> pd.DataFrame:
        """Descarga o carga desde caché los precios de cierre ajustados.

        Args:
            force_refresh: Si es True, ignora el caché y re-descarga todo.

        Returns:
            DataFrame con forma (T, N) donde T son días de trading y
            N la cantidad de activos. Índice: DatetimeIndex. Columnas: tickers.

        Raises:
            ValueError: Si la descarga resulta en datos vacíos.
        """
        if self._cache_file.exists() and not force_refresh:
            logger.info("Cargando precios desde caché: %s", self._cache_file)
            return pd.read_parquet(self._cache_file)

        logger.info(
            "Descargando %d activos desde %s hasta %s",
            len(self.tickers),
            self.start_date,
            self.end_date,
        )
        # TODO: Implementar descarga con manejo de errores por ticker
        raw: pd.DataFrame = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]

        if raw.empty:
            raise ValueError("La descarga de datos resultó vacía. Verificá los tickers.")

        prices = self._clean(raw)
        self._persist(prices)
        return prices

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida el DataFrame crudo de precios.

        Args:
            raw: DataFrame crudo devuelto por yfinance.

        Returns:
            DataFrame limpio sin columnas con más del 20 % de NaN.
        """
        # TODO: Implementar lógica de limpieza avanzada
        threshold = 0.20
        valid_cols = raw.columns[raw.isna().mean() < threshold]
        dropped = set(raw.columns) - set(valid_cols)
        if dropped:
            logger.warning("Tickers eliminados por exceso de NaN: %s", dropped)
        return raw[valid_cols].ffill().dropna()

    def _persist(self, prices: pd.DataFrame) -> None:
        """Persiste el DataFrame de precios en formato Parquet.

        Args:
            prices: DataFrame limpio a guardar en disco.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(self._cache_file, engine="pyarrow", compression="snappy")
        logger.info("Precios persistidos en %s", self._cache_file)

    def get_log_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calcula log-retornos diarios a partir de los precios.

        Args:
            prices: DataFrame de precios. Si es None, llama a fetch().

        Returns:
            DataFrame de log-retornos con forma (T-1, N).
        """
        if prices is None:
            prices = self.fetch()
        # TODO: Añadir opción de frecuencia (diaria, semanal, etc.)
        return prices.apply(lambda col: col.pct_change().add(1).pipe(__import__("numpy").log)).dropna()
