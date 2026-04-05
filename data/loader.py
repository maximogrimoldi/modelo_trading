"""
DataLoader: descarga y limpia precios de Yahoo Finance.
Solo trae datos — no sabe nada de la estrategia.
"""

import yfinance as yf
import pandas as pd
from dateutil.relativedelta import relativedelta


SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM",
    "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
    "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
]

FRECUENCIAS = {
    "diaria":  "1d",
    "semanal": "1wk",
    "mensual": "1mo",
}


class DataLoader:
    """
    Parámetros
    ----------
    tickers     : lista de tickers (default: S&P 100)
    periodo     : cuánto hacia atrás. Número entero (default: 3)
    unidad      : "años", "meses" o "dias" (default: "años")
    frecuencia  : "diaria", "semanal" o "mensual" (default: "diaria")
    start_date  : fecha de inicio explícita (sobreescribe periodo/unidad si se pasa)
    end_date    : fecha de fin (default: hoy)
    fill_method : cómo llenar huecos internos — 'ffill' o None
    min_coverage: fracción mínima de datos válidos por ticker (default: 0.9)
    """

    def __init__(
        self,
        tickers=None,
        periodo=3,
        unidad="años",
        frecuencia="diaria",
        start_date=None,
        end_date=None,
        fill_method="ffill",
        min_coverage=0.9,
    ):

        self.tickers = tickers or SP100_TICKERS
        self.fill_method = fill_method
        self.min_coverage = min_coverage

        if frecuencia not in FRECUENCIAS:
            raise ValueError(f"frecuencia debe ser una de: {list(FRECUENCIAS)}")
        self.frecuencia = frecuencia

        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()

        if start_date:
            self.start_date = pd.Timestamp(start_date)
        else:
            self.start_date = self._calcular_inicio(periodo, unidad)

    def _calcular_inicio(self, periodo, unidad):
        unidad = unidad.lower()
        if unidad in ("año", "años"):
            return self.end_date - relativedelta(years=periodo)
        elif unidad in ("mes", "meses"):
            return self.end_date - relativedelta(months=periodo)
        elif unidad in ("dia", "dias", "día", "días"):
            return self.end_date - relativedelta(days=periodo)
        else:
            raise ValueError(f"unidad debe ser 'años', 'meses' o 'dias'. Recibido: '{unidad}'")

    # ------------------------------------------------------------------
    def get_returns(self):
        """Descarga precios, limpia y devuelve retornos."""
        prices = self.bajar_precios()
        prices = self.limpiar_datos(prices)
        returns = self.calcular_retornos(prices)
        return returns

    def get_prices(self):
        """Devuelve precios de cierre ajustados."""
        prices = self.bajar_precios()
        return self.limpiar_datos(prices)

    def get_opens(self):
        """Devuelve precios de apertura ajustados (mismo universo que get_prices)."""
        opens = self.bajar_opens()
        return self.limpiar_datos(opens)

    # ------------------------------------------------------------------
    def _get_raw(self):
        """Descarga y cachea el raw OHLCV para no llamar a yfinance dos veces."""
        if not hasattr(self, "_raw_cache"):
            self._raw_cache = yf.download(
                self.tickers,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval=FRECUENCIAS[self.frecuencia],
                auto_adjust=True,
                progress=False,
            )
        return self._raw_cache

    def bajar_precios(self):
        raw = self._get_raw()
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
        return prices

    def bajar_opens(self):
        raw = self._get_raw()
        if isinstance(raw.columns, pd.MultiIndex):
            opens = raw["Open"]
        else:
            opens = raw[["Open"]].rename(columns={"Open": self.tickers[0]})
        return opens

    def limpiar_datos(self, prices):
        # Solo días de semana (irrelevante en semanal/mensual pero no rompe nada)
        prices = prices[prices.index.dayofweek < 5]

        if self.fill_method == "ffill":
            prices = prices.ffill()

        coverage = prices.notna().mean()
        good_tickers = coverage[coverage >= self.min_coverage].index
        dropped = set(prices.columns) - set(good_tickers)
        if dropped:
            print(f"[DataLoader] Tickers descartados por baja cobertura: {sorted(dropped)}")
        prices = prices[good_tickers]

        prices = prices.dropna(how="all")
        return prices

    def calcular_retornos(self, prices):
        return prices.pct_change().dropna(how="all")

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"DataLoader(tickers={len(self.tickers)}, "
            f"desde={self.start_date.date()}, hasta={self.end_date.date()}, "
            f"frecuencia={self.frecuencia})"
        )


# ─── Helpers para backtesting single-ticker ───────────────────────────────────

def load_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga datos OHLCV de un único ticker desde Yahoo Finance.

    Parámetros
    ----------
    ticker   : str  — símbolo bursátil (ej. "AAPL", "MSFT").
    start    : str  — fecha de inicio en formato "YYYY-MM-DD".
    end      : str  — fecha de fin (default: hoy).
    interval : str  — granularidad yfinance (default: "1d").

    Retorna
    -------
    pd.DataFrame con DatetimeIndex y columnas Open, High, Low, Close, Volume.
    """
    end_date = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    # yfinance a veces devuelve MultiIndex cuando se pasa un solo ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def load_ohlcv_from_csv(filepath: str, date_column: str = "Date") -> pd.DataFrame:
    """
    Carga datos OHLCV desde un CSV local.

    El CSV debe tener columnas: Date, Open, High, Low, Close, Volume.
    La columna de fechas se usa como índice.

    Parámetros
    ----------
    filepath    : str — ruta al archivo CSV.
    date_column : str — nombre de la columna de fechas (default: "Date").

    Retorna
    -------
    pd.DataFrame con DatetimeIndex y columnas Open, High, Low, Close, Volume.
    """
    df = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV debe tener columnas {required}. Faltan: {missing}")

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
