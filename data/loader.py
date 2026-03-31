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
        self._prices_cache = None

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

    # ------------------------------------------------------------------
    def bajar_precios(self):
        if self._prices_cache is not None:
            return self._prices_cache

        raw = yf.download(
            self.tickers,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            interval=FRECUENCIAS[self.frecuencia],
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})

        self._prices_cache = prices
        return prices

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
