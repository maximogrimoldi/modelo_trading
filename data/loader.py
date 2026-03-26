"""
DataLoader: descarga y limpia precios de Yahoo Finance.
Solo trae datos — no sabe nada de la estrategia.
"""

import yfinance as yf
import pandas as pd
from datetime import timedelta


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


class DataLoader:
    """
    Parámetros
    ----------
    tickers     : lista de tickers (default: S&P 100)
    lookback    : días hacia atrás desde hoy (default: 756 ≈ 3 años)
    start_date  : fecha de inicio explícita (sobreescribe lookback si se pasa)
    end_date    : fecha de fin (default: hoy)
    fill_method : cómo llenar huecos internos — 'ffill' o None 
    min_coverage: fracción mínima de datos válidos por ticker (default: 0.9)
    """

    def __init__(
        self,
        tickers=None,
        lookback=756,
        start_date=None,
        end_date=None,
        fill_method="ffill",
        min_coverage=0.9,
    ):
        
        self.tickers = tickers or SP100_TICKERS
        self.fill_method = fill_method
        self.min_coverage = min_coverage

        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
        if start_date:
            self.start_date = pd.Timestamp(start_date)
        else:
            self.start_date = self.end_date - timedelta(days=lookback)

    # ------------------------------------------------------------------
    def get_returns(self):
        """Descarga precios, limpia y devuelve retornos diarios."""
        prices = self.bajar_precios()
        prices = self.limpiar_datos(prices)
        returns = self.calcular_retornos(prices)
        return returns

    def get_prices(self):
        """Igual que get_returns() pero devuelve precios ajustados (no retornos)."""
        prices = self.bajar_precios()
        return self.limpiar_datos(prices)

    # ------------------------------------------------------------------
    def bajar_precios(self):
        raw = yf.download(
            self.tickers,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        # yfinance devuelve MultiIndex cuando hay varios tickers
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})

        return prices

    def limpiar_datos(self, prices):
        # 1. Solo días de semana (saca feriados que yf ya filtra, pero por las dudas)
        prices = prices[prices.index.dayofweek < 5]

        # 2. Llenar huecos internos hacia adelante
        if self.fill_method == "ffill":
            prices = prices.ffill()

        # 3. Tirar tickers con cobertura insuficiente
        coverage = prices.notna().mean()
        good_tickers = coverage[coverage >= self.min_coverage].index
        dropped = set(prices.columns) - set(good_tickers)
        if dropped:
            print(f"[DataLoader] Tickers descartados por baja cobertura: {sorted(dropped)}")
        prices = prices[good_tickers]

        # 4. Tirar filas donde todos son NaN (días sin mercado)
        prices = prices.dropna(how="all")

        return prices

    def calcular_retornos(self, prices):
        return prices.pct_change().dropna(how="all")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DataLoader(tickers={len(self.tickers)}, "
            f"desde={self.start_date.date()}, hasta={self.end_date.date()})"
        )

