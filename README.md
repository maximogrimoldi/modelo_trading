# Motor de Trading RMT — Backtesting & Señales

Motor de **statistical arbitrage** basado en **Random Matrix Theory (RMT)** con un motor de backtesting genérico y desacoplado de cualquier estrategia concreta.

---

## Arquitectura

```
modelo_trading/
├── data/
│   ├── loader.py          # DataLoader (multi-ticker) + load_ohlcv / load_ohlcv_from_csv
│   └── __init__.py
├── strategy/
│   ├── base.py            # Interfaz Strategy (ABC) — contrato que toda estrategia cumple
│   ├── signals.py         # RMTStrategy — pipeline RMT completo
│   └── __init__.py
├── backtest/
│   ├── engine.py          # BacktestEngine — loop, portafolio, métricas, gráfico
│   └── __init__.py
├── example_strategy.py    # SMACrossover — ejemplo de implementación de Strategy
├── run.py                 # Entry point: carga datos → estrategia → engine → resultados
├── main.py                # Pipeline RMT completo (generación de señales, no backtest)
└── requirements.txt
```

---

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## Uso rápido — Backtest

```bash
# Correr el backtest con la estrategia de ejemplo (SMA Crossover)
python run.py
```

Esto:
1. Descarga datos OHLCV de AAPL (2020–2024) vía yfinance.
2. Corre la estrategia de cruce de medias móviles.
3. Imprime métricas de performance (retorno, Sharpe, drawdown, win rate, profit factor).
4. Guarda en `results/`: CSV de trades, equity curve y gráfico PNG.

---

## Uso rápido — Señales RMT (pipeline original)

```bash
python main.py
```

Descarga 3 años del S&P 100, calcula residuos rolling y muestra señales long/short.

---

## Cómo crear una nueva estrategia

### 1. Creá el archivo

```python
# rmt_strategy.py  (en la raíz del proyecto)
import pandas as pd
from strategy.base import Strategy

class RMTBacktestStrategy(Strategy):
    """Adaptador de RMTStrategy a la interfaz de backtesting."""

    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, ventana=252):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.ventana = ventana
        # Estado interno (si tu estrategia necesita memoria entre barras)
        self._in_position = False

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        df: OHLCV con historia hasta la barra actual (sin lookahead).
        Retorna "buy", "sell" o "hold".
        """
        # Tu lógica acá — podés usar self._in_position para rastrear estado
        # Ejemplo: calculá z-scores sobre los retornos de df["Close"]
        ...
        return "hold"
```

### 2. Conectala en run.py

En `run.py`, reemplazá el bloque marcado con `──►`:

```python
# Antes (estrategia de ejemplo):
from example_strategy import SMACrossover
strategy = SMACrossover(fast_window=10, slow_window=30)

# Después (tu estrategia):
from rmt_strategy import RMTBacktestStrategy
strategy = RMTBacktestStrategy(entry_threshold=2.0, exit_threshold=0.5)
```

**Eso es todo.** No modificás ningún otro archivo.

---

## Contrato de la interfaz Strategy

```python
from strategy.base import Strategy
import pandas as pd

class MiEstrategia(Strategy):
    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Parámetro
        ─────────
        df : pd.DataFrame
            OHLCV con DatetimeIndex.
            Columnas: Open, High, Low, Close, Volume.
            Contiene solo barras hasta la actual (sin lookahead).

        Retorna
        ───────
        "buy"  → abrir posición larga (ignorado si ya hay posición).
        "sell" → cerrar posición larga (ignorado si no hay posición).
        "hold" → no hacer nada.
        """
        ...
```

---

## Carga de datos

```python
from data.loader import load_ohlcv, load_ohlcv_from_csv

# Desde yfinance
df = load_ohlcv("AAPL", start="2020-01-01", end="2024-12-31")

# Desde CSV local (debe tener columnas: Date, Open, High, Low, Close, Volume)
df = load_ohlcv_from_csv("data/AAPL.csv")
```

---

## Métricas calculadas por el engine

| Métrica | Descripción |
|---|---|
| `total_return_pct` | Retorno total del período (%) |
| `annualized_return_pct` | Retorno anualizado (%) |
| `sharpe_ratio` | Sharpe Ratio anualizado (exceso sobre rf) |
| `max_drawdown_pct` | Máximo drawdown desde pico (%) |
| `calmar_ratio` | Retorno anualizado / Max Drawdown |
| `n_trades` | Número de operaciones completadas |
| `win_rate_pct` | Porcentaje de trades ganadores |
| `profit_factor` | Ganancias brutas / Pérdidas brutas |
| `avg_profit_usd` | Ganancia promedio por trade (USD) |
| `avg_return_pct` | Retorno promedio por trade (%) |
| `avg_duration_days` | Duración promedio de cada trade (días) |

Los costos de transacción (comisión + slippage) se aplican en cada operación:
- **Comisión**: fracción del valor operado (default: 0.1%)
- **Slippage**: el precio de compra sube y el de venta baja (default: 0.05%)

---

## Decisiones de diseño

### Desacoplamiento total
El `BacktestEngine` importa únicamente `strategy.base.Strategy`. Nunca importa ninguna estrategia concreta. Si mañana tenés 10 estrategias distintas, el engine no cambia.

### Sin lookahead bias
En cada barra `i`, la estrategia recibe `df.iloc[:i+1]` — solo el pasado. Nunca tiene acceso a datos futuros.

### Modelo de ejecución
Las operaciones se ejecutan al cierre de la barra que genera la señal, con slippage aplicado:
- Compra: `close × (1 + slippage)`
- Venta: `close × (1 − slippage)`

### Position sizing
El engine implementa una posición simple "all-in": invierte todo el efectivo disponible en cada compra. Para sizing más sofisticado (Kelly, volatility targeting, etc.), extendé `BacktestEngine`.

---

## Referencias

- Marchenko & Pastur (1967). *Distribution of eigenvalues for some sets of random matrices.*
- Laloux et al. (1999). *Noise Dressing of Financial Correlation Matrices.* PRL.
- Avellaneda & Lee (2010). *Statistical arbitrage in the US equities market.* QF.
