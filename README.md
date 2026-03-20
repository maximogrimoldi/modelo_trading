# Factor Arbitraje RMT — Motor Cuantitativo

Motor de **Factor Arbitraje Estadístico** basado en **Random Matrix Theory (RMT)** y **Detección de Saltos de Lee-Mykland**. Diseñado para operar en universos de ~100 activos con un pipeline de señales riguroso y backtesting walk-forward.

---

## Arquitectura del Motor

```
┌──────────────────────────────────────────────────────────┐
│                      PIPELINE                            │
│                                                          │
│  [DataIngestion]  →  [RMTCleaner]  →  [FactorModel]     │
│       ↓                   ↓                ↓             │
│  Precios OHLC      Corr. denoised    Eigenportfolios     │
│  Log-retornos      Marchenko-Pastur  + Residuos          │
│                                         ↓                │
│                               [JumpDetector]             │
│                                Lee-Mykland               │
│                                         ↓                │
│                               [BacktestEngine]           │
│                               Walk-Forward + PnL         │
└──────────────────────────────────────────────────────────┘
```

---

## Descripción Técnica de los Módulos

### 1. `DataIngestion` — `src/data_io/ingestion.py`

Gestiona la ingesta y persistencia de datos de mercado.

- Descarga precios de cierre ajustados via **yfinance**.
- Caché incremental en formato **Parquet** (Snappy) para minimizar llamadas a la API.
- Limpieza automática: elimina tickers con más del 20 % de NaN y aplica forward-fill.
- Calcula **log-retornos** diarios: $r_t = \ln(P_t / P_{t-1})$.

### 2. `RMTCleaner` — `src/models/rmt_cleaner.py`

Elimina el ruido aleatorio de la matriz de correlación empírica usando la **Ley de Marchenko-Pastur**.

**Fundamento teórico:**
Dada una matriz de retornos $X \in \mathbb{R}^{T \times N}$ con $q = T/N$, la distribución de eigenvalores de una matriz de Wishart aleatoria tiene soporte:

$$\lambda^\pm = \sigma^2 \left(1 \pm \frac{1}{\sqrt{q}}\right)^2$$

Eigenvalores $\lambda_i > \lambda^+$ contienen señal genuina; el resto es ruido.

**Pasos:**
1. Calcular $C = \frac{1}{T} X^T X$ (correlación empírica).
2. Descomponer: $C = V \Lambda V^T$ con `np.linalg.eigh`.
3. Estimar $\sigma^2$ y $\lambda^+$ ajustando la PDF de Marchenko-Pastur.
4. Reemplazar eigenvalores de ruido preservando la traza: $C_{clean} = V \tilde{\Lambda} V^T$.

### 3. `FactorModel` — `src/models/factor_model.py`

Construye **eigenportfolios** ortogonales y calcula los **residuos idiosincráticos**.

**Modelo:**

$$r_t = B f_t + \varepsilon_t$$

Donde:
- $B \in \mathbb{R}^{N \times K}$: matriz de cargas (loadings), columnas = eigenvectores de señal.
- $f_t = B^T r_t \in \mathbb{R}^K$: retornos de los K eigenportfolios.
- $\varepsilon_t = r_t - B f_t$: residuos idiosincráticos (señal de trading).

Las señales surgen de z-scores rolling de $\varepsilon_t$: valores extremos ($|z| > 2$) sugieren reversión a la media estadísticamente significativa.

### 4. `JumpDetector` — `src/models/jump_detector.py`

Implementa el test no paramétrico de **Lee & Mykland (2008)** para identificar saltos de precio que contaminarían las señales de reversión.

**Estadístico:**

$$L(t) = \frac{r_t}{\widehat{BPV}_t^{1/2}}$$

Donde la **Variación Bipower** local estima la varianza del componente continuo:

$$\widehat{BPV}_t = \frac{\pi}{2} \sum_{i=t-W}^{t} |r_i| \cdot |r_{i-1}|$$

Se rechaza la hipótesis nula (sin salto) si $|L(t)|$ excede el cuantil de la distribución de máximos de normales estándar. Las observaciones con salto detectado son excluidas del cálculo de señales.

### 5. `BacktestEngine` — `src/backtest/engine.py`

Motor de backtesting **walk-forward** vectorizado.

**Flujo:**
1. Generar fechas de rebalanceo (mensual / trimestral).
2. Por cada período: ajustar RMT + FactorModel en ventana de entrenamiento.
3. Calcular z-scores out-of-sample y filtrar saltos.
4. Generar posiciones con regla de entrada/salida basada en z-score.
5. Calcular retornos del portafolio descontando costos de transacción.

**Métricas reportadas:** Sharpe, Sortino, Max Drawdown, CAGR, Calmar Ratio.

---

## Estructura del Proyecto

```
Modelo de Trading/
├── config/
│   ├── config.yaml          # Parámetros del pipeline y backtest
│   └── tickers.txt          # Universo de activos (un ticker por línea)
├── data/
│   ├── raw/                 # Datos crudos (excluidos de git)
│   ├── processed/           # Datos procesados (excluidos de git)
│   └── cache/               # Caché Parquet (excluido de git)
├── notebooks/               # Exploración y análisis (Jupyter)
├── src/
│   ├── backtest/
│   │   └── engine.py        # Motor walk-forward
│   ├── data_io/
│   │   └── ingestion.py     # Descarga y caché de precios
│   ├── models/
│   │   ├── rmt_cleaner.py   # Denoising Marchenko-Pastur
│   │   ├── jump_detector.py # Test Lee-Mykland
│   │   └── factor_model.py  # PCA + Eigenportfolios + Residuos
│   └── utils/
│       └── helpers.py       # Logger y carga de config
├── tests/                   # Tests unitarios e integración
├── main.py                  # Entry point (argparse)
├── requirements.txt
└── .gitignore
```

---

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

```bash
# Ejecutar backtest completo
python main.py backtest --config config/config.yaml

# Forzar re-descarga de datos
python main.py backtest --config config/config.yaml --refresh-data

# Escanear señales activas (top 10)
python main.py scan --config config/config.yaml --top-n 10

# Cambiar nivel de logging
python main.py --log-level DEBUG backtest
```

---

## Referencias

- Marchenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices.*
- Laloux, L. et al. (1999). *Noise Dressing of Financial Correlation Matrices.* PRL.
- Plerou, V. et al. (2002). *Random matrix approach to cross correlations in financial data.* PRE.
- Lee, S. S. & Mykland, P. A. (2008). *Jumps in financial markets: A new nonparametric test and jump dynamics.* RFS.
- Avellaneda, M. & Lee, J. H. (2010). *Statistical arbitrage in the US equities market.* QF.
