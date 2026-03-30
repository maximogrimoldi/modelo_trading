# Motor de Trading RMT — Guía para Claude

## Qué hace este proyecto

Motor de **statistical arbitrage** basado en **Random Matrix Theory (RMT)**. La idea central: los retornos de un universo de acciones tienen una estructura de correlación con dos partes separables — señal (factores comunes, ej. sector, mercado) y ruido aleatorio puro. RMT nos da un criterio matemático (distribución de Marchenko-Pastur) para separar ambas. Los **residuos** de esa descomposición son señales de reversión a la media.

## Cómo correrlo

```bash
# Activar el entorno virtual
source venv/bin/activate

# Correr el pipeline completo
python main.py
```

La primera vez descarga 3 años de datos del S&P 100, calcula residuos y muestra señales. En corridas posteriores carga el historial acumulado de residuos y actualiza el estado de posiciones.

## Qué hace cada archivo

| Archivo | Responsabilidad |
|---|---|
| `data/loader.py` | Descarga y limpia precios de Yahoo Finance. Solo datos, sin lógica de trading. |
| `strategy/signals.py` | Pipeline RMT completo. Funciones puras por paso + clase `RMTStrategy`. |
| `backtest/engine.py` | Lleva registro de posiciones abiertas, calcula P&L por trade, guarda log, calcula métricas. |
| `main.py` | Orquesta todo: carga historial → corre estrategia → muestra señales → guarda estado. |
| `results/residuos_acum.csv` | Historial persistido de residuos idiosincráticos (fechas × tickers). |
| `results/trades_log.csv` | Registro de todos los trades cerrados con su retorno. |
| `results/posiciones_abiertas.csv` | Estado actual de la cartera — se pasa a la próxima corrida. |

## Efecto de cambiar los thresholds

### `entry_threshold` (default: 2.0)

Define cuántas desviaciones estándar tiene que alejarse el residuo acumulado para abrir una posición.

- **Más alto (ej. 2.5):** menos señales, más selectivo, cada operación tiene una desviación más extrema → mayor expectativa de reversión pero menos trades.
- **Más bajo (ej. 1.5):** más señales, más operaciones, pero con menor margen estadístico → más ruido, más costos de transacción.

Calibrarlo requiere backtest. Un valor demasiado bajo en mercados trending genera pérdidas sostenidas.

### `exit_threshold` (default: 0.5)

Define el z-score al que se cierra la posición. Representa "volvió suficientemente al centro".

- **Más alto (ej. 1.0):** cierra antes de llegar a la media → menor retorno por trade pero menor tiempo de exposición.
- **Más bajo (ej. 0.0):** espera hasta que cruce la media → mayor retorno potencial pero mayor riesgo de que la divergencia continúe.

La asimetría natural: entrar con z=2 y salir con z=0.5 captura la mayor parte del recorrido de reversión.

## Decisiones de diseño

### Por qué RMT en vez de PCA directo

PCA estándar retiene los K primeros componentes sin criterio objetivo para elegir K. RMT resuelve esto: la distribución de Marchenko-Pastur describe exactamente cómo se ven los autovalores de una matriz de correlación **pura ruido** (Wishart aleatoria). Cualquier autovalor que supere el límite superior λ_max contiene señal real; el resto es ruido de estimación.

La consecuencia práctica: con pocas muestras (T) y muchos activos (N), la mayoría de los componentes de PCA son ruido — y usarlos contamina los residuos con correlaciones falsas. RMT filtra eso automáticamente, sin hiperparámetros que calibrar.

### Por qué persistir los residuos

El z-score se calcula sobre el residuo **acumulado** — es una serie de tiempo, no una métrica instantánea. Su significado estadístico mejora con más historia: con 30 días de historia, la media y el desvío son muy inestables; con 2 años, son mucho más robustos.

Si recalculáramos desde cero en cada corrida usando solo la ventana descargada, perderíamos esa historia. Al persistir los residuos acumulamos señal genuina: el z-score de hoy se calcula contra todos los días anteriores, no solo los últimos N días descargados.

El costo es disco (archivo CSV liviano) y la necesidad de mantener la consistencia del modelo entre corridas — si cambiás los factores RMT significativamente, tiene sentido borrar el historial y empezar de cero.
