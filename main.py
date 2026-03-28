from data.loader import DataLoader
from strategy.signals import RMTStrategy

# --- 1. Bajar datos ---
# periodo + unidad definen cuánto hacia atrás querés ir.
# frecuencia define el intervalo de cada barra.

# Ejemplos de configuración:
# DataLoader(periodo=3,  unidad="años",  frecuencia="diaria")    # 3 años de datos diarios
# DataLoader(periodo=18, unidad="meses", frecuencia="semanal")   # 18 meses de datos semanales
# DataLoader(periodo=90, unidad="dias",  frecuencia="diaria")    # 90 días de datos diarios
# DataLoader(periodo=5,  unidad="años",  frecuencia="mensual")   # 5 años de datos mensuales

loader = DataLoader(periodo=3, unidad="años", frecuencia="diaria")
retornos = loader.get_returns()

# Cómo se ve retornos:
#             AAPL      MSFT      NVDA  ...
# 2024-01-02  0.0123   -0.0045   0.0201
# 2024-01-03 -0.0056    0.0078  -0.0034
# ...

print(retornos.shape)       # (504, 100) aprox — filas x tickers
print(retornos.head())      # primeras filas

# --- 2. Generar señales ---
strategy = RMTStrategy(entry_threshold=2.0, exit_threshold=0.5)

# ---- EJEMPLO A: primera vez, sin posiciones abiertas ----
# cerrar_long y cerrar_short siempre van a ser listas vacías
signals = strategy.get_signals(retornos)

print("-- Sin posiciones abiertas --")
print("LONG        :", signals["long"])          # tickers a comprar
print("SHORT       :", signals["short"])         # tickers a vender
print("CERRAR LONG :", signals["cerrar_long"])   # [] siempre
print("CERRAR SHORT:", signals["cerrar_short"])  # [] siempre

# ---- EJEMPLO B: ya tenés posiciones abiertas de días anteriores ----
# La estrategia solo evalúa el cierre sobre lo que le pasás
posiciones_abiertas = {
    "long":  ["AAPL", "NVDA"],   # tenés estas compradas
    "short": ["WMT", "KO"],      # tenés estas vendidas
}
signals = strategy.get_signals(retornos, posiciones_abiertas=posiciones_abiertas)

print("\n-- Con posiciones abiertas --")
print("LONG        :", signals["long"])          # nuevas entradas long (no estaban en cartera)
print("SHORT       :", signals["short"])         # nuevas entradas short (no estaban en cartera)
print("CERRAR LONG :", signals["cerrar_long"])   # cuáles de ["AAPL", "NVDA"] hay que cerrar
print("CERRAR SHORT:", signals["cerrar_short"])  # cuáles de ["WMT", "KO"] hay que cerrar
