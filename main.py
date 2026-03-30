"""
Entry point del modelo de trading RMT.
Descarga datos, calcula residuos rolling, genera señales y guarda los resultados.
"""

import os
import numpy as np
import pandas as pd

from data.loader import DataLoader
from strategy.signals import RMTStrategy


class TradingRunner:
    """
    Orquesta el pipeline: datos → residuos rolling → z-scores → señales → outputs.
    """

    def __init__(
        self,
        rf=0.0,
        ventana_betas=252,
        ventana_zscore=252,
        entry_threshold=2.0,
        exit_threshold=0.5,
        results_path="results/",
    ):
        self.rf               = rf
        self.ventana_betas    = ventana_betas
        self.ventana_zscore   = ventana_zscore
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.results_path     = results_path

    def run(self):
        os.makedirs(self.results_path, exist_ok=True)
        strategy = RMTStrategy(
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
        )

        # 1. Bajar datos
        print("Descargando datos...")
        loader    = DataLoader(periodo=3, unidad="años", frecuencia="diaria")
        retornos  = loader.get_returns()
        precios   = loader.get_prices()
        fecha_hoy = retornos.index[-1]
        print(f"  {loader}")
        print(f"  Tickers activos: {retornos.shape[1]}  |  Fechas: {retornos.shape[0]}")

        # 2. Calcular residuos rolling sin lookahead bias → guardar
        print(f"\nCalculando residuos rolling (ventana_betas={self.ventana_betas} días)...")
        residuos, autovalores, lambda_max = strategy.calcular_residuos_rolling(
            retornos, ventana=self.ventana_betas
        )
        ruta_residuos = os.path.join(self.results_path, "residuos_historicos.csv")
        residuos.to_csv(ruta_residuos)
        print(f"  Guardado: {ruta_residuos}  ({residuos.dropna(how='all').shape[0]} días con residuos)")

        # 3. Z-scores sobre los últimos ventana_zscore días de residuos válidos
        # Ventana separada de ventana_betas: podés estimar betas con historia larga
        # y aun así calcular el z-score sobre un período más reciente y relevante
        residuos_validos = residuos.dropna(how="all").iloc[-self.ventana_zscore:]
        acum = np.cumsum(residuos_validos.values, axis=0)
        zs   = pd.Series(strategy.zscore(acum), index=residuos_validos.columns)

        # 4. Generar señales con entry_threshold
        long_list  = list(zs[zs < -strategy.entry_threshold].index)
        short_list = list(zs[zs >  strategy.entry_threshold].index)

        print(f"\n── Señales al {fecha_hoy.date()} ──────────────────────────────────")
        if long_list or short_list:
            todas = sorted(long_list + short_list, key=lambda t: abs(zs[t]), reverse=True)
            for t in todas:
                lado = "LONG " if t in long_list else "SHORT"
                print(f"  {lado}  {t:<8}  z = {zs[t]:+.2f}")
        else:
            print("  Sin señales hoy.")

        # 5. Guardar posiciones_hoy.csv
        filas = []
        for ticker in long_list:
            precio = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            filas.append({"ticker": ticker, "lado": "long",  "precio_entrada": precio,
                          "fecha": fecha_hoy.date(), "zscore": round(float(zs[ticker]), 4)})
        for ticker in short_list:
            precio = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            filas.append({"ticker": ticker, "lado": "short", "precio_entrada": precio,
                          "fecha": fecha_hoy.date(), "zscore": round(float(zs[ticker]), 4)})

        ruta_pos = os.path.join(self.results_path, "posiciones_hoy.csv")
        pd.DataFrame(filas).to_csv(ruta_pos, index=False)
        print(f"\n  Guardado: {ruta_pos}")

        # 6. Guardar plot de autovalores
        if autovalores is not None:
            ruta_plot = os.path.join(self.results_path, "autovalores_plot.png")
            strategy.plot_autovalores(autovalores, lambda_max, ruta_plot)
            print(f"  Guardado: {ruta_plot}")



# Ejemplos de uso
example = TradingRunner(rf=0.04)
example.run()

