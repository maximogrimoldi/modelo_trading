"""
Entry point del modelo de trading RMT.
Descarga datos, calcula residuos rolling, genera señales y guarda los resultados.

Gestión de estado entre corridas
──────────────────────────────────
posiciones_abiertas.csv persiste el portafolio actual entre ejecuciones.
Cada vez que corrés main.py:
  - Señal nueva            → se calcula la inversión sugerida y se agrega al estado.
  - Ya en cartera          → se muestra el retorno no realizado desde la entrada.
  - En cartera pero z-score cruzó exit_threshold → se marca como "CERRAR" y se elimina del estado.
"""

import os
import numpy as np
import pandas as pd

from data.loader import DataLoader
from strategy.signals import RMTStrategy


class TradingRunner:
    """
    Orquesta el pipeline: datos → residuos rolling → z-scores → señales → outputs.

    Parámetros
    ----------
    capital          : float — capital total disponible para sizing (default: 100 000).
    position_fraction: float — fracción del capital a desplegar por barra (default: 0.20).
                               Se distribuye proporcionalmente al |z-score| de cada señal nueva.
    rf               : float — tasa libre de riesgo anual (default: 0.0).
    ventana_betas    : int   — días para estimar factores RMT (default: 252).
    ventana_zscore   : int   — días de historial para el z-score (default: 252).
    entry_threshold  : float — z-score para abrir posición (default: 2.0).
    exit_threshold   : float — z-score para cerrar posición (default: 0.5).
    results_path     : str   — carpeta donde se guardan los outputs (default: "results/").
    """

    def __init__(
        self,
        capital=100_000.0,
        position_fraction=0.20,
        rf=0.0,
        ventana_betas=252,
        ventana_zscore=252,
        entry_threshold=2.0,
        exit_threshold=0.5,
        results_path="results/",
    ):
        self.capital          = capital
        self.position_fraction = position_fraction
        self.rf               = rf
        self.ventana_betas    = ventana_betas
        self.ventana_zscore   = ventana_zscore
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.results_path     = results_path
        self.ruta_estado      = os.path.join(results_path, "posiciones_abiertas.csv")

    # ── Estado persistido ─────────────────────────────────────────────────────

    def _cargar_estado(self) -> pd.DataFrame:
        """Carga el portafolio actual desde disco. Devuelve DataFrame vacío si no existe."""
        if os.path.exists(self.ruta_estado):
            return pd.read_csv(self.ruta_estado, parse_dates=["fecha_entrada"])
        cols = ["ticker", "lado", "precio_entrada", "fecha_entrada",
                "zscore_entrada", "inversion_usd"]
        return pd.DataFrame(columns=cols)

    def _guardar_estado(self, estado: pd.DataFrame) -> None:
        estado.to_csv(self.ruta_estado, index=False)

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _sizing(self, nuevas: list, zs: pd.Series) -> dict:
        """
        Distribuye position_fraction × capital entre las señales nuevas
        proporcionalmente al |z-score|.
        """
        if not nuevas:
            return {}
        total = self.capital * self.position_fraction
        pesos = {t: abs(float(zs[t])) for t in nuevas}
        suma  = sum(pesos.values())
        return {t: round(total * pesos[t] / suma, 2) for t in nuevas}

    # ── Pipeline principal ────────────────────────────────────────────────────

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

        # 3. Z-scores
        residuos_validos = residuos.dropna(how="all").iloc[-self.ventana_zscore:]
        acum = np.cumsum(residuos_validos.values, axis=0)
        zs   = pd.Series(strategy.zscore(acum), index=residuos_validos.columns)

        # 4. Señales del día
        long_list  = set(zs[zs < -strategy.entry_threshold].index)
        short_list = set(zs[zs >  strategy.entry_threshold].index)
        señales_hoy = long_list | short_list

        # 5. Cargar estado anterior
        estado = self._cargar_estado()
        en_cartera = set(estado["ticker"]) if not estado.empty else set()

        nuevas_long  = [t for t in long_list  if t not in en_cartera]
        nuevas_short = [t for t in short_list if t not in en_cartera]
        nuevas = nuevas_long + nuevas_short

        # Posiciones que deben cerrarse (en cartera pero z-score cruzó exit)
        a_cerrar = []
        for _, row in estado.iterrows():
            t = row["ticker"]
            if t not in zs.index:
                continue
            z = float(zs[t])
            if row["lado"] == "long"  and z > -strategy.exit_threshold:
                a_cerrar.append(t)
            if row["lado"] == "short" and z <  strategy.exit_threshold:
                a_cerrar.append(t)

        # 6. Sizing para señales nuevas
        inversiones = self._sizing(nuevas, zs)

        # 7. Construir output del día
        filas = []

        # Señales nuevas
        for ticker in nuevas:
            lado   = "long" if ticker in long_list else "short"
            precio = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            filas.append({
                "ticker":              ticker,
                "tipo":                "NUEVA",
                "lado":                lado,
                "precio_actual":       round(precio, 2) if precio else None,
                "zscore_hoy":          round(float(zs[ticker]), 4),
                "inversion_sugerida":  inversiones[ticker],
                "precio_entrada":      None,
                "fecha_entrada":       None,
                "retorno_no_realizado": None,
            })

        # Posiciones en cartera (activas, sin cerrar)
        for _, row in estado.iterrows():
            ticker = row["ticker"]
            if ticker in a_cerrar:
                continue
            if ticker not in zs.index:
                continue
            precio_actual = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            precio_entrada = float(row["precio_entrada"])
            if precio_actual and precio_entrada:
                ret = (precio_actual - precio_entrada) / precio_entrada * 100
                if row["lado"] == "short":
                    ret = -ret
            else:
                ret = None
            filas.append({
                "ticker":               ticker,
                "tipo":                 "EN_CARTERA",
                "lado":                 row["lado"],
                "precio_actual":        round(precio_actual, 2) if precio_actual else None,
                "zscore_hoy":           round(float(zs[ticker]), 4),
                "inversion_sugerida":   None,
                "precio_entrada":       round(precio_entrada, 2),
                "fecha_entrada":        str(row["fecha_entrada"])[:10],
                "retorno_no_realizado": round(ret, 2) if ret is not None else None,
            })

        # Posiciones a cerrar
        for _, row in estado[estado["ticker"].isin(a_cerrar)].iterrows():
            ticker = row["ticker"]
            precio_actual = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            precio_entrada = float(row["precio_entrada"])
            if precio_actual and precio_entrada:
                ret = (precio_actual - precio_entrada) / precio_entrada * 100
                if row["lado"] == "short":
                    ret = -ret
            else:
                ret = None
            filas.append({
                "ticker":               ticker,
                "tipo":                 "CERRAR",
                "lado":                 row["lado"],
                "precio_actual":        round(precio_actual, 2) if precio_actual else None,
                "zscore_hoy":           round(float(zs[ticker]), 4),
                "inversion_sugerida":   None,
                "precio_entrada":       round(precio_entrada, 2),
                "fecha_entrada":        str(row["fecha_entrada"])[:10],
                "retorno_no_realizado": round(ret, 2) if ret is not None else None,
            })

        # 8. Guardar posiciones_hoy.csv
        df_hoy = pd.DataFrame(filas)
        ruta_pos = os.path.join(self.results_path, "posiciones_hoy.csv")
        df_hoy.to_csv(ruta_pos, index=False)

        # 9. Actualizar estado persistido
        # Eliminar posiciones cerradas
        estado = estado[~estado["ticker"].isin(a_cerrar)]
        # Agregar posiciones nuevas
        nuevas_filas = []
        for ticker in nuevas:
            lado   = "long" if ticker in long_list else "short"
            precio = float(precios[ticker].iloc[-1]) if ticker in precios.columns else None
            nuevas_filas.append({
                "ticker":        ticker,
                "lado":          lado,
                "precio_entrada": precio,
                "fecha_entrada":  fecha_hoy.date(),
                "zscore_entrada": round(float(zs[ticker]), 4),
                "inversion_usd":  inversiones[ticker],
            })
        if nuevas_filas:
            estado = pd.concat([estado, pd.DataFrame(nuevas_filas)], ignore_index=True)
        self._guardar_estado(estado)

        # 10. Print resumen
        print(f"\n── Señales al {fecha_hoy.date()} ──────────────────────────────────")
        nuevas_df  = df_hoy[df_hoy["tipo"] == "NUEVA"]
        cartera_df = df_hoy[df_hoy["tipo"] == "EN_CARTERA"]
        cerrar_df  = df_hoy[df_hoy["tipo"] == "CERRAR"]

        if not nuevas_df.empty:
            print(f"\n  NUEVAS ({len(nuevas_df)}):")
            for _, r in nuevas_df.iterrows():
                print(f"    {r['lado'].upper():<5}  {r['ticker']:<8}  z={r['zscore_hoy']:+.2f}"
                      f"  →  invertir ${r['inversion_sugerida']:,.0f}")
        if not cerrar_df.empty:
            print(f"\n  CERRAR ({len(cerrar_df)}):")
            for _, r in cerrar_df.iterrows():
                signo = "+" if r["retorno_no_realizado"] and r["retorno_no_realizado"] > 0 else ""
                ret_str = f"{signo}{r['retorno_no_realizado']:.2f}%" if r["retorno_no_realizado"] is not None else "N/A"
                print(f"    {r['lado'].upper():<5}  {r['ticker']:<8}  z={r['zscore_hoy']:+.2f}"
                      f"  retorno: {ret_str}")
        if not cartera_df.empty:
            print(f"\n  EN CARTERA ({len(cartera_df)}):")
            for _, r in cartera_df.iterrows():
                signo = "+" if r["retorno_no_realizado"] and r["retorno_no_realizado"] > 0 else ""
                ret_str = f"{signo}{r['retorno_no_realizado']:.2f}%" if r["retorno_no_realizado"] is not None else "N/A"
                print(f"    {r['lado'].upper():<5}  {r['ticker']:<8}  z={r['zscore_hoy']:+.2f}"
                      f"  retorno: {ret_str}  (entrada: ${r['precio_entrada']:,.2f})")
        if nuevas_df.empty and cerrar_df.empty and cartera_df.empty:
            print("  Sin señales hoy.")

        print(f"\n  Guardado: {ruta_pos}")
        print(f"  Estado actualizado: {self.ruta_estado}")

        # 11. Plot de autovalores
        if autovalores is not None:
            ruta_plot = os.path.join(self.results_path, "autovalores_plot.png")
            strategy.plot_autovalores(autovalores, lambda_max, ruta_plot)
            print(f"  Guardado: {ruta_plot}")


# Ejemplos de uso
example = TradingRunner(capital=100_000, position_fraction=0.20, rf=0.04)
example.run()

