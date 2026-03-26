"""
RMTStrategy: genera señales de trading usando Random Matrix Theory.
Recibe retornos, devuelve señales. No sabe nada de dinero ni fechas.
"""

import numpy as np
import pandas as pd


class RMTStrategy:
    """
    Parámetros
    ----------
    entry_threshold : abrís posición cuando |z-score| supera este valor (default: 2.0) --> a calibrar con Backtest
    exit_threshold  : cerrás posición cuando |z-score| vuelve a este valor (default: 0.5) --> a calibrar con Backtest

    Lógica:
        z > +entry  → short (abrir)
        z < +exit   → cerrar short

        z < -entry  → long (abrir)
        z > -exit   → cerrar long
    """

    def __init__(
            self,
            entry_threshold=2.0,
            exit_threshold=0.5,
            ):

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    # ------------------------------------------------------------------
    def get_signals(self, retornos, posiciones_abiertas=None):
        """
        Recibe un DataFrame de retornos (filas=fechas, columnas=tickers).

        posiciones_abiertas : dict con las posiciones que ya tenés abiertas.
            Ejemplo: {"long": ["AAPL", "NVDA"], "short": ["WMT"]}
            Si es None (primera vez), solo calcula entradas — no hay nada que cerrar.

        Devuelve {"long": [...], "short": [...], "cerrar_long": [...], "cerrar_short": [...]}.
        """
        R = retornos.values          # (T, N)
        tickers = list(retornos.columns)
        T, N = R.shape

        # 1. Matriz de correlación
        corr = np.corrcoef(R.T)      # (N, N)

        # 2. PCA — autovalores y autovectores
        autovalores, autovectores = np.linalg.eigh(corr)
        # eigh devuelve en orden ascendente — invertir para tener los mayores primero
        idx = np.argsort(autovalores)[::-1]
        autovalores = autovalores[idx]
        autovectores = autovectores[:, idx]  # columnas = autovectores

        # 3. Filtro Marchenko-Pastur
        sigma2 = 1.0   # correlación ya está normalizada
        lambda_max = sigma2 * (1 + np.sqrt(N / T)) ** 2
        factores_idx = autovalores > lambda_max
        V = autovectores[:, factores_idx]    # (N, k) — solo autovectores significativos

        if V.shape[1] == 0:
            return {"long": [], "short": [], "cerrar_long": [], "cerrar_short": []}

        # 4. Series de tiempo de factores: F = R @ V → (T, k)
        F = R @ V

        # 5. Betas por OLS: B = (F'F)^-1 F'R → (k, N)
        FtF_inv = np.linalg.pinv(F.T @ F)
        B = FtF_inv @ F.T @ R        # (k, N)

        # 6. Residuos diarios: e = R - F @ B → (T, N)
        residuos = R - F @ B

        # 7. Residuo acumulado
        residuo_acum = np.cumsum(residuos, axis=0)   # (T, N)

        # 8. Z-score sobre la ventana completa
        media = residuo_acum.mean(axis=0)
        desvio = residuo_acum.std(axis=0)
        desvio[desvio == 0] = np.nan
        zscore = (residuo_acum[-1] - media) / desvio   # último día

        # 9. Señales de entrada y salida
        zs = pd.Series(zscore, index=tickers)

        # Posiciones ya abiertas (vacías si es la primera vez)
        if posiciones_abiertas is None:
            abiertas_long  = set()
            abiertas_short = set()
        else:
            abiertas_long  = set(posiciones_abiertas.get("long",  []))
            abiertas_short = set(posiciones_abiertas.get("short", []))

        ya_en_cartera = abiertas_long | abiertas_short

        # Entradas: solo tickers que no están ya abiertos
        long_list  = list(zs[zs < -self.entry_threshold].drop(index=ya_en_cartera, errors="ignore").index)
        short_list = list(zs[zs >  self.entry_threshold].drop(index=ya_en_cartera, errors="ignore").index)

        # Cierres: solo sobre posiciones que realmente tenés abiertas
        cerrar_long_list  = [t for t in abiertas_long  if zs.get(t, 0) > -self.exit_threshold]
        cerrar_short_list = [t for t in abiertas_short if zs.get(t, 0) <  self.exit_threshold]

        return {
            "long":         long_list,
            "short":        short_list,
            "cerrar_long":  cerrar_long_list,
            "cerrar_short": cerrar_short_list,
        }

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"RMTStrategy(entry_threshold={self.entry_threshold}, "
            f"exit_threshold={self.exit_threshold})"
        )
