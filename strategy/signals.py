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
    zscore_entry : umbral de z-score para abrir posición (default: 2.0)
    """

    def __init__(
            self, 
            zscore_entry=2.0
            ):
        
        self.zscore_entry = zscore_entry

    # ------------------------------------------------------------------
    def get_signals(self, retornos):
        """
        Recibe un DataFrame de retornos (filas=fechas, columnas=tickers).
        Devuelve {"long": [...], "short": [...]}.
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
            return {"long": [], "short": []}

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

        # 9. Señales
        zs = pd.Series(zscore, index=tickers)
        long_list  = list(zs[zs < -self.zscore_entry].index)
        short_list = list(zs[zs >  self.zscore_entry].index)

        return {"long": long_list, "short": short_list}

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"RMTStrategy(zscore_entry={self.zscore_entry})"

