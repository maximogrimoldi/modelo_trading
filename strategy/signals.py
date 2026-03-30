"""
Señales de trading usando Random Matrix Theory (RMT).
Recibe retornos, devuelve señales. No sabe nada de dinero ni fechas.
"""

import os
import numpy as np
import pandas as pd


class RMTStrategy:
    """
    Genera señales de entrada/salida usando RMT + z-score de residuos.
    Parámetros: entry_threshold (abre posición) y exit_threshold (cierra posición).
    """

    def __init__(self, entry_threshold=2.0, exit_threshold=0.5):
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold


    def correlacion(self, R):
        return np.corrcoef(R.T)  # (N, N)

    def pca(self, corr):
        """eigh asume simetría → más estable que eig para matrices de correlación."""
        autovalores, autovectores = np.linalg.eigh(corr)
        idx = np.argsort(autovalores)[::-1]
        return autovalores[idx], autovectores[:, idx]

    def filtrar_marchenko_pastur(self, autovalores, autovectores, T, N):
        """Los autovalores bajo λ_max son ruido de Wishart puro — se descartan."""
        lambda_max = (1 + np.sqrt(N / T)) ** 2
        return autovectores[:, autovalores > lambda_max]  # (N, k)

    def betas(self, R, V):
        """F = R @ V  →  B = (F'F)^{-1} F'R. Devuelve F (T, k) y B (k, N)."""
        F = R @ V
        B = np.linalg.pinv(F.T @ F) @ F.T @ R
        return F, B

    def residuos(self, R, F, B):
        """Lo que no explican los factores — señal de reversión."""
        return R - F @ B  # (T, N)

    def zscore(self, residuos_acum):
        """
        Z-score del último día respecto a toda la historia acumulada.
        Usar la historia completa (no una ventana fija) es la razón para persistir los residuos.
        """
        media  = residuos_acum.mean(axis=0)
        desvio = residuos_acum.std(axis=0)
        desvio[desvio == 0] = np.nan
        return (residuos_acum[-1] - media) / desvio  # (N,)

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_signals(self, retornos, residuos_historicos=None, posiciones_abiertas=None):
        """
        Recibe retornos (T, N) y opcionalmente residuos históricos acumulados.

        residuos_historicos : DataFrame (fechas × tickers) cargado desde disco.
            Si se pasa, los residuos nuevos se agregan al historial y el z-score
            se calcula sobre toda la historia — no solo la ventana actual.

        posiciones_abiertas : dict {"long": [...], "short": [...]}
            Si es None, solo calcula entradas.

        Devuelve:
            signals  : {"long", "short", "cerrar_long", "cerrar_short"}
            residuos : DataFrame con los residuos del período actual (para persistir)
            zscore   : Series con el z-score de cada ticker
        """
        R       = retornos.values
        tickers = list(retornos.columns)
        T, N    = R.shape

        corr           = self.correlacion(R)
        autovalores, autovectores = self.pca(corr)
        V              = self.filtrar_marchenko_pastur(autovalores, autovectores, T, N)

        if V.shape[1] == 0:
            vacío = {"long": [], "short": [], "cerrar_long": [], "cerrar_short": []}
            return vacío, pd.DataFrame(), pd.Series(dtype=float)

        F, B            = self.betas(R, V)
        residuos_nuevos = self.residuos(R, F, B)

        df_residuos_nuevos = pd.DataFrame(residuos_nuevos, index=retornos.index, columns=tickers)

        # Acumular con el historial si existe — el z-score necesita toda la historia
        if residuos_historicos is not None and not residuos_historicos.empty:
            cols_comunes     = residuos_historicos.columns.intersection(tickers)
            df_para_zscore   = pd.concat([residuos_historicos[cols_comunes], df_residuos_nuevos[cols_comunes]])
        else:
            cols_comunes     = tickers
            df_para_zscore   = df_residuos_nuevos

        acum_array = np.cumsum(df_para_zscore.values, axis=0)
        zs = pd.Series(self.zscore(acum_array), index=df_para_zscore.columns).reindex(tickers)

        # Señales de entrada / salida
        if posiciones_abiertas is None:
            abiertas_long, abiertas_short = set(), set()
        else:
            abiertas_long  = set(posiciones_abiertas.get("long",  []))
            abiertas_short = set(posiciones_abiertas.get("short", []))

        ya_en_cartera = abiertas_long | abiertas_short

        signals = {
            "long":         list(zs[zs < -self.entry_threshold].drop(index=ya_en_cartera, errors="ignore").index),
            "short":        list(zs[zs >  self.entry_threshold].drop(index=ya_en_cartera, errors="ignore").index),
            "cerrar_long":  [t for t in abiertas_long  if zs.get(t, 0) > -self.exit_threshold],
            "cerrar_short": [t for t in abiertas_short if zs.get(t, 0) <  self.exit_threshold],
        }

        return signals, df_residuos_nuevos, zs

    def calcular_residuos_rolling(self, retornos, ventana=252):
        """
        Calcula residuos diarios sin lookahead bias.

        Para cada día t, estima betas usando solo retornos[t-ventana:t] (pasado puro),
        luego proyecta el día t con esos betas para obtener el residuo de ese día.

        Por qué rolling y no ventana fija desde el inicio:
        Los factores del mercado cambian con el tiempo — un régimen de alta volatilidad
        genera eigenvectores distintos a uno de baja volatilidad. Si estimáramos betas
        una sola vez con todos los datos, los residuos del año 1 estarían contaminados
        por la estructura del año 3. Rolling mantiene siempre `ventana` días de historia
        relevante y recalibra el modelo a medida que el mercado cambia.

        Devuelve (residuos_df, autovalores_ultimo, lambda_max_ultimo).
        Los primeros `ventana` días quedan como NaN — no hay historia suficiente.
        """
        R       = retornos.values
        tickers = list(retornos.columns)
        T, N    = R.shape

        residuos           = np.full((T, N), np.nan)
        autovalores_ultimo = None
        lambda_max_ultimo  = None

        for t in range(ventana, T):
            R_train = R[t - ventana : t]  # (ventana, N) — solo pasado, nunca el día t

            corr             = self.correlacion(R_train)
            autovalores, autovectores = self.pca(corr)
            V                = self.filtrar_marchenko_pastur(autovalores, autovectores, ventana, N)

            if V.shape[1] == 0:
                continue

            _, B     = self.betas(R_train, V)
            f_t      = R[t] @ V        # factores del día t proyectados con V del pasado
            residuos[t] = R[t] - f_t @ B

            autovalores_ultimo = autovalores
            lambda_max_ultimo  = (1 + np.sqrt(N / ventana)) ** 2

        residuos_df = pd.DataFrame(residuos, index=retornos.index, columns=tickers)
        return residuos_df, autovalores_ultimo, lambda_max_ultimo

    def plot_autovalores(self, autovalores, lambda_max, path):
        """Guarda un gráfico de barras de los autovalores con el umbral Marchenko-Pastur."""
        import matplotlib.pyplot as plt

        colores = ["green" if v > lambda_max else "lightgray" for v in autovalores]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(autovalores)), autovalores, color=colores, width=1.0)
        ax.axhline(
            lambda_max, color="red", linewidth=1.5, linestyle="--",
            label=f"λ_max Marchenko-Pastur = {lambda_max:.2f}",
        )
        ax.set_xlabel("Autovalor (índice, mayor a menor)")
        ax.set_ylabel("Magnitud")
        ax.set_title("Autovalores de la matriz de correlación — verde = señal, gris = ruido")
        ax.legend()

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def __repr__(self):
        return (
            f"RMTStrategy(entry_threshold={self.entry_threshold}, "
            f"exit_threshold={self.exit_threshold})"
        )
