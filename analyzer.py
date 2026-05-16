"""
Módulo de análisis exploratorio para el dataset Bank Marketing.
Encapsula estadística descriptiva, clasificación de variables y visualizaciones.
"""

from io import StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def clasificar_variables(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Función personalizada: identifica variables numéricas y categóricas.

    Returns:
        Diccionario con listas 'numericas' y 'categoricas' y conteos.
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "numericas": numericas,
        "categoricas": categoricas,
        "conteo_numericas": len(numericas),
        "conteo_categoricas": len(categoricas),
    }


class DataAnalyzer:
    """Encapsula operaciones de EDA sobre un DataFrame de marketing bancario."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.clasificacion = clasificar_variables(self.df)

    @property
    def columnas_numericas(self) -> list[str]:
        return self.clasificacion["numericas"]

    @property
    def columnas_categoricas(self) -> list[str]:
        return self.clasificacion["categoricas"]

    def resumen_info(self) -> dict[str, Any]:
        """Equivalente estructurado a .info() con tipos y nulos."""
        tipos = self.df.dtypes.astype(str).reset_index()
        tipos.columns = ["columna", "tipo_dato"]

        nulos = self.df.isnull().sum().reset_index()
        nulos.columns = ["columna", "valores_nulos"]

        buffer = StringIO()
        self.df.info(buf=buffer)

        return {
            "filas": self.df.shape[0],
            "columnas": self.df.shape[1],
            "tipos": tipos,
            "nulos": nulos,
            "memoria_kb": round(self.df.memory_usage(deep=True).sum() / 1024, 2),
            "info_texto": buffer.getvalue(),
        }

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        return self.df.describe(include="all").T

    def conteo_nulos(self) -> pd.DataFrame:
        total = len(self.df)
        conteo = self.df.isnull().sum()
        porcentaje = (conteo / total * 100).round(2)
        return pd.DataFrame(
            {"conteo": conteo, "porcentaje": porcentaje}
        ).sort_values("conteo", ascending=False)

    def media(self, columna: str) -> float | None:
        if columna in self.columnas_numericas:
            return float(self.df[columna].mean())
        return None

    def mediana(self, columna: str) -> float | None:
        if columna in self.columnas_numericas:
            return float(self.df[columna].median())
        return None

    def moda(self, columna: str) -> Any:
        if columna not in self.df.columns:
            return None
        moda_serie = self.df[columna].mode()
        return moda_serie.iloc[0] if len(moda_serie) > 0 else None

    def figura_valores_faltantes(self) -> plt.Figure:
        nulos = self.conteo_nulos()
        nulos_filtrados = nulos[nulos["conteo"] > 0]

        fig, ax = plt.subplots(figsize=(10, max(4, len(nulos_filtrados) * 0.35)))
        if nulos_filtrados.empty:
            ax.text(0.5, 0.5, "No hay valores nulos", ha="center", va="center")
            ax.set_axis_off()
        else:
            sns.barplot(
                x=nulos_filtrados["conteo"],
                y=nulos_filtrados.index,
                ax=ax,
                palette="viridis",
            )
            ax.set_title("Valores faltantes por columna")
            ax.set_xlabel("Conteo")
            ax.set_ylabel("Columna")
        fig.tight_layout()
        return fig

    def figura_histograma(
        self, columna: str, bins: int = 30, mostrar_kde: bool = False
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            self.df[columna],
            bins=bins,
            kde=mostrar_kde,
            ax=ax,
            color="#2E86AB",
            edgecolor="white",
        )
        media_val = self.media(columna)
        mediana_val = self.mediana(columna)
        if media_val is not None:
            ax.axvline(media_val, color="#E94F37", linestyle="--", label=f"Media: {media_val:.2f}")
        if mediana_val is not None:
            ax.axvline(
                mediana_val, color="#F39C12", linestyle="-.", label=f"Mediana: {mediana_val:.2f}"
            )
        ax.set_title(f"Distribución de {columna}")
        ax.legend()
        fig.tight_layout()
        return fig

    def resumen_categorica(self, columna: str) -> pd.DataFrame:
        conteos = self.df[columna].value_counts(dropna=False)
        proporciones = self.df[columna].value_counts(normalize=True, dropna=False) * 100
        return pd.DataFrame(
            {"conteo": conteos, "proporcion_%": proporciones.round(2)}
        )

    def figura_barras_categorica(
        self, columna: str, top_n: int | None = None, horizontal: bool = False
    ) -> plt.Figure:
        conteos = self.df[columna].value_counts()
        if top_n:
            conteos = conteos.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 5))
        if horizontal:
            conteos.plot(kind="barh", ax=ax, color="#44AF69")
        else:
            conteos.plot(kind="bar", ax=ax, color="#44AF69")
        ax.set_title(f"Distribución de {columna}")
        ax.set_ylabel("Frecuencia")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def figura_bivariado_num_cat(
        self, numerica: str, categorica: str, tipo_grafico: str = "boxplot"
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        if tipo_grafico == "boxplot":
            sns.boxplot(data=self.df, x=categorica, y=numerica, ax=ax, palette="Set2")
        else:
            sns.violinplot(data=self.df, x=categorica, y=numerica, ax=ax, palette="Set2")
        ax.set_title(f"{numerica} vs {categorica}")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def tabla_bivariado_num_cat(self, numerica: str, categorica: str) -> pd.DataFrame:
        return (
            self.df.groupby(categorica, observed=True)[numerica]
            .agg(["count", "mean", "median", "std"])
            .round(2)
        )

    def figura_bivariado_cat_cat(
        self, cat1: str, cat2: str, normalizar: bool = False
    ) -> plt.Figure:
        tabla = pd.crosstab(self.df[cat1], self.df[cat2], normalize=normalizar)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(tabla, annot=True, fmt=".2f" if normalizar else "d", cmap="Blues", ax=ax)
        ax.set_title(f"{cat1} vs {cat2}")
        fig.tight_layout()
        return fig

    def tasa_exito_por_grupo(self, columna: str) -> pd.DataFrame:
        if "y" not in self.df.columns:
            return pd.DataFrame()
        total = self.df.groupby(columna, observed=True).size()
        exitos = self.df[self.df["y"] == "yes"].groupby(columna, observed=True).size()
        tasa = (exitos / total * 100).fillna(0).round(2)
        return pd.DataFrame({"contactos": total, "aceptaciones": exitos.fillna(0).astype(int), "tasa_%": tasa})

    def analisis_dinamico(self, columnas: list[str]) -> pd.DataFrame:
        columnas_validas = [c for c in columnas if c in self.df.columns]
        if not columnas_validas:
            return pd.DataFrame()

        resultado = pd.DataFrame(index=columnas_validas)
        for col in columnas_validas:
            if col in self.columnas_numericas:
                resultado.loc[col, "tipo"] = "numérica"
                resultado.loc[col, "media"] = self.media(col)
                resultado.loc[col, "mediana"] = self.mediana(col)
                resultado.loc[col, "desv_std"] = self.df[col].std()
                resultado.loc[col, "min"] = self.df[col].min()
                resultado.loc[col, "max"] = self.df[col].max()
            else:
                resultado.loc[col, "tipo"] = "categórica"
                resultado.loc[col, "valores_unicos"] = self.df[col].nunique()
                resultado.loc[col, "moda"] = self.moda(col)
        return resultado.round(2)

    def metricas_clave(self) -> dict[str, Any]:
        """Métricas resumen para hallazgos y conclusiones."""
        total = len(self.df)
        tasa_global = (
            (self.df["y"] == "yes").sum() / total * 100 if "y" in self.df.columns else 0
        )

        duracion_si = self.df.loc[self.df["y"] == "yes", "duration"].median() if "duration" in self.df.columns else 0
        duracion_no = self.df.loc[self.df["y"] == "no", "duration"].median() if "duration" in self.df.columns else 0

        contacto_tasa = {}
        if "contact" in self.df.columns:
            contacto_tasa = self.tasa_exito_por_grupo("contact")["tasa_%"].to_dict()

        return {
            "total_registros": total,
            "tasa_aceptacion_global": round(tasa_global, 2),
            "duracion_mediana_si": duracion_si,
            "duracion_mediana_no": duracion_no,
            "tasa_por_contacto": contacto_tasa,
            "edad_media": self.media("age"),
            "campana_media": self.media("campaign"),
        }
