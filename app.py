"""
Bank Marketing EDA — Aplicación interactiva con Streamlit.
Especialización en Python for Analytics — Caso de Estudio N°1
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import DataAnalyzer

# ── Configuración global ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTOR = {
    "nombre": "Henry Alex Enciso Gomez",
    "curso": "Especialización en Python for Analytics — DMC",
    "anio": 2026,
}

TECNOLOGIAS = ["Python", "NumPy", "Pandas", "Matplotlib", "Seaborn", "Streamlit"]


def cargar_dataframe(archivo) -> pd.DataFrame:
    """Lee CSV con separador ';' y comillas."""
    return pd.read_csv(archivo, sep=";", quotechar='"')


def mostrar_figura(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


def requiere_datos() -> bool:
    if st.session_state.get("df") is None:
        st.warning("⚠️ Primero carga el dataset en **Carga del dataset**.")
        return False
    return True


# ── Módulo 1: Home ────────────────────────────────────────────────────────────
def modulo_home() -> None:
    st.title("🏦 Bank Marketing — Análisis Exploratorio de Datos")
    st.markdown(
        """
        ### Objetivo del análisis
        Explorar los factores que influyen en la **aceptación de campañas de marketing**
        de una institución financiera, cuya efectividad cayó del **12% al 8%** en los
        últimos 6 meses. Esta herramienta permite un EDA interactivo **sin modelado predictivo**,
        orientado a decisiones comerciales basadas en evidencia.
        """
    )
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Datos del autor")
        st.markdown(
            f"""
            - **Nombre:** {AUTOR['nombre']}
            - **Curso:** {AUTOR['curso']}
            - **Año:** {AUTOR['anio']}
            """
        )
    with col2:
        st.subheader("🛠️ Tecnologías utilizadas")
        st.markdown(" · ".join(f"`{t}`" for t in TECNOLOGIAS))

    st.divider()
    st.subheader("📂 Sobre el dataset")
    st.markdown(
        """
        El archivo **BankMarketing.csv** contiene registros de contactos telefónicos
        y personales realizados durante una campaña de depósitos a plazo fijo. Incluye
        **21 variables** demográficas, financieras, de contacto y macroeconómicas, con
        la variable objetivo **`y`** (`yes` / `no`) que indica si el cliente aceptó la oferta.

        | Variable | Descripción |
        |----------|-------------|
        | `age`, `job`, `marital`, `education` | Perfil del cliente |
        | `default`, `housing`, `loan` | Situación crediticia |
        | `contact`, `month`, `day_of_week` | Canal y timing del contacto |
        | `duration`, `campaign`, `pdays`, `previous` | Historial de contacto |
        | `emp.var.rate`, `cons.price.idx`, `euribor3m`… | Indicadores macro |
        | **`y`** | **Resultado de la campaña** |
        """
    )


# ── Módulo 2: Carga del dataset ───────────────────────────────────────────────
def modulo_carga() -> None:
    st.header("📥 Carga del dataset")
    st.markdown(
        "Sube el archivo **BankMarketing.csv** para habilitar el análisis exploratorio."
    )

    archivo = st.file_uploader(
        "Selecciona el archivo CSV",
        type=["csv"],
        help="Formato esperado: separador ';' y comillas dobles",
    )

    usar_local = st.checkbox("Usar dataset incluido en el repositorio (BankMarketing.csv)")

    df_cargado = None

    if usar_local:
        try:
            df_cargado = cargar_dataframe("BankMarketing.csv")
            st.success("✅ Dataset local cargado correctamente.")
        except FileNotFoundError:
            st.error("No se encontró BankMarketing.csv en el directorio del proyecto.")

    if archivo is not None:
        try:
            df_cargado = cargar_dataframe(archivo)
            st.success(f"✅ Archivo **{archivo.name}** cargado correctamente.")
        except Exception as exc:
            st.error(f"Error al leer el archivo: {exc}")

    if df_cargado is not None:
        st.session_state["df"] = df_cargado
        st.session_state["analyzer"] = DataAnalyzer(df_cargado)

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{df_cargado.shape[0]:,}")
        c2.metric("Columnas", df_cargado.shape[1])
        c3.metric("Memoria (aprox.)", f"{df_cargado.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        st.subheader("Vista previa")
        st.dataframe(df_cargado.head(10), use_container_width=True)


# ── Módulo 3: EDA ─────────────────────────────────────────────────────────────
def modulo_eda() -> None:
    if not requiere_datos():
        return

    analyzer: DataAnalyzer = st.session_state["analyzer"]
    df = st.session_state["df"]

    st.header("🔍 Análisis Exploratorio de Datos (EDA)")

    tabs = st.tabs(
        [
            "1 · Info general",
            "2 · Clasificación",
            "3 · Estadísticas",
            "4 · Valores faltantes",
            "5 · Numéricas",
            "6 · Categóricas",
            "7 · Num vs Cat",
            "8 · Cat vs Cat",
            "9 · Dinámico",
            "10 · Hallazgos",
        ]
    )

    # ── Ítem 1 ────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Ítem 1: Información general del dataset")
        info = analyzer.resumen_info()

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{info['filas']:,}")
        c2.metric("Variables", info["columnas"])
        c3.metric("Memoria (KB)", info["memoria_kb"])

        st.markdown("**Salida equivalente a `.info()`**")
        st.code(info["info_texto"], language="text")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Tipos de datos**")
            st.dataframe(info["tipos"], use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Conteo de valores nulos**")
            st.dataframe(info["nulos"], use_container_width=True, hide_index=True)

    # ── Ítem 2 ────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Ítem 2: Clasificación de variables")
        st.markdown(
            "Clasificación mediante la función personalizada `clasificar_variables()` "
            "y la clase `DataAnalyzer`."
        )

        clasif = analyzer.clasificacion
        c1, c2 = st.columns(2)
        c1.metric("Variables numéricas", clasif["conteo_numericas"])
        c2.metric("Variables categóricas", clasif["conteo_categoricas"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Numéricas**")
            for col in clasif["numericas"]:
                st.write(f"- `{col}`")
        with col_b:
            st.markdown("**Categóricas**")
            for col in clasif["categoricas"]:
                st.write(f"- `{col}`")

    # ── Ítem 3 ────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Ítem 3: Estadísticas descriptivas")
        desc = analyzer.estadisticas_descriptivas()
        st.dataframe(desc, use_container_width=True)

        col_sel = st.selectbox(
            "Variable numérica para interpretación",
            analyzer.columnas_numericas,
            key="desc_var",
        )
        media_v = analyzer.media(col_sel)
        mediana_v = analyzer.mediana(col_sel)
        std_v = df[col_sel].std()

        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"{media_v:.2f}" if media_v else "—")
        c2.metric("Mediana", f"{mediana_v:.2f}" if mediana_v else "—")
        c3.metric("Desv. estándar", f"{std_v:.2f}")

        st.info(
            f"La **media** ({media_v:.2f}) y la **mediana** ({mediana_v:.2f}) de `{col_sel}` "
            f"{'difieren notablemente, sugiriendo asimetría' if media_v and mediana_v and abs(media_v - mediana_v) > std_v * 0.3 else 'son similares, distribución relativamente simétrica'}."
        )

    # ── Ítem 4 ────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Ítem 4: Análisis de valores faltantes")
        nulos = analyzer.conteo_nulos()
        st.dataframe(nulos, use_container_width=True)

        hay_nulos = nulos["conteo"].sum() > 0
        if hay_nulos:
            mostrar_figura(analyzer.figura_valores_faltantes())
            st.markdown(
                "**Discusión:** Identificar columnas con mayor proporción de nulos "
                "permite decidir imputación o exclusión antes de cualquier acción comercial."
            )
        else:
            st.success("El dataset no presenta valores nulos explícitos (NaN).")
            st.markdown(
                "**Nota:** Valores como `unknown` en variables categóricas representan "
                "información faltante codificada y deben tratarse en etapas posteriores."
            )

    # ── Ítem 5 ────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Ítem 5: Distribución de variables numéricas")
        col_num = st.selectbox("Variable numérica", analyzer.columnas_numericas, key="hist_var")
        bins = st.slider("Número de bins", 10, 80, 30, key="hist_bins")
        mostrar_kde = st.checkbox("Mostrar curva KDE", key="hist_kde")

        mostrar_figura(analyzer.figura_histograma(col_num, bins=bins, mostrar_kde=mostrar_kde))
        st.markdown(
            f"**Interpretación:** Observa la concentración, colas y la relación entre "
            f"media y mediana en `{col_num}` para detectar outliers o asimetrías."
        )

    # ── Ítem 6 ────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Ítem 6: Análisis de variables categóricas")
        col_cat = st.selectbox("Variable categórica", analyzer.columnas_categoricas, key="cat_var")
        mostrar_prop = st.checkbox("Mostrar proporciones (%)", value=True, key="cat_prop")
        top_n = st.slider("Top categorías en gráfico", 3, 20, 10, key="cat_top")

        resumen = analyzer.resumen_categorica(col_cat)
        st.dataframe(resumen, use_container_width=True)

        if mostrar_prop:
            st.bar_chart(resumen["proporcion_%"])

        mostrar_figura(
            analyzer.figura_barras_categorica(col_cat, top_n=top_n, horizontal=True)
        )

    # ── Ítem 7 ────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Ítem 7: Análisis bivariado (numérico vs categórico)")
        st.markdown("Ejemplos sugeridos: `age` vs `y`, `duration` vs `y`.")

        c1, c2 = st.columns(2)
        with c1:
            var_num = st.selectbox(
                "Variable numérica",
                analyzer.columnas_numericas,
                index=analyzer.columnas_numericas.index("age")
                if "age" in analyzer.columnas_numericas
                else 0,
                key="biv_num",
            )
        with c2:
            var_cat = st.selectbox(
                "Variable categórica",
                analyzer.columnas_categoricas,
                index=analyzer.columnas_categoricas.index("y")
                if "y" in analyzer.columnas_categoricas
                else 0,
                key="biv_cat",
            )

        tipo_g = st.selectbox("Tipo de gráfico", ["boxplot", "violinplot"], key="biv_tipo")
        mostrar_figura(analyzer.figura_bivariado_num_cat(var_num, var_cat, tipo_g))
        st.dataframe(
            analyzer.tabla_bivariado_num_cat(var_num, var_cat),
            use_container_width=True,
        )

    # ── Ítem 8 ────────────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Ítem 8: Análisis bivariado (categórico vs categórico)")
        st.markdown("Ejemplos sugeridos: `education` vs `y`, `contact` vs `y`.")

        c1, c2 = st.columns(2)
        opciones_cat = analyzer.columnas_categoricas
        idx_edu = opciones_cat.index("education") if "education" in opciones_cat else 0
        idx_y = opciones_cat.index("y") if "y" in opciones_cat else 0

        with c1:
            cat1 = st.selectbox("Variable 1", opciones_cat, index=idx_edu, key="cat_cat1")
        with c2:
            cat2 = st.selectbox("Variable 2", opciones_cat, index=idx_y, key="cat_cat2")

        normalizar = st.checkbox("Mostrar proporciones normalizadas", key="cat_norm")
        mostrar_figura(analyzer.figura_bivariado_cat_cat(cat1, cat2, normalizar))

        if cat2 == "y":
            st.markdown("**Tasa de aceptación por grupo**")
            st.dataframe(analyzer.tasa_exito_por_grupo(cat1), use_container_width=True)

    # ── Ítem 9 ────────────────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Ítem 9: Análisis dinámico por parámetros seleccionados")
        columnas_sel = st.multiselect(
            "Selecciona columnas a analizar",
            df.columns.tolist(),
            default=["age", "duration", "campaign", "y"],
            key="dyn_cols",
        )

        if columnas_sel:
            resultado = analyzer.analisis_dinamico(columnas_sel)
            st.dataframe(resultado, use_container_width=True)

            cols_num_sel = [c for c in columnas_sel if c in analyzer.columnas_numericas]
            if len(cols_num_sel) >= 2:
                st.markdown("**Matriz de correlación (variables numéricas seleccionadas)**")
                st.dataframe(
                    df[cols_num_sel].corr().round(3),
                    use_container_width=True,
                )
        else:
            st.info("Selecciona al menos una columna.")

    # ── Ítem 10 ───────────────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("Ítem 10: Hallazgos clave")
        metricas = analyzer.metricas_clave()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros analizados", f"{metricas['total_registros']:,}")
        c2.metric("Tasa aceptación global", f"{metricas['tasa_aceptacion_global']}%")
        c3.metric("Duración mediana (sí)", f"{metricas['duracion_mediana_si']:.0f}s")
        c4.metric("Duración mediana (no)", f"{metricas['duracion_mediana_no']:.0f}s")

        if metricas["tasa_por_contacto"]:
            st.markdown("**Tasa de éxito por canal de contacto**")
            tasa_df = pd.Series(metricas["tasa_por_contacto"], name="tasa_%").sort_values(ascending=False)
            st.bar_chart(tasa_df)

        st.divider()
        st.subheader("📌 Insights principales del EDA")
        insights = [
            f"La tasa global de aceptación es **{metricas['tasa_aceptacion_global']}%**, "
            "coherente con la caída de efectividad reportada (12% → 8%).",
            f"Los contactos exitosos tienen duración mediana de **{metricas['duracion_mediana_si']:.0f}s** "
            f"vs **{metricas['duracion_mediana_no']:.0f}s** en rechazos — el tiempo de conversación es un indicador clave.",
            "El canal de contacto muestra diferencias en tasas de conversión; conviene priorizar el canal con mejor desempeño.",
            "Variables macro (`emp.var.rate`, `euribor3m`) y demográficas (`education`, `job`) muestran patrones diferenciados por segmento.",
            "Valores `unknown` y códigos como `pdays=999` deben documentarse antes de decisiones operativas.",
        ]
        for i, texto in enumerate(insights, 1):
            st.markdown(f"{i}. {texto}")

        mostrar_resumen = st.checkbox("Mostrar gráfico resumen: tasa por educación", value=True)
        if mostrar_resumen and "education" in df.columns:
            tasa_edu = analyzer.tasa_exito_por_grupo("education")["tasa_%"].sort_values(ascending=True)
            st.bar_chart(tasa_edu)


# ── Módulo 4: Conclusiones ────────────────────────────────────────────────────
def modulo_conclusiones() -> None:
    if not requiere_datos():
        return

    analyzer: DataAnalyzer = st.session_state["analyzer"]
    m = analyzer.metricas_clave()

    st.header("📋 Conclusiones finales")
    st.markdown(
        "Cinco conclusiones basadas en el EDA, orientadas a **toma de decisiones comerciales** "
        "(no predicción):"
    )

    conclusiones = [
        (
            "**La duración del contacto es el factor más accionable.** "
            f"Los clientes que aceptan (`yes`) mantienen conversaciones notablemente más largas "
            f"(mediana {m['duracion_mediana_si']:.0f}s vs {m['duracion_mediana_no']:.0f}s). "
            "Capacitar al equipo para extender el diálogo de calidad —sin ser invasivo— puede mejorar conversiones."
        ),
        (
            "**La efectividad global ({:.1f}%) está por debajo del histórico del 12%.** "
            "Se requiere revisar segmentación y mensajes; el EDA sugiere enfocar esfuerzos "
            "en perfiles con mayor tasa observada por educación y ocupación."
        ).format(m["tasa_aceptacion_global"]),
        (
            "**El canal de contacto importa.** "
            "Las tasas de aceptación difieren entre canales (telephone vs cellular). "
            "Redistribuir presupuesto hacia el canal con mejor desempeño es una decisión de bajo costo."
        ),
        (
            "**Menos contactos repetidos, más calidad.** "
            f"La campaña actual promedia {m['campana_media']:.1f} intentos por cliente. "
            "Reducir saturación y priorizar clientes con `pdays` favorables puede recuperar confianza."
        ),
        (
            "**Contexto macroeconómico condiciona resultados.** "
            "Variables como `emp.var.rate` y `euribor3m` varían en el periodo analizado; "
            "las decisiones comerciales deben ajustar metas cuando el entorno económico es adverso."
        ),
    ]

    for i, texto in enumerate(conclusiones, 1):
        st.markdown(f" {i}. {texto}")


# ── Sidebar y navegación ──────────────────────────────────────────────────────
def main() -> None:
    with st.sidebar:
        st.markdown("## 🏦 Bank Marketing")
        st.title("Menú principal")
        modulo = st.radio(
            "Navegación",
            ["Home", "Carga del dataset", "EDA", "Conclusiones"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"© {AUTOR['anio']} — {AUTOR['nombre']}")
        if st.session_state.get("df") is not None:
            st.success("Dataset cargado ✓")
        else:
            st.warning("Sin dataset cargado")

    if modulo == "Home":
        modulo_home()
    elif modulo == "Carga del dataset":
        modulo_carga()
    elif modulo == "EDA":
        modulo_eda()
    else:
        modulo_conclusiones()


if __name__ == "__main__":
    if "df" not in st.session_state:
        st.session_state["df"] = None
        st.session_state["analyzer"] = None
    main()
