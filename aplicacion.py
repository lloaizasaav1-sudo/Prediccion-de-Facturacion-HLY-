import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y REFERENTE FINANCIERO
# ==============================================================================
st.set_page_config(
    page_title="Gerencia Financiera - Hospital de Yotoco",
    page_icon="🏥",
    layout="wide"
)

GASTO_REFERENCIA = 581_014_027  # Promedio de gastos (nómina, insumos, operación)

plt.style.use("default")
sns.set_theme(style="whitegrid")

# Estilo visual corporativo limpio
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CABECERA INSTITUCIONAL CON LOS DOS LOGOS DEL HOSPITAL
# ==============================================================================
col_logo1, col_titulo, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    try:
        st.image("logo1.png", width=120)
    except Exception:
        st.write("📌 [Logo 1]")

with col_titulo:
    st.title("🏥 Inteligencia Financiera y Costos Hospitalarios")
    st.markdown("**E.S.E. Hospital Local de Yotoco** | *Autora:* Laura Juliana Loaiza Saavedra (ID: 428429) - **UNIMINUTO**")

with col_logo2:
    try:
        st.image("logo2.png", width=120)
    except Exception:
        st.write("📌 [Logo 2]")

st.markdown("---")


def leer_excel_xenco(fuente):
    """Lee un archivo Excel priorizando 'Plantilla De Informe' cuando está disponible."""
    try:
        xl = pd.ExcelFile(fuente)
        if "Plantilla De Informe" in xl.sheet_names:
            return pd.read_excel(xl, sheet_name="Plantilla De Informe")
        return pd.read_excel(xl, sheet_name=0)
    except Exception:
        return pd.read_excel(fuente)


def convertir_a_excel(df_exportar, nombre_hoja_excel):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_exportar.to_excel(writer, index=False, sheet_name=nombre_hoja_excel)
    return output.getvalue()


def generar_excel_informe_predictivo(
    df_comparativa,
    df_predicciones,
    df_unidades_err,
    resumen_texto,
    mes_str,
    anio_int,
    unidad_str
):
    """Genera un archivo Excel multipestaña con el informe predictivo y su conclusión."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_comparativa.to_excel(writer, index=False, sheet_name="Comparativa_Modelos")
        if df_unidades_err is not None and not df_unidades_err.empty:
            df_unidades_err.to_excel(writer, index=False, sheet_name="Error_Unidades")
        df_predicciones.head(1000).to_excel(writer, index=False, sheet_name="Predicciones_Muestra")
        df_resumen = pd.DataFrame({
            "Periodo": [f"{mes_str} {anio_int}"],
            "Unidad_Funcional": [unidad_str],
            "Gasto_Referencia_COP": [GASTO_REFERENCIA],
            "Resumen_Ejecutivo": [resumen_texto]
        })
        df_resumen.to_excel(writer, index=False, sheet_name="Resumen_Ejecutivo")
    return output.getvalue()


# ==============================================================================
# PANEL LATERAL: CARGA DE ARCHIVO Y SELECCIÓN DE PERIODO (XENCO)
# ==============================================================================
st.sidebar.header("📁 Carga de Datos y Periodo")
st.sidebar.markdown("Suba el archivo de Xenco y especifique el periodo:")

mes_seleccionado = st.sidebar.selectbox(
    "Mes de Facturación",
    ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
)
anio_seleccionado = st.sidebar.selectbox("Año", [2026, 2025, 2024, 2023])

archivo_subido = st.sidebar.file_uploader("Cargar Facturación Xenco (.xlsx)", type=["xlsx"])


@st.cache_data
def cargar_datos_por_defecto():
    try:
        return leer_excel_xenco("CONSOLIDADO.xlsx")
    except Exception:
        return None


if archivo_subido is not None:
    try:
        df = leer_excel_xenco(archivo_subido)
        st.sidebar.success(f"✅ Archivo Xenco cargado para **{mes_seleccionado} {anio_seleccionado}**")
        datos_cargados = True
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al leer el archivo: {e}")
        datos_cargados = False
else:
    df = cargar_datos_por_defecto()
    if df is not None:
        datos_cargados = True
        st.sidebar.info(f"📌 Usando archivo predeterminado (**Periodo: {mes_seleccionado} {anio_seleccionado}**).")
    else:
        datos_cargados = False
        st.warning("📌 Por favor, suba el archivo Excel de Xenco en el panel lateral.")

if datos_cargados:
    # Mapeo seguro de columnas de Xenco
    col_unidad = "UNIDAD FUNCIONAL" if "UNIDAD FUNCIONAL" in df.columns else df.columns[0]
    col_detalle = "C. COSTOS DETALLE" if "C. COSTOS DETALLE" in df.columns else (df.columns[1] if len(df.columns) > 1 else col_unidad)
    col_articulo = "NOMBRE ARTICULO" if "NOMBRE ARTICULO" in df.columns else (df.columns[2] if len(df.columns) > 2 else col_unidad)
    col_factura = "VALOR FACTURADO" if "VALOR FACTURADO" in df.columns else "VALOR FACTURADO"
    col_costo = "VALOR COSTO" if "VALOR COSTO" in df.columns else "VALOR COSTO"
    col_profesional = "NOMBRE PROFESIONAL" if "NOMBRE PROFESIONAL" in df.columns else "NOMBRE PROFESIONAL"
    col_entidad = "NOMBRE ENTIDAD" if "NOMBRE ENTIDAD" in df.columns else "NOMBRE ENTIDAD"
    col_servicio = "TIPO SERVICIO" if "TIPO SERVICIO" in df.columns else None
    col_cantidad = "CANTIDAD" if "CANTIDAD" in df.columns else None

    if col_factura in df.columns:
        df[col_factura] = pd.to_numeric(df[col_factura], errors="coerce").fillna(0)
    if col_costo in df.columns:
        df[col_costo] = pd.to_numeric(df[col_costo], errors="coerce").fillna(0)
    if col_cantidad and col_cantidad in df.columns:
        df[col_cantidad] = pd.to_numeric(df[col_cantidad], errors="coerce").fillna(1)

    # ==============================================================================
    # FILTROS GERENCIALES
    # ==============================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filtros Gerenciales")

    unidades_disponibles = list(df[col_unidad].dropna().unique()) if col_unidad in df.columns else []
    unidad_seleccionada = st.sidebar.selectbox("Filtrar por Unidad Funcional", ["Todas"] + [str(u) for u in unidades_disponibles])

    if unidad_seleccionada != "Todas" and col_unidad in df.columns:
        df_filtrado = df[df[col_unidad].astype(str) == unidad_seleccionada].copy()
    else:
        df_filtrado = df.copy()

    # ==============================================================================
    # CABECERA DEL PERIODO
    # ==============================================================================
    st.subheader(f"📅 Reporte Financiero Xenco - **{mes_seleccionado} de {anio_seleccionado}**")
    st.markdown(f"**Unidad Funcional Filtrada:** `{unidad_seleccionada}`")
    st.markdown("---")

    # ==============================================================================
    # MÉTRICAS EJECUTIVAS GLOBALES + INDICADORES NUEVOS
    # ==============================================================================
    st.header("📈 Indicadores Financieros Globales")

    total_facturado = df_filtrado[col_factura].sum() if col_factura in df_filtrado.columns else 0
    total_costo = df_filtrado[col_costo].sum() if col_costo in df_filtrado.columns else 0
    utilidad_operativa = total_facturado - total_costo
    margen_utilidad = (utilidad_operativa / total_facturado * 100) if total_facturado > 0 else 0
    total_atenciones = len(df_filtrado)

    col1, col2, col3, col4 = st.columns(4)
