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
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
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
    st.title("🏥 Inteligencia Financiera, Costos y Analítica Predictiva")
    st.markdown(
        "**E.S.E. Hospital Local de Yotoco** | *Autora:* Laura Juliana Loaiza Saavedra "
        "(ID: 428429) - **UNIMINUTO**"
    )

with col_logo2:
    try:
        st.image("logo2.png", width=120)
    except Exception:
        st.write("📌 [Logo 2]")

st.markdown("---")


def leer_excel_xenco(fuente):
    """Lee un archivo Excel (.xlsx) priorizando 'Plantilla De Informe' si existe."""
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
    """Genera un libro Excel multipestaña con el informe predictivo completo."""
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
    [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
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
        st.warning("📌 Por favor, suba el archivo Excel de Xenco (.xlsx) en el panel lateral.")

if datos_cargados:
    # Mapeo seguro de columnas de Xenco
    col_unidad = "UNIDAD FUNCIONAL" if "UNIDAD FUNCIONAL" in df.columns else df.columns[0]
