import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y META FINANCIERA MENSUAL
# ==============================================================================
st.set_page_config(
    page_title="Meta de Facturación - Hospital de Yotoco",
    page_icon="🏥",
    layout="wide"
)

META_GASTO_MENSUAL = 582_000_000  # Promedio / meta mensual de gastos (COP)

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
# CABECERA INSTITUCIONAL
# ==============================================================================
col_logo1, col_titulo, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    try:
        st.image("logo1.png", width=120)
    except Exception:
        st.write("📌 [Logo 1]")

with col_titulo:
    st.title("🏥 Proyección de Meta de Facturación Hospitalaria")
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


def normalizar_columnas(df_in):
    df_out = df_in.copy()
    df_out.columns = [str(c).strip() for c in df_out.columns]
    return df_out


def cargar_consolidado(fuente):
    """Carga un archivo Excel priorizando 'Plantilla De Informe' si existe."""
    try:
        xl = pd.ExcelFile(fuente)
        hoja = "Plantilla De Informe" if "Plantilla De Informe" in xl.sheet_names else xl.sheet_names[0]
        return normalizar_columnas(pd.read_excel(xl, sheet_name=hoja))
    except Exception:
        return normalizar_columnas(pd.read_excel(fuente))


def buscar_col(data, candidatos, fallback=None):
    for c in candidatos:
        if c in data.columns:
            return c
    for col in data.columns:
        col_up = col.upper()
        for cand in candidatos:
            if cand.upper() in col_up:
                return col
    if fallback is not None and fallback < len(data.columns):
        return data.columns[fallback]
    return None


def calcular_meta_categoria(df_base, col_cat, col_val, meta_total, col_pred=None):
    """
    Calcula cuánto debería facturar cada categoría para llegar a `meta_total`.
    Combina facturación del mes anterior y predicción del modelo base.
    """
    if not col_cat or col_cat not in df_base.columns:
        return pd.DataFrame()

    tmp = df_base.copy()
    tmp[col_cat] = tmp[col_cat].fillna("SIN ESPECIFICAR").astype(str).str.strip()

    if col_pred and col_pred in tmp.columns:
        agrup = tmp.groupby(col_cat, dropna=False).agg(
            Facturacion_Mes_Anterior=(col_val, "sum"),
            Prediccion_Modelo_Base=(col_pred, "sum"),
            Registros=(col_val, "count")
        ).reset_index()
        base_peso = (
            0.5 * agrup["Facturacion_Mes_Anterior"].clip(lower=0)
            + 0.5 * agrup["Prediccion_Modelo_Base"].clip(lower=0)
        )
    else:
        agrup = tmp.groupby(col_cat, dropna=False).agg(
            Facturacion_Mes_Anterior=(col_val, "sum"),
            Registros=(col_val, "count")
        ).reset_index()
        base_peso = agrup["Facturacion_Mes_Anterior"].clip(lower=0)

    suma_pesos = float(base_peso.sum())
    if suma_pesos > 0:
        agrup["Participacion_%"] = (base_peso / suma_pesos) * 100.0
        agrup["Meta_Facturacion_Mes_Sgtes ($)"] = (base_peso / suma_pesos) * meta_total
    else:
        n_cats = max(len(agrup), 1)
        agrup["Participacion_%"] = 100.0 / n_cats
        agrup["Meta_Facturacion_Mes_Sgtes ($)"] = meta_total / n_cats

    agrup["Incremento_Requerido ($)"] = agrup["Meta_Facturacion_Mes_Sgtes ($)"] - agrup["Facturacion_Mes_Anterior"]
    return agrup.sort_values(by="Meta_Facturacion_Mes_Sgtes ($)", ascending=False).reset_index(drop=True)


def generar_excel_meta(
    resumen_df,
    df_servicios,
    df_medicos,
    df_entidades,
    df_unidades,
    df_modelos,
    resumen_txt
):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        resumen_df.to_excel(writer, index=False, sheet_name="Resumen_Meta_582M")
        if not df_servicios.empty:
            df_servicios.to_excel(writer, index=False, sheet_name="Meta_Por_Servicios")
        if not df_medicos.empty:
            df_medicos.to_excel(writer, index=False, sheet_name="Meta_Por_Medicos")
        if not df_entidades.empty:
            df_entidades.to_excel(writer, index=False, sheet_name="Meta_Por_Entidades")
        if not df_unidades.empty:
            df_unidades.to_excel(writer, index=False, sheet_name="Meta_Por_Unidades")
        if not df_modelos.empty:
            df_modelos.to_excel(writer, index=False, sheet_name="Comparativa_Modelos")
        pd.DataFrame({"Informe_Ejecutivo": [resumen_txt]}).to_excel(
            writer, index=False, sheet_name="Texto_Informe"
        )
    return output.getvalue()


# ==============================================================================
# PANEL LATERAL: SUBIDA DEL CONSOLIDADO Y CONFIGURACIÓN DEL MES SIGUIENTE
# ==============================================================================
st.sidebar.header("📁 1. Subir Consolidado")
archivo_subido = st.sidebar.file_uploader(
    "Suba el archivo Consolidado (.xlsx)",
    type=["xlsx"]
)

meta_gasto = st.sidebar.number_input(
    "Promedio / Meta de Gastos Mensual (COP)",
    min_value=1_000_000,
    value=META_GASTO_MENSUAL,
    step=1_000_000,
    format="%d"
)

meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]
mes_siguiente = st.sidebar.selectbox("Mes Siguiente a Proyectar", meses, index=11)
anio_proyeccion = st.sidebar.selectbox("Año de Proyección", [2026, 2025, 2027], index=0)

df_raw = None
origen_datos = None

if archivo_subido is not None:
    try:
        df_raw = cargar_consolidado(archivo_subido)
        origen_datos = f"Archivo subido: {archivo_subido.name}"
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo subido: {e}")
else:
    try:
        df_raw = cargar_consolidado("CONSOLIDADO.xlsx")
        origen_datos = "Archivo local predeterminado: CONSOLIDADO.xlsx"
    except Exception:
        df_raw = None

if df_raw is None or df_raw.empty:
    st.warning("Sube el archivo Excel del **Consolidado** en el panel lateral para analizar y calcular la meta de facturación.")
    st.stop()

st.sidebar.success(f"Datos cargados: {origen_datos} ({len(df_raw):,} registros)")

# ==============================================================================
# DETECCIÓN DE COLUMNAS, CÁLCULO DE META Y DESGLOSE POR RUBROS
# ==============================================================================
try:
    col_factura = buscar_col(df_raw, ["VALOR FACTURADO", "VALOR_FACTURADO", "FACTURADO", "FACTURACION"])
    if not col_factura:
        st.error(f"No se encontró la columna 'VALOR FACTURADO'. Columnas disponibles: {list(df_raw.columns)}")
        st.stop()

    col_fecha = buscar_
