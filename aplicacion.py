import io
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Analítica Predictiva - Hospital de Yotoco",
    page_icon="🏥",
    layout="wide"
)

sns.set_theme(style="whitegrid")

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
    st.title("🏥 Analítica Predictiva de Facturación y Operación Hospitalaria")
    st.markdown(
        "**E.S.E. Hospital Local de Yotoco** | *Autora:* Laura Juliana Loaiza Saavedra (ID: 428429) - **UNIMINUTO**"
    )

with col_logo2:
    try:
        st.image("logo2.png", width=120)
    except Exception:
        st.write("📌 [Logo 2]")

st.markdown("---")

# ==============================================================================
# PANEL LATERAL: CARGA DE ARCHIVO Y PERIODO
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


def buscar_columna(data, candidatos, fallback_idx=None):
    for c in candidatos:
        if c in data.columns:
            return c
    if fallback_idx is not None and fallback_idx < len(data.columns):
        return data.columns[fallback_idx]
    return None


@st.cache_data
def cargar_datos_por_defecto():
    rutas_candidatas = ['CONSOLIDADO.xlsx'] + glob.glob('**/*CONSOLIDADO*.xlsx', recursive=True)
    for ruta in rutas_candidatas:
        try:
            return pd.read_excel(ruta, sheet_name='Plantilla De Informe')
        except Exception:
            try:
                return pd.read_excel(ruta)
            except Exception:
                continue
    return None


if archivo_subido is not None:
    try:
        try:
            df = pd.read_excel(archivo_subido, sheet_name='Plantilla De Informe')
        except Exception:
            archivo_subido.seek(0)
            df = pd.read_excel(archivo_subido)
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
    col_unidad = buscar_columna(df, ['UNIDAD FUNCIONAL', 'UNIDAD_FUNCIONAL'], 0)
    col_detalle = buscar_columna(df, ['C. COSTOS DETALLE', 'C_COSTOS_DETALLE'], 1)
    col_articulo = buscar_columna(df, ['NOMBRE ARTICULO', 'NOMBRE_ARTICULO', 'ARTICULO'], 2)
    col_factura = buscar_columna(df, ['VALOR FACTURADO', 'VALOR_FACTURADO'])
    col_costo = buscar_columna(df, ['VALOR COSTO', 'VALOR_COSTO'])
    col_profesional = buscar_columna(df, ['NOMBRE PROFESIONAL', 'PROFESIONAL'])
    col_entidad = buscar_columna(df, ['NOMBRE ENTIDAD', 'ENTIDAD', 'EPS'])
    col_cantidad = buscar_columna(df, ['CANTIDAD', 'CANT'])
    col_servicio = buscar_columna(df, ['TIPO SERVICIO', 'TIPO_SERVICIO'])

    if col_factura and col_factura in df.columns:
        df[col_factura] = pd.to_numeric(df[col_factura], errors='coerce')
    if col_costo and col_costo in df.columns:
        df[col_costo] = pd.to_numeric(df[col_costo], errors='coerce')

    # ==========================================================================
    # FILTROS GERENCIALES
    # ==========================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filtros Gerenciales")

    unidades_disponibles = sorted(df[col_unidad].dropna().astype(str).unique().tolist()) if col_unidad else []
    unidad_seleccionada = st.sidebar.selectbox(
        "Filtrar por Unidad Funcional",
        ["Todas"] + unidades_disponibles
    )

    if unidad_seleccionada != "Todas" and col_unidad:
        df_filtrado = df[df[col_unidad].astype(str) == unidad_seleccionada].copy()
    else:
        df_filtrado = df.copy()

    def convertir_a_excel(df_exportar, nombre_hoja_excel):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_exportar.to_excel(writer, index=False, sheet_name=nombre_hoja_excel[:31])
        return output.getvalue()

    st.subheader(f"📅 Reporte y Modelado Predictivo Xenco - **{mes_seleccionado} de {anio_seleccionado}**")
    st.markdown(f"**Unidad Funcional Filtrada:** `{unidad_seleccionada}` | **Registros:** `{len(df_filtrado):,}`")
    st.markdown("---")

    # ==========================================================================
    # MÉTRICAS EJECUTIVAS GLOBALES
    # ==========================================================================
    st.header("📈 Indicadores Financieros Globales")

    total_facturado = df_filtrado[col_factura].sum() if col_factura else 0.0
    total_costo = df_filtrado[col_costo].fillna(0).sum() if col_costo else 0.0
    utilidad_operativa = total_facturado - total_costo
    margen_utilidad = (utilidad_operativa / total_facturado * 100) if total_facturado > 0 else 0.0
    total_atenciones = len(df_filtrado)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Facturación Total", f"${total_facturado:,.2f}")
    c2.metric("📉 Costo Operativo Total", f"${total_costo:,.2f}")
    c3.metric("💼 Utilidad Operativa", f"${utilidad_operativa:,.2f}", delta=f"{margen_utilidad:.1f}% Margen")
    c4.metric("📋 Total Registros Xenco", f"{total_atenciones:,}")

    st.markdown("---")

    # ==========================================================================
    # PESTAÑAS: MODELADO PREDICTIVO (CRISP-DM) + CONTEXTO OPERATIVO
    # ==========================================================================
    tab_pred, tab_contexto, tab_unidades = st.tabs([
        "🤖 Modelado Predictivo y Residuos",
        "📊 Contexto Operativo (Servicios, EPS y Médicos)",
        "🏢 Unidades Funcionales y Exportación"
    ])

    with tab_pred:
        st.subheader("Predicción de `VALOR FACTURADO` y Diagnóstico de Errores")
        st.markdown(
            "Entrenamiento supervisado (80% entrenamiento / 20% prueba) sin fuga de datos (*data leakage*), "
            "comparando modelos y evaluando **MAE**, **RMSE**, **R²** y **residuos**."
        )

        if not col_factura:
            st.warning("⚠️ No se encontró la columna `VALOR FACTURADO` para entrenar el modelo.")
        else:
            df_ml = df_filtrado.dropna(subset=[col_factura]).copy()
            features_x = []

            if col_cantidad:
                df_ml[col_cantidad] = pd.to_numeric(df_ml[col_cantidad], errors='coerce').fillna(1)
                features_x.append(col_cantidad)
            if col_costo:
                df_ml[col_costo] = pd.to_numeric(df_ml[col_costo], errors='coerce').fillna(0)
                features_x.append(col_costo)

            for cod_name, orig_col in [
                ('TIPO_SERVICIO_COD', col_servicio),
                ('EPS_COD', col_entidad),
                ('ARTICULO_COD', col_articulo),
                ('PROFESIONAL_COD', col_profesional),
                ('UNIDAD_COD', col_unidad),
                ('DETALLE_COD', col_detalle)
            ]:
                if orig_col and orig_col in df_ml.columns:
                    serie_cat = df_ml[orig_col].fillna('SIN_ESPECIFICAR').astype(str).str.strip()
                    df_ml[cod_name] = pd.factorize(serie_cat)[0]
                    features_x.append(cod_name)

            if len(df_ml) < 15 or not features_x:
                st.warning("⚠️ Se requieren al menos 15 registros válidos y variables predictoras para entrenar y evaluar.")
            else:
                X = df_ml[features_x]
                y = df_ml[col_factura]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                modelos = {
                    "Regresión Lineal": LinearRegression(),
                    "Árbol de Decisión": DecisionTreeRegressor(max_depth=5, random_state=42),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                }

                resumen_modelos = []
                predicciones = {}

                for nombre_mod, est in modelos.items():
                    est.fit(X_train, y_train)
                    pred_test = est.predict(X_test)
                    predicciones[nombre_mod] = pred_test
                    resumen_modelos.append({
                        "Modelo": nombre_mod,
                        "MAE ($)": mean_absolute_error(y_test, pred_test),
                        "RMSE ($)": np.sqrt(mean_squared_error(y_test, pred_test)),
                        "R²": r2_score(y_test, pred_test)
                    })

                df_modelos = pd.DataFrame(resumen_modelos).sort_values(by="RMSE ($)", ascending=True)
                st.dataframe(
                    df_modelos.style.format({
                        "MAE ($)": "${:,.2f}",
                        "RMSE ($)": "${:,.2f}",
                        "R²": "{:.4f}"
                    }),
                    use_container_width=True
                )

                y_pred_rf = predicciones["Random Forest Regressor"]
                residuos = y_test - y_pred_rf

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE (Random Forest)", f"${mean_absolute_error(y_test, y_pred_rf):,.2f}")
                m2.metric("RMSE (Random Forest)", f"${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}")
                m3.metric("R² (Random Forest)", f"{r2_score(y_test, y_pred_rf):.4f}")

                g1, g2 = st.columns(2)
                with g1:
                    fig, ax = plt.subplots(figsize=(6.5, 4))
                    ax.scatter(y_test, y_pred_rf, alpha=0.45, color='teal', label='Predicciones')
                    lim_min = min(y_test.min(), y_pred_rf.min())
                    lim_max = max(y_test.max(), y_pred_rf.max())
                    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Ideal (1:1)')
                    ax.set_title("Valor Real vs. Predicho (Random Forest)")
                    ax.set_xlabel("VALOR FACTURADO Real ($)")
                    ax.set_ylabel("VALOR FACTURADO Predicho ($)")
                    ax.legend()
                    st.pyplot(fig)

                with g2:
                    fig, ax = plt.subplots(figsize=(6.5, 4))
                    sns.histplot(residuos, bins=30, kde=True, color='crimson', ax=ax)
                    ax.axvline(0, color='black', linestyle='--', lw=1.5, label='Error 0')
                    ax.set_title("Distribución de Residuos (Real - Predicho)")
                    ax.set_xlabel("Residuo ($)")
                    ax.set_ylabel("Frecuencia")
                    ax.legend()
                    st.pyplot(fig)

                if col_unidad and col_unidad in df_ml.columns:
                    st.markdown("#### Error Absoluto Medio (MAE) por Unidad Funcional en Prueba")
                    df_err_uf = pd.DataFrame({
                        "Unidad Funcional": df_ml.loc[X_test.index, col_unidad].values,
                        "Real ($)": y_test.values,
                        "Predicho ($)": y_pred_rf,
                        "Error Absoluto ($)": np.abs(residuos.values)
                    })
                    res_uf = df_err_uf.groupby("Unidad Funcional").agg(
                        Casos_Prueba=("Real ($)", "count"),
                        MAE_Unidad=("Error Absoluto ($)", "mean"),
                        Total_Real=("Real ($)", "sum"),
                        Total_Predicho=("Predicho ($)", "sum")
                    ).sort_values(by="MAE_Unidad", ascending=False)
                    st.dataframe(res_uf, use_container_width=True)

    with tab_contexto:
        if col_profesional and col_factura:
            st.subheader("👨‍⚕️ Productividad por Profesional Médico")
            resumen_medicos = (
                df_filtrado.groupby(col_profesional)[col_factura]
                .agg(['sum', 'count', 'mean'])
                .reset_index()
            )
            resumen_medicos.columns = [
                'Profesional Médico', 'Facturación Total ($)',
                'Cantidad Atenciones', 'Promedio por Atención ($)'
            ]
            resumen_medicos = resumen_medicos.sort_values(by='Facturación Total ($)', ascending=False)

            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown("**Top 10 Médicos con Mayor Facturación**")
                st.bar_chart(resumen_medicos.head(10).set_index('Profesional Médico')['Facturación Total ($)'])
            with cm2:
                st.markdown("**Médicos con Menor Facturación Registrada**")
                st.bar_chart(resumen_medicos.tail(10).set_index
