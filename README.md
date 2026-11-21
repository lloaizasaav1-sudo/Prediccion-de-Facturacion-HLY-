# 🏥 Analítica Predictiva de Facturación y Operación Hospitalaria (CRISP-DM)

Proyecto académico y aplicativo analítico desarrollado bajo la metodología **CRISP-DM** para predecir `VALOR FACTURADO` y evaluar el desempeño y los márgenes de error del modelo entre unidades funcionales y centros de costos hospitalarios, apoyando la planeación y la toma de decisiones gerenciales.

## 📌 Información Académica

* **Estudiante:** Laura Juliana Loaiza Saavedra
* **ID:** 428429
* **Programa:** Ingeniería Industrial
* **Institución:** Corporación Universitaria Minuto de Dios (UNIMINUTO)
* **Docente:** Xammy Alexander Victoria González

## 🎯 Enfoque Metodológico (CRISP-DM)

1. **Comprensión del Negocio y de los Datos:** Definición del problema predictivo sobre `VALOR FACTURADO` a partir de registros institucionales (`CONSOLIDADO.xlsx`), identificando variables operativas, de servicio, entidad y costos (`UNIDAD FUNCIONAL`, `C. COSTOS DETALLE`, `C. COSTOS ENCAB-EMISOR`, entre otras) y previniendo fugas de información (*data leakage*).
2. **Preparación de los Datos:** Limpieza, tipificación numérica, codificación de variables categóricas y división en conjuntos de entrenamiento (80 %) y prueba (20 %) para validar la capacidad de generalización del modelo.
3. **Modelado Predictivo:** Entrenamiento y comparación de modelos de regresión (Regresión Lineal, Árbol de Decisión y `RandomForestRegressor`) para estimar `VALOR FACTURADO` sobre registros no vistos.
4. **Evaluación y Diagnóstico de Errores:** Análisis de desempeño mediante **MAE**, **RMSE** y **R²**, inspección de residuos (`Real - Predicho`) para detectar dispersión en valores atípicos y revisión de errores por unidad funcional.
5. **Visualización y Proyección Analítica:** Presentación interactiva de resultados y proyecciones de soporte en Streamlit para contrastar valores reales vs. predichos y apoyar el análisis operativo por unidad funcional.

## 🛠️ Tecnologías

* **Lenguaje y análisis de datos:** Python, `pandas`, `numpy`
* **Modelado y evaluación:** `scikit-learn` (`RandomForestRegressor`, `LinearRegression`, `DecisionTreeRegressor`, métricas `MAE`, `RMSE`, `R²`)
* **Visualización y aplicación web:** `matplotlib`, `seaborn`, `Streamlit` y despliegue en Streamlit Cloud
