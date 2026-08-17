import streamlit as st 
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
DATA_DIR = BASE_DIR / "Data"

st.set_page_config(
    page_title = '🍷 Clasificador de Vinos - Análisis Técnico',
    page_icon = '🍷',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# Título principal CORREGIDO
st.title('🍷 Clasificador de Calidad de Vinos - Decisiones Técnicas')
st.markdown('''
**Predice si un vino es bueno o malo basado en características químicas**  
*Modelo Random Forest optimizado con decisiones técnicas específicas*
''')

# 🔥 NUEVO: Diccionario con importancias exactas
FEATURE_IMPORTANCES = {
    'alcohol': 0.205,
    'sulphates': 0.152,
    'volatile acidity': 0.119,
    'total sulfur dioxide': 0.113,
    'density': 0.100,
    'chlorides': 0.089,
    'pH': 0.083,
    'citric acid': 0.079,
    'residual sugar': 0.060
}

@st.cache_resource
def cargar_modelo_y_features():
    """Carga el modelo y las features"""
    try:
        if not (MODELS_DIR / "rf_wine.pkl").exists():
            st.error(f"❌ No se encuentra: {MODELS_DIR / 'rf_wine.pkl'}")
            return None, None
        if not (MODELS_DIR / "features.pkl").exists():
            st.error(f"❌ No se encuentra: {MODELS_DIR / 'features.pkl'}")
            return None, None
        
        modelo = joblib.load(MODELS_DIR / "rf_wine.pkl")
        features = joblib.load(MODELS_DIR / "features.pkl")
        return modelo, features
        
    except Exception as e:
        st.error(f'⚠️ Error cargando modelo: {str(e)}')
        return None, None

@st.cache_data
def cargar_datos():
    """Carga el dataset para estadísticas"""
    try:
        if (DATA_DIR / "wine_no.csv").exists():
            return pd.read_csv(DATA_DIR / "wine_no.csv")
        elif (DATA_DIR / "wine.csv").exists():
            return pd.read_csv(DATA_DIR / "wine.csv")
        else:
            st.error("❌ No se encuentra ningún dataset en Data/")
            return None
    except Exception as e:
        st.error(f'⚠️ Error cargando datos: {str(e)}')
        return None

# CARGAR RECURSOS
with st.spinner('Cargando modelo y datos...'):
    modelo, features = cargar_modelo_y_features()
    df_wine = cargar_datos()

if modelo is None or features is None or df_wine is None:
    st.error("❌ No se pudieron cargar todos los recursos necesarios")
    st.stop()

st.sidebar.success('✅ Modelo y datos cargados correctamente')

# =============================================
# SIDEBAR CON MÉTRICAS EXACTAS
# =============================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Decisiones Técnicas")

with st.sidebar.expander("📊 Métricas Elegidas", expanded = True):
    st.markdown("""
    **F1-Score como métrica principal:**
    - 🤝 **Media armónica** entre Precision y Recall
    - ⚖️ **Balance ideal** incluso con clases balanceadas
    - 🎯 **Evita sesgos** hacia Precision o Recall individual
    - 📈 **Métrica robusta** para evaluación completa
    """)
    st.metric("F1-Score Logrado", "81.42%")

with st.sidebar.expander("📈 Tratamiento de Correlaciones", expanded = True):
    st.markdown("""
    **Eliminación de características con correlación > 0.65:**
    - 🔍 **Umbral conservador** para evitar multicolinealidad
    - ⚖️ **Compromiso consciente**: interpretabilidad vs poder predictivo
    - 💡 **Podría usarse 0.8** para retener más información
    - 🎯 **Decision**: Modelo más generalizable y estable
    """)

with st.sidebar.expander("🗑️ Selección de Dataset", expanded = True):
    st.markdown("""
    **Dataset sin outliers seleccionado:**
    - ✅ **Mejores resultados** en validación cruzada
    - 🎯 **Objetivo práctico**: Vinos de calidad estándar
    - 📊 **Outliers eliminados**: vinos excepcionales/defectuosos
    - 💡 **Enfoque realista** para aplicación comercial
    - 🔄 **Dataset con outliers** disponible para análisis específicos
    """)

# Métricas exactas en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Métricas del Modelo")

# Mostrar todas las métricas
st.sidebar.metric("Accuracy", "80.19%")
st.sidebar.metric("AUC-ROC", "85.32%")
st.sidebar.metric("F1-Score", "81.42%")
st.sidebar.metric("Precision", "82.88%")
st.sidebar.metric("Recall", "80.00%")

# Top features en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔝 Top 5 Features")
top_features = sorted(FEATURE_IMPORTANCES.items(), key = lambda x: x[1], reverse = True)[:5]
for i, (feature, importance) in enumerate(top_features, 1):
    st.sidebar.write(f"{i}. **{feature}** ({importance*100:.1f}%)")

# Información sobre balance de clases
distribucion_calidad = df_wine['quality'].value_counts()
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Distribución de Clases")
for calidad, count in distribucion_calidad.items():
    porcentaje = (count / len(df_wine)) * 100
    st.sidebar.write(f"**Calidad {calidad}:** {count} ({porcentaje:.1f}%)")

st.sidebar.write(f"**Total registros:** {len(df_wine)}")

# =============================================
# SECCIÓN PRINCIPAL 
# =============================================
st.markdown("---")

# Explicación técnica CORREGIDA
with st.expander("🔍 **Análisis Técnico Detallado - Decisiones Fundamentadas**", expanded = True):
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.subheader("🎯 Estrategia de Modelado")
        st.markdown("""
        **1. Selección de F1-Score como Métrica Principal**
        - ⚖️ **Clases balanceadas** - distribución equilibrada
        - 🤝 **F1-Score** combina Precision y Recall armónicamente
        - 🎯 **Evaluación completa** - no solo aciertos totales
        - 📊 **Robustez** frente a cambios en el threshold
        
        **2. Tratamiento de Correlaciones (umbral 0.65)**
        - 🔍 **Multicolinealidad controlada** - modelo más estable
        - ⚖️ **Compromiso consciente** - interpretabilidad vs poder
        - 💡 **Alternativa considerada**: umbral 0.8 para más features
        - 🎯 **Decisión final**: generalización sobre sobreajuste
        """)
    
    with col_tech2:
        st.subheader("📊 Decisiones de Preprocesamiento")
        st.markdown("""
        **3. Dataset Sin Outliers - Justificación**
        - 📈 **Mejor rendimiento** en validación cruzada
        - 🎯 **Caso de uso real**: vinos de calidad estándar
        - 🚫 **Outliers**: vinos excepcionales o defectuosos
        - 💼 **Aplicación comercial** - mayoría de casos
        
        **4. Validación Robusta**
        - 🔄 **10-fold cross validation** - estimación estable
        - ⚖️ **Train/Test estratificado** - mantiene distribución
        - 🎯 **RandomizedSearchCV** - optimización eficiente
        - 📊 **Múltiples métricas** - evaluación comprehensiva
        """)

# GRÁFICO DE IMPORTANCIAS 
st.markdown("---")
st.subheader("📈 Importancia de las Características")

# Gráfico o
col_graph1, col_graph2 = st.columns([2, 1])

with col_graph1:
    # Crear gráfico de importancias MÁS PEQUEÑO
    fig, ax = plt.subplots(figsize = (6, 4))  #
    features_sorted = sorted(FEATURE_IMPORTANCES.items(), key =  lambda x: x[1])
    features_names = [x[0].title() for x in features_sorted]
    importances = [x[1] for x in features_sorted]

    # Usar colores viridis para el gráfico
    colors = plt.cm.viridis(np.linspace(0, 1, len(features_names)))
    bars = ax.barh(features_names, importances, color = colors, height = 0.6)  # 

    # Añadir valores en las barras con fuente más pequeña
    for bar, importance in zip(bars, importances):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance*100:.1f}%', ha = 'left', va = 'center', 
                fontweight='bold', fontsize=9)   

    ax.set_xlabel('Importancia', fontsize = 10)
    ax.set_title('Importancia de Características', fontsize = 12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 9)
    plt.tight_layout()

    st.pyplot(fig)

with col_graph2:
    st.markdown("#### 💡 Interpretación")
    st.markdown("""
    **Alta Importancia:**
    - 🥇 **Alcohol** (20.5%)
    - 🥈 **Sulphates** (15.2%)
    - 🥉 **Volatile Acidity** (11.9%)
    
    **Factores clave** que determinan la calidad del vino.
    
    **Total importancia acumulada:** 100%
    """)

# SECCIÓN DE PREDICCIÓN
st.markdown("---")
st.subheader("🎯 Realizar Predicción")

col1, col2, col3 = st.columns(3)

inputs = {}
valores_por_defecto = {
    'alcohol': 10.5,
    'sulphates': 0.65,
    'volatile acidity': 0.26,
    'total sulfur dioxide': 38.0,
    'density': 0.994,
    'chlorides': 0.045,
    'pH': 3.31,
    'citric acid': 0.31,
    'residual sugar': 2.5
}

#  Crear tooltips con importancia
def get_feature_help(feature_name):
    importance = FEATURE_IMPORTANCES.get(feature_name, 0)
    help_texts = {
        'alcohol': f"Importancia: {importance*100:.1f}% - Contenido alcohólico del vino",
        'sulphates': f"Importancia: {importance*100:.1f}% - Nivel de sulfatos (conservante)",
        'volatile acidity': f"Importancia: {importance*100:.1f}% - Acidez volátil (vinagre)",
        'total sulfur dioxide': f"Importancia: {importance*100:.1f}% - Total de SO₂ (antioxidante)",
        'density': f"Importancia: {importance*100:.1f}% - Densidad del vino",
        'chlorides': f"Importancia: {importance*100:.1f}% - Salinidad del vino",
        'pH': f"Importancia: {importance*100:.1f}% - Acidez del vino",
        'citric acid': f"Importancia: {importance*100:.1f}% - Acidez cítrica",
        'residual sugar': f"Importancia: {importance*100:.1f}% - Azúcar residual después de fermentación"
    }
    return help_texts.get(feature_name, f"Importancia: {importance*100:.1f}%")

with col1:
    # Features alta importancia
    st.markdown("#### 🥇 Alta Importancia")
    inputs['alcohol'] = st.slider("Alcohol (%)", 8.0, 15.0, valores_por_defecto['alcohol'], 0.1, 
                                 help = get_feature_help('alcohol'))
    inputs['sulphates'] = st.slider("Sulphates", 0.3, 1.2, valores_por_defecto['sulphates'], 0.01,
                                   help = get_feature_help('sulphates'))
    inputs['volatile acidity'] = st.slider("Volatile Acidity", 0.1, 1.0, valores_por_defecto['volatile acidity'], 0.01,
                                          help = get_feature_help('volatile acidity'))

with col2:
    # Features media importancia
    st.markdown("#### 🥈 Media Importancia")
    inputs['total sulfur dioxide'] = st.slider("Total SO₂", 10.0, 150.0, valores_por_defecto['total sulfur dioxide'], 1.0,
                                              help = get_feature_help('total sulfur dioxide'))
    inputs['density'] = st.slider("Density", 0.990, 1.000, valores_por_defecto['density'], 0.001,
                                 help = get_feature_help('density'))
    inputs['chlorides'] = st.slider("Chlorides", 0.01, 0.15, valores_por_defecto['chlorides'], 0.001,
                                   help = get_feature_help('chlorides'))

with col3:
    # Features baja importancia
    st.markdown("#### 🥉 Baja Importancia")
    inputs['pH'] = st.slider("pH", 2.5, 4.5, valores_por_defecto['pH'], 0.01,
                            help = get_feature_help('pH'))
    inputs['citric acid'] = st.slider("Citric Acid", 0.0, 1.0, valores_por_defecto['citric acid'], 0.01,
                                     help = get_feature_help('citric acid'))
    inputs['residual sugar'] = st.slider("Residual Sugar", 1.0, 10.0, valores_por_defecto['residual sugar'], 0.1,
                                        help = get_feature_help('residual sugar'))
# BOTÓN DE PREDICCIÓN
st.markdown("---")
if st.button("🍷 **Predecir Calidad del Vino**", type = "primary", use_container_width = True):
    try:
        datos_prediccion = pd.DataFrame([inputs])
        datos_prediccion = datos_prediccion[features]
        
        prediccion = modelo.predict(datos_prediccion)[0]
        probabilidades = modelo.predict_proba(datos_prediccion)[0]
        
        # Mostrar resultados
        st.success("## 📊 Resultados de la Predicción")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            if prediccion == 1:
                st.success("### ✅ Vino BUENO")
                st.metric("Calidad", "ALTA")
            else:
                st.error("### ❌ Vino MALO")
                st.metric("Calidad", "BAJA")
        
        with col_res2:
            st.metric("Probabilidad Vino BUENO", f"{probabilidades[1]:.1%}")
        
        with col_res3:
            st.metric("Probabilidad Vino MALO", f"{probabilidades[0]:.1%}")
        
        # Análisis de confianza 
        st.subheader("🎯 Análisis de Confianza")
        confianza = max(probabilidades)
        
        if confianza > 0.8:
            st.success(f"**Alta confianza en la predicción:** {confianza:.1%}")
            st.info("📊 El modelo tiene alta certeza en esta clasificación")
        elif confianza > 0.6:
            st.warning(f"**Confianza moderada en la predicción:** {confianza:.1%}")
            st.info("📊 Considera revisar características adicionales")
        else:
            st.error(f"**Baja confianza en la predicción:** {confianza:.1%}")
            st.info("📊 Los valores ingresados pueden estar en zona de transición")
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")

# Sección de análisis técnico con métricas exactas
st.markdown("---")
with st.expander("🔍 Análisis Técnico del Modelo", expanded = False):
    st.subheader("📊 Rendimiento del Modelo")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        **Métricas de Clasificación:**
        - **Accuracy (Exactitud):** 80.19% - Porcentaje de aciertos totales
        - **Precision (Precisión):** 82.88% - Capacidad de no predecir falsos buenos
        - **Recall (Sensibilidad):** 80.00% - Capacidad de encontrar todos los vinos buenos
        - **F1-Score:** 81.42% - Balance entre Precision y Recall
        """)
    
    with col_tech2:
        st.markdown("""
        **Calidad del Modelo:**
        - **AUC-ROC:** 85.32% - Excelente capacidad discriminativa
        - **Balance:** Buen equilibrio entre clases
        - **Robustez:** Modelo generalizable a nuevos datos
        - **Interpretabilidad:** Features con importancia clara
        """)
    
    st.subheader("🎯 Interpretación de Importancias")
    st.markdown("""
    ***Alta Importancia (>11,5%):**
    - **Alcohol (20.5%):** El factor más determinante en la calidad
    - **Sulphates (15.2%):** Conservantes que afectan positivamente
    - **Volatile Acidity (11.9%):** Acidez que en exceso es negativa
        
    **Media Importancia (8,5% - 11,5%):**
    - **Total Sulfur Dioxide (11.3%):** Antioxidante y conservante
    - **Density (10.0%):** Relacionado con el cuerpo del vino
    - **Chlorides (8.9%):** Salinidad que afecta el sabor
    
    **Baja Importancia (<8,5%):**
    - **pH (8.3%):** Nivel de acidez del vino
    - **Citric Acid (7.9%):** Acidez cítrica que aporta frescura
    - **Residual Sugar (6.0%):** Azúcar residual después de fermentación
    
    **Los 9 features combinados** proveen una evaluación completa y robusta.
    """)

# SECCIÓN DE ANÁLISIS DE DECISIONES
st.markdown("---")
with st.expander("🔬 Análisis de Trade-offs Técnicos", expanded = False):
    st.subheader("Evaluación de Decisiones Alternativas")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("""
        **📈 Correlación > 0.8 (alternativa):**
        - ✅ **Posible**: +1-2% en métricas
        - ❌ **Riesgo**: Multicolinealidad
        - 🔍 **Impacto**: Menos interpretable
        - 🎯 **Decisión**: Estabilidad sobre máximo rendimiento
        """)
    
    with col_comp2:
        st.markdown("""
        **🗑️ Con outliers (alternativa):**
        - ✅ **Ventaja**: Detección de vinos excepcionales
        - ❌ **Desventaja**: Menor precisión en casos típicos
        - 📊 **Resultado**: Peor generalización
        - 🎯 **Decisión**: Casos comunes sobre casos raros
        """)

# FOOTER ACTUALIZADO
st.markdown("---")
st.caption("""
🍷 **Clasificador de Vinos - Análisis Técnico Completo** | 
Accuracy: 80.19% | AUC-ROC: 85.32% | F1-Score: 81.42% |
9 Features | Modelo Random Forest Optimizado
""")