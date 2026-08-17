# 🍷 Clasificador de Calidad de Vinos - Proyecto de Machine Learning
📋 Descripción del Proyecto
Este proyecto implementa un sistema de clasificación para predecir la calidad de vinos tintos basado en sus características químicas. Utiliza algoritmos de Machine Learning para determinar si un vino es "bueno" (calidad ≥ 6) o "malo" (calidad < 6).

## 🎯 Objetivos

- **Análisis Exploratorio:** Comprender la distribución y relaciones entre las variables químicas del vino
- **Preprocesamiento:** Manejar outliers, multicolinealidad y preparar los datos para modelado
- **Modelado:** Implementar y comparar múltiples algoritmos de clasificación
- **Optimización:** Mejorar el rendimiento mediante RandomizedSearchCV
- **Despliegue:** Crear una aplicación web interactiva con Streamlit

## 🏗️ Estructura del Proyecto

📁 Wine/
│  ├── 📁 Data/
│  │  ├── winequality-red.csv          # Dataset original
│  │  ├── wine_EDA.csv                 # Dataset después de EDA
│  │  ├── wine.csv                     # Dataset con outliers
|  │  └── wine_no.csv                  # Dataset sin outliers
│  ├── 📁 Models/
│  │  ├── rf_wine.pkl                  # Modelo Random Forest
│  │  ├── features.pkl                 # Lista de features
│  │  └── top_features.pkl             # Features más importantes
│  └── 📁 Graphs/
│     ├── distribucion_variables.png
│     └── mapa_correlaciones.png
├── 📁 Scripts/
│   ├── 01_EDA.py                       # Análisis exploratorio
│   ├── 02_FeatureEngineering.py        # Ingeniería de características
│   └── 03_Modelos.py                   # Entrenamiento de modelos
├── app.py                              # Aplicación Streamlit
├── requirements.txt                    # Dependencias del proyecto
├── README.md                           # Este archivo
└── .gitignore                          # Archivos a ignorar en Git

## Crear entorno virtual

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

## Instalar dependencias

pip install -r requirements.txt

## 📊 Decisiones Técnicas

### Preprocesamiento

- **Eliminación de duplicados:** 240 filas duplicadas removidas
- **Tratamiento de outliers:** Dataset sin outliers seleccionado por mejor rendimiento
- **Multicolinealidad:** Eliminación de features con correlación > 0.65
- **Features eliminadas:** 'fixed acidity', 'free sulfur dioxide'

### Modelado

- **Algoritmos comparados:** Logistic Regression, Random Forest, XGBoost, SVC
- **Métrica principal:** F1-Score (balance entre Precision y Recall)
- **Optimización:** RandomizedSearchCV 
- **Dataset seleccionado:** Sin outliers 

## **📄 Licencia**
Este proyecto está bajo la Licencia MIT.

## **👨‍🎓 Autor**
Samuel Martínez
GitHub: @SamuelMtnz