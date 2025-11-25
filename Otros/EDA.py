import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from pathlib import Path

#-------------------------------------------
#      Carga y Compobacion de archivo
#-------------------------------------------

try:
    df = pd.read_csv('Wine/Data/winequality-red.csv')

    print(f'\n ✅ Datos cargaos correctamente \n Vista previa: ')
    print(df.head())
    
    print(f'\n 📊 Resumen estadístico:')
    print(df.describe())
    
except FileNotFoundError as e:
    print(f'\n ❌ Error: {str(e)}')
except ValueError as e:
    print(f'\n ❌ Error: {str(e)}')
except pd.errors.EmptyDataError:
    print(f'\n ❌ Archivo vacío')
except pd.errors.ParserError:
    print(f'\n ❌ No se puede leer, revisar formato')
except Exception as e:
    print(f'\n ❌ Error inesperado: {str(e)}')
    
    
print(df.info())

d = df.duplicated()
nd = df.duplicated().sum()
print(f'\n ⚠️ Número de filas duplicadas: {nd}')

df = df.drop_duplicates()
print(f'\n ⚠️ Número de filas duplicadas: {df.duplicated().sum()}')

null = df.isnull()
print(f'\n ⚠️ Número de valores nulos: \n{null.sum()}')


#-------------------------------------------
#                GRÁFICOS
#-------------------------------------------

#-------------------------------------------
#    DISTRIBUCIÓN: SKEWNNES - KOURTOSIS
#-------------------------------------------

print("\n" + "-"*70)
print("RESUMEN SKEWNESS - DISTRIBUCIÓN DE VARIABLES")
print("-"*70)

resumen = []
for column in df.columns:
    skewness = df[column].skew()
    
    if abs(skewness) < 0.5:
        distrib = "✅ Simétrica"
    elif skewness > 1:
        distrib = "⚠️  Muy positiva"
    elif skewness > 0.5:
        distrib = "📈 Positiva"
    elif skewness < -1:
        distrib = "⚠️  Muy negativa"
    else:
        distrib = "📉 Negativa"
    
    resumen.append({
        'Variable': column,
        'Skewness': f"{skewness:.3f}",
        'Distribución': distrib,
        'Media': f"{df[column].mean():.3f}",
        'Mediana': f"{df[column].median():.3f}"
    })

resumen_df = pd.DataFrame(resumen)
print(resumen_df.to_string(index=False))

print(f"\n📝 Leyenda:")
print("✅ Simétrica: |skewness| < 0.5")
print("📈 Positiva: skewness > 0.5")
print("📉 Negativa: skewness < -0.5")
print("⚠️  Muy positiva/negativa: |skewness| > 1")


#-------------------------------------------
#        Análisis de Distribución
#-------------------------------------------

print("\n" + "-"*50)
print("ANÁLISIS DE DISTRIBUCIÓN DE VARIABLES")
print("-"*50)

# Configurar el estilo de los gráficos
plt.style.use('default')
sns.set_palette("husl")

# Crear subplots organizados
fig, axes = plt.subplots(4, 3, figsize = (15, 12))
axes = axes.ravel()  # Aplanar el array de ejes

# Generar histogramas para cada variable
for i, column in enumerate(df.columns):
    if i < len(axes):  # Asegurarse de no exceder el número de ejes
        # Histograma con curva de densidad
        sns.histplot(df[column], ax = axes[i], kde = True, bins=30)
        axes[i].set_title(f'Distribución de {column}', fontsize = 10, fontweight = 'bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frecuencia')
        
        # Añadir líneas de media y mediana
        mean_val = df[column].mean()
        median_val = df[column].median()
        axes[i].axvline(mean_val, color = 'red', linestyle = '--', linewidth = 1, label = f'Media: {mean_val:.2f}')
        axes[i].axvline(median_val, color = 'green', linestyle = '--', linewidth = 1, label = f'Mediana: {median_val:.2f}')
        axes[i].legend(fontsize = 8)

plt.tight_layout()


# Creamos directorio
graphs_dir = Path("Wine/Graphs")
graphs_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Carpeta para gráficos creada/verificada: {graphs_dir}")

distribucion_path = graphs_dir / "distribucion_variables.png"
plt.savefig(distribucion_path, dpi=300, bbox_inches='tight') 
print(f"✅ Gráfico de distribución guardado: {distribucion_path}")
plt.close(fig)


#-------------------------------------------
#               OUTLIERS
#-------------------------------------------

plt.figure(figsize = (15, 12))
for i, col in enumerate(df.columns, 1):
    plt.subplot(6, 2, i)
    sns.boxplot( x = col, y = 'quality', data = df)
    plt.title(f'Quality vs {col}')
    plt.xlabel(col)
    plt.ylabel('Quality')
    
plt.tight_layout()


# Gráfico de outliers
outliers_path = graphs_dir / "boxplots_outliers.png"
plt.savefig(outliers_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico de outliers guardado: {outliers_path}")
plt.close('all')


#-------------------------------------------
#             CORRELACIONES
#-------------------------------------------

plt.figure(figsize = (10, 10))

sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.title('Mapa Correlaciones')

# plt.close(fig)


# Mapa de correlaciones
corr_path = graphs_dir / "mapa_correlaciones.png"
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
print(f"✅ Mapa de correlaciones guardado: {corr_path}")
plt.close('all')


#-------------------------------------------
#                 INSIGHTS EDA
#-------------------------------------------

print("\n" + "="*70)
print("INSIGHTS PRINCIPALES - ANÁLISIS EXPLORATORIO")
print("="*70)

print("\n🎯 HALLAZGOS CLAVES IDENTIFICADOS:")

print("\n📊 1. CALIDAD DE DATOS INICIAL:")
print(f"   • Duplicados eliminados: {nd} filas")
print(f"   • Dataset final: {len(df)} muestras, {len(df.columns)} variables")
print("   • No hay valores nulos - Calidad de datos buena")

print("\n📈 2. DISTRIBUCIÓN DE VARIABLES:")
print("   • La mayoría presenta ASIMETRÍA POSITIVA (cola derecha)")
print("   • Esto indica distribuciones no normales")
print("   • Variables con mayor asimetría requieren transformación")

print("\n⚠️  3. PROBLEMAS DE OUTLIERS:")
print("   • Presencia significativa de valores atípicos")
print("   • Detectados visualmente en múltiples variables")
print("   • Pueden afectar modelos sensibles a outliers")

print("\n🔗 4. MULTICOLINEALIDAD:")
print("   • Correlaciones fuertes (>0.5) entre variables")
print("   • Riesgo de redundancia en features")
print("   • Posible necesidad de selección/reducción")

print("\n🍷 5. CONTEXTO DEL DOMINIO (VINO):")
print("   • Las correlaciones pueden tener sentido químico")
print("   • Algunos outliers podrían ser vinos excepcionales")
print("   • Consultar con experto en enología para validar")

print("\n" + "-"*70)
print("OPCIONES PARA FEATURE ENGINEERING")
print("-"*70)

print("\n🛠️  TRANSFORMACIONES:")
print("   1. Aplicar log/box-cox a variables con alta asimetría")
print("   2. Evaluar escalado (StandardScaler, RobustScaler)")
print("   3. Considerar técnicas de reducción de dimensionalidad")

print("\n🎯 DECISIONES A TOMAR:")
print("   • ¿Eliminar o transformar outliers?")
print("   • ¿Mantener todas las variables o seleccionar features?")
print("   • ¿Aplicar PCA para multicolinealidad?")

print("\n📝 PASOS A SEGUIR:")
print("   1. Exportar insights a documento")
print("   2. Pasar a archivo de Feature Engineering")
print("   3. Definir estrategia basada en estos hallazgos")

print("\n" + "="*70)

#-------------------------------------------
#                 GUARDADO
#-------------------------------------------

df.to_csv('Wine/Data/wine_EDA.csv', index = False)
print(f'\n💾 Wine/Data/wine_EDA.csv')

