import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from pathlib import Path

class EDA:
    
    def __init__(self, data_path, graphs_dir = 'Wine/Graphs'):
        self.data_path = Path(data_path)
        self.graphs_dir = Path(graphs_dir)
        self.df = None
        
        self.graphs_dir.mkdir(parents = True, exist_ok = True)
        print(f'📁 Carpeta de gráficos creada/verificada: {self.graphs_dir}')
        
    def guardar_grafico(self, fig, nombre):
        save_path = self.graphs_dir / nombre
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.close(fig)
        print(f'✅ Gráfico guardado: {save_path}')
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f'\n✅ Datos cargados correctamente desde {self.data_path}')
            print('\nVista previa:')
            print(self.df.head(),'\n')
            print('📊 Resumen estadístico:')
            print(self.df.describe(), '\n')
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

    def clean_data(self):
        if self.df is None:
            print('⚠️ No hay datos cargados. Usa load_data() primero.')
            return
        n_dup = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace = True)
        print(f'⚠️ Filas duplicadas eliminadas: {n_dup}')
        print(f'🚫 Valores nulos por columna:\n{self.df.isnull().sum()}\n')
    
    def distribucion(self):
        resumen = []
        for column in self.df.columns:
            skewness = self.df[column].skew()
            
            if abs(skewness) < 0.5:
                distrib = '✅ Simétrica'
            elif skewness > 1:
                distrib = '⚠️  Muy positiva'
            elif skewness > 0.5:
                distrib = '📈 Positiva'
            elif skewness < -1:
                distrib = '⚠️  Muy negativa'
            else:
                distrib = '📉 Negativa'

            resumen.append({
                'Variable': column,
                'Skewness': f'{skewness:.3f}',
                'Distribución': distrib,
                'Media': f'{self.df[column].mean():.3f}',
                'Mediana': f'{self.df[column].median():.3f}'
            })

        resumen_df = pd.DataFrame(resumen)
        print(resumen_df.to_string(index=False))
        print('\n📝 Leyenda:')
        print('✅ Simétrica: |skewness| < 0.5')
        print('📈 Positiva: skewness > 0.5')
        print('📉 Negativa: skewness < -0.5')
        print('⚠️  Muy positiva/negativa: |skewness| > 1')
        return resumen_df
    
    def graficar(self):
        sns.set_palette('husl')
        fig, axes = plt.subplots(4, 3, figsize = (15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate (self.df.columns):
            sns.histplot(self.df[col], ax = axes[i],kde = True, bins = 30 )
            mean_val, median_val = self.df[col].mean(), self.df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Media: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=1, label=f'Mediana: {median_val:.2f}')
            axes[i].set_title(f'Distribución de {col}', fontsize=10, fontweight='bold')
            axes[i].legend(fontsize=8)

        fig.tight_layout()
        self.guardar_grafico(fig, 'distribucion_variables.png')
        
    def outliers(self):
        fig, axes = plt.subplots(6, 2, figsize = (15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(self.df.columns):
            sns.boxplot(x = col, y = 'quality', data = self.df, ax = axes[i])
            axes[i].set_title(f'Quality vs {col}', fontsize = 9)
            
        fig.tight_layout()
        self.guardar_grafico(fig, 'boxplot_outliers')
        
        
    def correlaciones(self):
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(self.df.corr(), annot = True, cmap = 'coolwarm', ax = ax)
        ax.set_title('Mapa de correlaciones')
        self.guardar_grafico(fig, 'mapa_correlaciones')
        
    def exportar(self):
        
        if self.df is not None:
            output_path = self.data_path.parent / 'wine_EDA.csv'
            self.df.to_csv(output_path, index = True)
            print(f'\n💾 Archivo exportado correctamente: {output_path}')
        else:
            print('⚠️ No hay datos cargados para exportar.')
            
    def ejecutar(self):
        self.load_data()
        self.clean_data()
        self.distribucion()
        self.graficar()
        self.outliers()
        self.correlaciones()
        self.exportar()
        print('\n✅ EDA completo finalizado correctamente.\n')   


eda = EDA("Wine/Data/winequality-red.csv")
eda.ejecutar()