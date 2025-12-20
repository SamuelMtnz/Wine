import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




class Feature_Engineering:
    
    def __init__(self, ruta):
        self.ruta = ruta
        self.df = pd.read_csv(ruta)
        print(f'✅Datos cargados correctamente: {self.df.shape}')

    def multicolinealidad(self, unmbral = 0.65):
        corr_matrix = self.df.corr().abs()
        
        to_drop = ['fixed acidity', 'free sulfur dioxide']
        
        self.df = self.df.drop(columns = to_drop)
        
        plt.figure(figsize = (12, 12))
        sns.heatmap(self.df.corr(), annot = True, cmap = 'coolwarm')
        plt.title('Mapa de correlaciones reducido con umbral < 0.65')
        plt.show()
        
        print(f'\n✅ Variables eliminadas: {to_drop}')
        print(f'\n📊 Data set reducido: {self.df.shape}')
        
        return self.df
        
    def dec_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask_out = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        return mask_out

    def del_outliers(self):
        size = len(self.df)
        outliers = {}
        
        for col in self.df.columns:
            if col != 'quality':
                outliers[col] = self.dec_outliers(col)
                
        filas_out = pd.Series([False] * len(self.df))
        for col, mask in outliers.items():
            filas_out = filas_out | mask
            
        df_clean = self.df[~filas_out].copy()
        
        print(f"🧹 Eliminados {size - len(df_clean)} registros con outliers")
        print(f"📊 Dataset limpio: {df_clean.shape}")
        return df_clean

    def quality(self, df, tipo=''):
        df = df.copy()
        df['quality'] = df['quality'].apply(lambda x:1 if x >= 6 else 0)
        
        print(f'\nDistribucion {tipo} outliers: \n{df["quality"].value_counts()}')
        return df

    def save_db(self, df, ruta):
        df.to_csv(ruta, index = False)
        print(f'\n💾 Dataset guardado en: {ruta}')

    def dis_outliers(self, df):
        plt.figure(figsize = (12, 16))
        var = [col for col in df.columns if col != 'quality']
        
        for i, col in enumerate(var, 1):
            plt.subplot(6, 2, i)
            
            sns.histplot(df[col], kde = True, color = 'blue', label = 'Datos', bins = 20)
            
            mask_out = self.dec_outliers(col)
            out_data = df[mask_out][col]
            
            if len(out_data) > 0:
                sns.histplot(out_data, color = 'red', label = 'Outliers', alpha = 0.7, bins = 10)
                plt.title(f'Distribución de {col}')
                
            plt.xlabel(col)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def run_pipeline(self,
                     ruta_sin_outliers='Wine/Data/wine_no.csv',
                     ruta_con_outliers='Wine/Data/wine.csv'):
        """Ejecuta el flujo completo de feature engineering."""

        # 1. Multicolinealidad
        df_reduce = self.multicolinealidad()

        # 2. Outliers
        df_clean = self.del_outliers()

        # 3. Binarización de 'quality'
        df_clean = self.quality(df_clean, 'SIN')
        df_reduce = self.quality(df_reduce, 'CON')

        # 4. Guardar datasets
        self.save_db(df_clean, ruta_sin_outliers)
        self.save_db(df_reduce, ruta_con_outliers)

        # 5. Visualizar distribuciones
        self.dis_outliers(df_reduce)

        return df_reduce, df_clean
        
if __name__ == '__main__':
    fe = Feature_Engineering('Wine/Data/wine_EDA.csv')
    fe.run_pipeline()