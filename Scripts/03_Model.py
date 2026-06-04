import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif


df_wine = pd.read_csv('Wine\Data\wine.csv')
df_no = pd.read_csv('Wine\Data\wine_no.csv')

print('Data set CON outliers cargado')
print(f'\n {df_wine.shape}')

print('Data set SIN outliers cargado')
print(f'\n {df_no.shape}')

def models_value (df, features, target, test_size = 0.2, random_state = 42):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, shuffle = True, stratify = y)
    
    results = {}
    models = {
        'LogisticRegression' : make_pipeline(StandardScaler(), LogisticRegression(random_state = random_state, max_iter = 1000)),
        'RandomForest' : RandomForestClassifier(random_state = random_state),
        'XGBoost' : make_pipeline(StandardScaler(), XGBClassifier(random_state = random_state)),
        'SVC' : make_pipeline(StandardScaler(), SVC(probability = True, random_state = random_state))
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'ACC': acc,
                'AUC-ROC': roc,
                'F1-score': f1,
                'Precision': precision,
                'Recall': recall,
                'Confusion Matrix': cm
            }
            
            print(f'✅ {name} entrenado correctamente')
            
        except Exception as e:
            print(f'❌ Error en {name}: {str(e)}')
            results[name] = {    
                'ACC': 0,
                'AUC-ROC': 0,
                'F1-score': 0,
                'Precision': 0,
                'Recall': 0,
                'Confusion Matrix': np.array([[0, 0], [0, 0]])
                }
            
    return results

features = ['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target = 'quality'

r_no = models_value(df_no, features, target)
r_wine = models_value(df_wine, features, target)

def final_results(results, name_db):
  print(f'\n📊 Resultados para {name_db.upper()}: ')
  for model, metrics in results.items():
    print(f'\n {model}: ')
    print(f'ACC: {metrics['ACC']:.2f}') # Porcentaje de aciertos
    print(f'AUC-ROC: {metrics['AUC-ROC']:.2f}') #Capacidad para distinguir entre clases
    print(f'F1-score: {metrics['F1-score']:.2f}')
    print(f'Precision: {metrics['Precision']:.2f}')
    print(f'Recall: {metrics['Recall']:.2f}')
    
    cm = metrics['Confusion Matrix']
    print(f'   Matriz de Confusión:')
    print(f'   [{cm[0][0]:3d}  {cm[0][1]:3d}] → TN  FP')
    print(f'   [{cm[1][0]:3d}  {cm[1][1]:3d}] → FN  TP')

  best_model_acc = max(results, key=lambda x: results[x]['ACC'])
  print(f"\n🏆 MEJOR MODELO ({name_db}): {best_model_acc} (ACC = {results[best_model_acc]['ACC']:.3f})")

  best_model_auc = max(results, key=lambda x: results[x]['AUC-ROC'])
  print(f"\n🏆 MEJOR MODELO ({name_db}): {best_model_auc} (AUC-ROC = {results[best_model_auc]['AUC-ROC']:.3f})")

  best_model_f1 = max(results, key=lambda x: results[x]['F1-score'])
  print(f"\n🏆 MEJOR MODELO ({name_db}): {best_model_f1} (F1-score = {results[best_model_f1]['F1-score']:.3f})")

final_results(r_no, 'df_no')
final_results(r_wine, 'df_wine')

#--------------------------------------------
#         Optimización de Parámetros
#--------------------------------------------

mejor_dataset = df_no
mejor_modelo_nombre = "RandomForest"

print(f"Optimizando: {mejor_modelo_nombre}")

X = mejor_dataset[features]
y = mejor_dataset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True, stratify = y)

# Hiperparámetros simples para RandomForest
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

modelo_base = RandomForestClassifier(random_state=42)

# Búsqueda aleatoria
random_search = RandomizedSearchCV(
    modelo_base, 
    param_grid, 
    n_iter = 100,           
    cv = 10,               
    scoring = 'accuracy', 
    random_state = 42,
    n_jobs = -1,          
    verbose = 1           
)

print("Buscando mejores parámetros...")
random_search.fit(X_train, y_train)

# Mostrar resultados
print("✅ Mejores parámetros:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

# Evaluar
mejor_modelo = random_search.best_estimator_
y_pred = mejor_modelo.predict(X_test)
y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
pr = precision_score(y_test, y_pred)
rcl = recall_score(y_test, y_pred)

print(f"\n🎯 Accuracy del modelo optimizado: {acc:.4f}")
print(f"\n🎯 AUC-ROC del modelo optimizado: {roc:.4f}")
print(f"\n🎯 F1-scoring del modelo optimizado: {f1:.4f}")
print(f"\n🎯 Precision del modelo optimizado: {pr:.4f}")
print(f"\n🎯 Recall del modelo optimizado: {rcl:.4f}")


import joblib
from pathlib import Path

models_dir = Path("Wine/Models")
models_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Carpeta para modelos creada/verificada: {models_dir}")

def guardar_features(features_list):
    joblib.dump(features_list, models_dir / "features.pkl")
    print(f'✅ {len(features_list)} Features guardadas como: {features_list}')

def guardar_modelo(modelo, nombre_archivo):
    joblib.dump(modelo, models_dir / f'{nombre_archivo}.pkl')
    print(f"✅ Modelo guardado como: {nombre_archivo}.pkl")

def cargar_modelo_simple(nombre_archivo):
    modelo = joblib.load(models_dir / f'{nombre_archivo}.pkl')
    print("✅ Modelo cargado correctamente")
    return modelo

guardar_features(features)
guardar_modelo(mejor_modelo, "rf_wine")
modelo_cargado = cargar_modelo_simple("rf_wine")


# FEATURES MÁS IMPORTANTES 

importancias = mejor_modelo.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importancias
}).sort_values('importance', ascending=False)

print("\n🎯 FEATURES MÁS IMPORTANTES:")
for i, row in feature_importance_df.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# Seleccionar top features 
top_features = feature_importance_df.head(6)['feature'].tolist()

print(f"🔝 Top features seleccionadas: {len(top_features)}")
print(f"📋 Lista: {top_features}")

# Guardar las top features
joblib.dump(top_features, models_dir / "top_features.pkl")
