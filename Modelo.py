import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
import io

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from google.colab import files


file_name_kep = 'cumulative_2025.10.01_10.25.26.csv'
df_kep = pd.read_csv(file_name_kep, skiprows=53, sep=',')

url_tess = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/ncat/ncat_handler.py?objid=TESS_TOI_CATALOG&output=csv'
url_k2 = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/ncat/ncat_handler.py?objid=K2_CANDIDATE_CATALOG&output=csv'

def download_and_load(url):
    response = requests.get(url)
    df_raw = pd.read_csv(io.StringIO(response.content.decode('utf-8')), comment='#')
    return df_raw

try:
    print("Descargando TESS (TOI)...")
    df_tess = download_and_load(url_tess)
    print("Descargando K2...")
    df_k2 = download_and_load(url_k2)

    print(f"\nDatos Kepler: {len(df_kep)} filas. TESS: {len(df_tess)} filas. K2: {len(df_k2)} filas.")

except Exception as e:
    print(f"Error al descargar datos TESS/K2. Solo se usará Kepler. Error: {e}")
    df_tess = pd.DataFrame()
    df_k2 = pd.DataFrame()


print(f"\nDatos Kepler cargados con {len(df_kep)} filas.")
print(f"Columnas de Kepler disponibles: {df_kep.columns.tolist()[:10]}...")



COL_MAP = {
    'koi_disposition': 'target_disposition', 'tfowpg_disposition': 'target_disposition', 'k2c_disp': 'target_disposition',
    'koi_period': 'period', 'toi_period': 'period', 'k2c_period': 'period',
    'koi_duration': 'duration', 'toi_duration': 'duration', 'k2c_dur': 'duration',
    'koi_impact': 'impact', 'toi_impact': 'impact', 'k2c_impact': 'impact',
    'koi_depth': 'depth', 'toi_depth': 'depth', 'k2c_depth': 'depth',
    'koi_teq': 'teq', 'toi_teq': 'teq', 'k2c_teq': 'teq',
    'koi_prad': 'prad', 'toi_prad': 'prad', 'k2c_prad': 'prad',
    'koi_slogg': 'slogg', 'toi_slogg': 'slogg', 'k2c_slogg': 'slogg',
    'koi_steff': 'steff', 'toi_steff': 'steff', 'k2c_steff': 'steff',
    'koi_srad': 'srad', 'toi_srad': 'srad', 'k2c_srad': 'srad',
}


def clean_and_rename(df, source):
    df.rename(columns=COL_MAP, inplace=True)
    cols_to_keep = list(set(COL_MAP.values()))
    df = df.filter(items=cols_to_keep)
    df['source'] = source
    return df


df_kep_clean = clean_and_rename(df_kep, 'Kepler')
if not df_tess.empty:
    df_tess_clean = clean_and_rename(df_tess, 'TESS')
    df_k2_clean = clean_and_rename(df_k2, 'K2')
    df_combined = pd.concat([df_kep_clean, df_tess_clean, df_k2_clean], ignore_index=True)
else:
    df_combined = df_kep_clean

print(f"\nDataFrame combinado total: {len(df_combined)} filas.")


FEATURES_HARMONIZED = ['period', 'duration', 'impact', 'depth', 'teq', 'prad', 'slogg', 'steff', 'srad']


df_combined['target'] = df_combined['target_disposition'].map({
    'CONFIRMED': 1, 'CANDIDATE': 2, 'PC': 2, 'KP': 1,
    'FALSE POSITIVE': 0, 'FP': 0, 'VETTED FALSE POSITIVE': 0,
})

df_combined.dropna(subset=['target'], inplace=True)
df_combined['target'] = df_combined['target'].astype(int)


X = df_combined[FEATURES_HARMONIZED]
y = df_combined['target']


preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDatos de entrenamiento: {len(X_train)} filas.")
print(f"Datos de prueba: {len(X_test)} filas.")

import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=500,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    num_boost_round=2000,
)

print("\nIniciando entrenamiento del modelo XGBoost...")
model.fit(X_train, y_train)
print("¡Entrenamiento completado!")

y_pred = model.predict(X_test)
target_names = ['FP (0)', 'CONFIRMED (1)', 'CANDIDATE (2)']

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'recall_confirmed': recall_score(y_test, y_pred, labels=[1], average='micro'),
    'precision_fp': precision_score(y_test, y_pred, labels=[0], average='micro'),
}
joblib.dump(metrics, 'model_metrics.pkl')
print("\nArchivos de métricas guardados exitosamente.")

feature_importances = model.get_booster().get_fscore()
importance_list = sorted([
    (f, score) for f, score in feature_importances.items()
], key=lambda item: item[1], reverse=True)
joblib.dump(importance_list, 'feature_importances.pkl')
print("Archivos de importancia guardados exitosamente.")

print("\n--- Resultados del Modelo ---")
print(f"Accuracy Score (Precisión General): {metrics['accuracy']:.4f}")
print("\nClassification Report (Métricas clave):\n",
      classification_report(y_test, y_pred, target_names=target_names))

print("\nMatriz de Confusión (Filas=Real, Columnas=Predicción):\n",
      confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax, title="Importancia de las Características Astrofísicas (F-score)")
plt.show()

joblib.dump(model, 'exoplanet_classifier.pkl')
joblib.dump(preprocessor, 'exoplanet_preprocessor.pkl')

files.download('feature_importances.pkl')

print("\nArchivos .pkl guardados exitosamente en Colab.")

files.download('exoplanet_classifier.pkl')
files.download('exoplanet_preprocessor.pkl')
