# src/models/ml_pipeline.py

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from ..data_ingestion.db_config import get_engine

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True) # Crea la carpeta si no existe

class FeatureEngineer:
    """
    Clase responsable de crear las características para los modelos de ML.
    """
    def __init__(self, engine, rolling_window=5):
        self.engine = engine
        self.rolling_window = rolling_window
        self.raw_data = self._load_data()
        self.feature_data = None

    def _load_data(self) -> pd.DataFrame:
        """Carga los datos de la tabla 'partidos'."""
        print("Cargando datos para Ingeniería de Características...")
        query = "SELECT * FROM partidos ORDER BY date;"
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Se cargaron {len(df)} partidos, ordenados por fecha.")
        return df

    def create_features(self):
        """
        Orquesta la creación de todas las características. (VERSIÓN FINAL CORREGIDA)
        """
        print(f"Iniciando creación de características con una ventana de {self.rolling_window} partidos...")
        df = self.raw_data.copy()

        # --- Característica 1: Forma Reciente (Rolling Averages) ---
        # Usamos .transform() para aplicar la media móvil dentro de cada grupo y mantener el índice original.
        
        # Goles anotados y recibidos por el equipo LOCAL en sus últimos partidos (local o visitante)
        df['home_avg_scored'] = df.groupby('home_team')['home_goals'].transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=1).mean()
        )
        df['home_avg_conceded'] = df.groupby('home_team')['away_goals'].transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=1).mean()
        )
        
        # Goles anotados y recibidos por el equipo VISITANTE en sus últimos partidos (local o visitante)
        df['away_avg_scored'] = df.groupby('away_team')['away_goals'].transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=1).mean()
        )
        df['away_avg_conceded'] = df.groupby('away_team')['home_goals'].transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=1).mean()
        )
        
        # Eliminar partidos con datos nulos (los primeros partidos donde no hay historial suficiente)
        df.dropna(inplace=True)
        
        # --- Característica 2: Diferenciales ---
        df['diff_avg_scored'] = df['home_avg_scored'] - df['away_avg_scored']
        df['diff_avg_conceded'] = df['home_avg_conceded'] - df['away_avg_conceded']
        
        print(f"Ingeniería de Características completa. Dataset final con {len(df)} partidos listos para entrenar.")
        self.feature_data = df
        return self.feature_data

class ModelTrainer:
    """
    Clase para entrenar, evaluar y guardar los modelos de ML.
    """
    def __init__(self, feature_df: pd.DataFrame):
        self.df = feature_df
        # Definimos los modelos que vamos a entrenar
        self.models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "random_forest": RandomForestClassifier(random_state=42),
            "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        }
        self.trained_models = {}

    def train_and_evaluate(self):
        """
        Ejecuta el pipeline de entrenamiento y evaluación.
        """
        print("\n--- Iniciando Proceso de Entrenamiento y Evaluación ---")
        
        # 1. Definir Features (X) y Target (y)
        features = ['home_avg_scored', 'home_avg_conceded', 'away_avg_scored', 'away_avg_conceded', 'diff_avg_scored', 'diff_avg_conceded']
        X = self.df[features]
        
        # El target 'result' ('H', 'D', 'A') es categórico, hay que codificarlo
        le = LabelEncoder()
        y = le.fit_transform(self.df['result'])
        # Guardamos el encoder para poder decodificar las predicciones después
        joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")
        print(f"Target codificado. Clases: {le.classes_}")

        # 2. Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

        # 3. Entrenar y evaluar cada modelo
        for name, model in self.models.items():
            print(f"\nEntrenando modelo: {name}...")
            model.fit(X_train, y_train)
            
            # Evaluación
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f" Precisión (Accuracy) de {name}: {acc:.4f}")
            
            # Guarda el modelo entrenado
            model_path = ARTIFACTS_DIR / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            print(f"Modelo guardado en: {model_path}")
            
            self.trained_models[name] = model

if __name__ == '__main__':
    db_engine = get_engine()
    if db_engine:
        # 1. Crear las características
        feature_engineer = FeatureEngineer(engine=db_engine, rolling_window=5)
        feature_dataset = feature_engineer.create_features()
        
        if feature_dataset is not None and not feature_dataset.empty:
            # 2. Entrenar los modelos con las características creadas
            trainer = ModelTrainer(feature_df=feature_dataset)
            trainer.train_and_evaluate()
            print("\n¡Pipeline de Machine Learning completado!")