# src/models/meta_model.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

from ..data_ingestion.db_config import get_engine
from .poisson_model import PoissonModel
from .ml_pipeline import FeatureEngineer as MLFeatureEngineer
from .dl_pipeline import DLDataPreprocessor

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
SEQUENCE_LENGTH = 5


class MetaModelTrainer:
    def __init__(self, engine):
        self.engine = engine
        self._load_base_models()

    def _load_base_models(self):
        print("Cargando todos los modelos base entrenados...")
        self.ml_models = {
            "logistic_regression": joblib.load(ARTIFACTS_DIR / "logistic_regression_model.joblib"),
            "random_forest": joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib"),
            "xgboost": joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib")
        }
        self.label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
        self.dl_model = tf.keras.models.load_model(ARTIFACTS_DIR / "deep_learning_model.h5")
        with open(ARTIFACTS_DIR / 'team_mapping.json', 'r') as f:
            self.team_mapping = json.load(f)
        self.poisson_model = PoissonModel(engine=self.engine)
        self.poisson_model.train()
        print(" Todos los modelos base han sido cargados.")

    def _generate_meta_dataset(self, df_holdout):
        print(f"Generando predicciones base sobre {len(df_holdout)} partidos de holdout...")

        # --- 1. Predicciones del Modelo Poisson ---
        poisson_preds = []
        for _, row in df_holdout.iterrows():
            pred = self.poisson_model.predict(row['home_team'], row['away_team'])
            if pred:
                probs = pred['prediccion']
                poisson_preds.append([
                    probs['victoria_local'] / 100,
                    probs['empate'] / 100,
                    probs['victoria_visitante'] / 100
                ])
            else:
                poisson_preds.append([np.nan, np.nan, np.nan])

        # --- 2. Predicciones de los Modelos de ML ---
        ml_features = [
            'home_avg_scored', 'home_avg_conceded',
            'away_avg_scored', 'away_avg_conceded',
            'diff_avg_scored', 'diff_avg_conceded'
        ]
        X_ml_holdout = df_holdout[ml_features]
        ml_preds = {}
        for name, model in self.ml_models.items():
            ml_preds[name] = model.predict_proba(X_ml_holdout)

        # --- 3. Predicciones del Modelo de DL ---
        home_team_ids = df_holdout['home_team'].map(self.team_mapping).values
        away_team_ids = df_holdout['away_team'].map(self.team_mapping).values
        home_seqs = np.array(df_holdout['home_sequence'].tolist(), dtype=np.int32)
        away_seqs = np.array(df_holdout['away_sequence'].tolist(), dtype=np.int32)

        X_dl_holdout = [home_team_ids, away_team_ids, home_seqs, away_seqs]
        dl_preds = self.dl_model.predict(X_dl_holdout)

        # --- 4. Ensambla el Meta-Dataset ---
        meta_features = np.hstack([
            np.array(poisson_preds),
            ml_preds['logistic_regression'],
            ml_preds['random_forest'],
            ml_preds['xgboost'],
            dl_preds
        ])

        meta_df = pd.DataFrame(meta_features, index=df_holdout.index)
        meta_df.columns = (
            [f'poisson_{c}' for c in ['H', 'D', 'A']] +
            [f'lr_{c}' for c in ['A', 'D', 'H']] +
            [f'rf_{c}' for c in ['A', 'D', 'H']] +
            [f'xgb_{c}' for c in ['A', 'D', 'H']] +
            [f'dl_{c}' for c in ['H', 'D', 'A']]
        )
        meta_df.dropna(inplace=True)
        meta_target = df_holdout.loc[meta_df.index, 'result']

        print(f" Meta-dataset creado con {meta_df.shape[1]} características y {len(meta_df)} filas.")
        return meta_df, meta_target

    def train(self):
        print("\n--- Iniciando Entrenamiento del Meta-Modelo ---")
        print("Preparando dataset completo para la división...")

        ml_fe = MLFeatureEngineer(self.engine)
        df_ml_base = ml_fe.create_features()

        dl_preprocessor = DLDataPreprocessor(self.engine)
        dl_features_dict = dl_preprocessor.create_sequences()

        team_map_inv = {v: k for k, v in self.team_mapping.items()}
        target_map_inv = {0: 'H', 1: 'D', 2: 'A'}

        df_dl_reconstructed = pd.DataFrame({
            'home_team': pd.Series(dl_features_dict['home_team']).map(team_map_inv),
            'away_team': pd.Series(dl_features_dict['away_team']).map(team_map_inv),
            'result_code': np.argmax(dl_features_dict['target'], axis=1),
            'home_sequence': list(dl_features_dict['home_sequence']),
            'away_sequence': list(dl_features_dict['away_sequence'])
        })
        df_dl_reconstructed['result'] = df_dl_reconstructed['result_code'].map(target_map_inv)

        
        df_ml_base_reset = df_ml_base.reset_index().rename(columns={'index': 'id'})
        df_full_features = pd.merge(
            df_ml_base_reset,
            df_dl_reconstructed,
            on=['home_team', 'away_team', 'result'],
            how='inner'
        )

        if 'id' not in df_full_features.columns:
            df_full_features['id'] = range(len(df_full_features))

        df_full_features.drop_duplicates(subset=['id'], inplace=True)

        if df_full_features.empty:
            print(" ERROR: No se pudieron alinear los datasets de ML y DL. Abortando.")
            return

        print(f" Datasets alineados. {len(df_full_features)} partidos comunes encontrados.")

        combined_path = ARTIFACTS_DIR / "combined_features.csv"
        df_full_features.to_csv(combined_path, index=False)
        print(f" Dataset combinado guardado en: {combined_path}")

        _, df_holdout = train_test_split(
            df_full_features,
            test_size=0.2,
            random_state=42,
            stratify=df_full_features['result']
        )

        # Crear dataset meta
        X_meta, y_meta = self._generate_meta_dataset(df_holdout)

        print("\nEntrenando meta-modelo (Logistic Regression)...")
        meta_model = LogisticRegression(max_iter=1000, random_state=42)

        y_meta_encoded = self.label_encoder.transform(y_meta)
        meta_model.fit(X_meta, y_meta_encoded)

        preds = meta_model.predict(X_meta)
        acc = accuracy_score(y_meta_encoded, preds)
        print(f" Precisión (Accuracy) del Meta-Modelo: {acc:.4f}")

        model_path = ARTIFACTS_DIR / "meta_model.joblib"
        joblib.dump(meta_model, model_path)
        print(f" Meta-Modelo final guardado en: {model_path}")


if __name__ == '__main__':
    db_engine = get_engine()
    if db_engine:
        meta_trainer = MetaModelTrainer(engine=db_engine)
        meta_trainer.train()
