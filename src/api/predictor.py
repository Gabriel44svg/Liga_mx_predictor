# src/api/predictor.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import tensorflow as tf

from ..data_ingestion.db_config import get_engine
from ..models.poisson_model import PoissonModel

# --- CORRECCIÓN AQUÍ ---
# La ruta correcta tiene 3 .parent, no 4
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
SEQUENCE_LENGTH = 5

class Predictor:
    """
    Clase que encapsula toda la lógica para cargar modelos y realizar una predicción.
    """
    def __init__(self):
        print("Inicializando el Predictor y cargando todos los modelos...")
        self.engine = get_engine()
        self._load_models()
        self.all_data = self._load_all_data()
        print("✅ Predictor inicializado correctamente.")

    def _load_models(self):
        """Carga todos los artefactos de modelo necesarios."""
        self.ml_models = {
            "logistic_regression": joblib.load(ARTIFACTS_DIR / "logistic_regression_model.joblib"),
            "random_forest": joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib"),
            "xgboost": joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib")
        }
        self.label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
        self.dl_model = tf.keras.models.load_model(ARTIFACTS_DIR / "deep_learning_model.h5")
        self.meta_model = joblib.load(ARTIFACTS_DIR / "meta_model.joblib")
        
        with open(ARTIFACTS_DIR / 'team_mapping.json', 'r') as f:
            self.team_mapping = json.load(f)
            
        self.poisson_model = PoissonModel(engine=self.engine)
        self.poisson_model.train()
        
    def _load_all_data(self):
        """Carga todos los partidos para calcular características al vuelo."""
        return pd.read_sql("SELECT * FROM partidos ORDER BY date DESC", self.engine)

    def predict_single_match(self, home_team: str, away_team: str):
        """
        Realiza una predicción completa para un solo partido.
        """
        # --- 1. Generar predicción de Poisson ---
        poisson_pred = self.poisson_model.predict(home_team, away_team)
        if not poisson_pred:
            raise ValueError("No se pudo generar la predicción de Poisson.")
        poisson_probs = np.array([[
            poisson_pred['prediccion']['victoria_local'] / 100,
            poisson_pred['prediccion']['empate'] / 100,
            poisson_pred['prediccion']['victoria_visitante'] / 100
        ]])
        
        # --- 2. Generar características y predicciones de ML ---
        home_games = self.all_data[(self.all_data['home_team'] == home_team) | (self.all_data['away_team'] == home_team)].head(5)
        home_goals_scored = home_games.apply(lambda row: row['home_goals'] if row['home_team'] == home_team else row['away_goals'], axis=1).mean()
        home_goals_conceded = home_games.apply(lambda row: row['away_goals'] if row['home_team'] == home_team else row['home_goals'], axis=1).mean()
        
        away_games = self.all_data[(self.all_data['home_team'] == away_team) | (self.all_data['away_team'] == away_team)].head(5)
        away_goals_scored = away_games.apply(lambda row: row['home_goals'] if row['home_team'] == away_team else row['away_goals'], axis=1).mean()
        away_goals_conceded = away_games.apply(lambda row: row['away_goals'] if row['home_team'] == away_team else row['home_goals'], axis=1).mean()
        
        X_ml = pd.DataFrame([{
            'home_avg_scored': home_goals_scored,
            'home_avg_conceded': home_goals_conceded,
            'away_avg_scored': away_goals_scored,
            'away_avg_conceded': away_goals_conceded,
            'diff_avg_scored': home_goals_scored - away_goals_scored,
            'diff_avg_conceded': home_goals_conceded - away_goals_conceded
        }])
        
        ml_preds = {}
        for name, model in self.ml_models.items():
            ml_preds[name] = model.predict_proba(X_ml)
            
        # --- 3. Generar secuencias y predicción de DL ---
        result_map = {'H': 2, 'D': 1, 'A': 0}
        
        def get_team_sequence(team_name):
            games = self.all_data[(self.all_data['home_team'] == team_name) | (self.all_data['away_team'] == team_name)].head(SEQUENCE_LENGTH)
            seq = []
            for _, row in games.iterrows():
                if (row['home_team'] == team_name and row['result'] == 'H') or \
                   (row['away_team'] == team_name and row['result'] == 'A'):
                    seq.append(result_map['H']) # Win
                elif row['result'] == 'D':
                    seq.append(result_map['D']) # Draw
                else:
                    seq.append(result_map['A']) # Loss
            return seq

        home_seq = get_team_sequence(home_team)
        away_seq = get_team_sequence(away_team)
        
        if len(home_seq) < SEQUENCE_LENGTH or len(away_seq) < SEQUENCE_LENGTH:
            raise ValueError("No hay suficiente historial para generar secuencias para uno de los equipos.")

        X_dl = [
            np.array([self.team_mapping[home_team]]),
            np.array([self.team_mapping[away_team]]),
            np.array([home_seq]),
            np.array([away_seq])
        ]
        dl_preds = self.dl_model.predict(X_dl, verbose=0)
        
        # --- 4. Ensamblar características y predecir con el Meta-Modelo ---
        meta_features = np.hstack([
            poisson_probs,
            ml_preds['logistic_regression'],
            ml_preds['random_forest'],
            ml_preds['xgboost'],
            dl_preds
        ])
        
        meta_prediction_probs = self.meta_model.predict_proba(meta_features)[0]
        winner_class_index = np.argmax(meta_prediction_probs)
        winner = self.label_encoder.inverse_transform([winner_class_index])[0]

        prob_map = {
            'A': meta_prediction_probs[0],
            'D': meta_prediction_probs[1],
            'H': meta_prediction_probs[2]
        }
        
        return {
            "winner": winner,
            "home_win_prob": round(prob_map.get('H', 0) * 100, 2),
            "draw_prob": round(prob_map.get('D', 0) * 100, 2),
            "away_win_prob": round(prob_map.get('A', 0) * 100, 2),
            "poisson_most_likely_score": poisson_pred['marcador_mas_probable']
        }