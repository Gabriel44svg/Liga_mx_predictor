# src/models/dl_pipeline.py

import pandas as pd
import numpy as np
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from ..data_ingestion.db_config import get_engine

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
SEQUENCE_LENGTH = 5

class DLDataPreprocessor:
    """
    Prepara los datos para el modelo de Deep Learning. (VERSIÓN FINAL Y ROBUSTA)
    """
    def __init__(self, engine):
        self.engine = engine
        self.team_mapping = {}
        self.target_mapping = {'H': 0, 'D': 1, 'A': 2} 

    def create_sequences(self):
        print("Cargando y pre-procesando datos para DL (Método Definitivo)...")
        query = "SELECT * FROM partidos ORDER BY date;"
        df = pd.read_sql(query, self.engine)
        
        df['home_team'] = df['home_team'].astype(str).str.strip()
        df['away_team'] = df['away_team'].astype(str).str.strip()
        
        all_teams = sorted(pd.unique(df[['home_team', 'away_team']].values.ravel('K')))
        self.team_mapping = {team: i for i, team in enumerate(all_teams)}
        
        with open(ARTIFACTS_DIR / 'team_mapping.json', 'w') as f:
            json.dump(self.team_mapping, f)
        print(f"Se cargaron {len(df)} partidos. {len(all_teams)} equipos únicos encontrados.")

        df_reset = df.reset_index().rename(columns={'index': 'original_index'})
        
        home_df = df_reset[['original_index', 'date', 'home_team', 'result']].rename(columns={'home_team': 'team'})
        away_df = df_reset[['original_index', 'date', 'away_team', 'result']].rename(columns={'away_team': 'team'})
        home_df['is_home'] = 1
        away_df['is_home'] = 0
        
        home_df['team_result_code'] = home_df['result'].map({'H': 2, 'D': 1, 'A': 0}) # Win, Draw, Loss
        away_df['team_result_code'] = away_df['result'].map({'H': 0, 'D': 1, 'A': 2}) # Loss, Draw, Win
        
        team_appearances = pd.concat([home_df, away_df]).sort_values(by=['team', 'date'])
        
        print("Creando secuencias de forma para cada equipo...")

        def get_sequences(group):
            shifted = group['team_result_code'].shift(1)
            sequences = []
            for i in range(len(group)):
                if i < (SEQUENCE_LENGTH - 1):
                    sequences.append(None)
                    continue
                window = shifted.iloc[i - (SEQUENCE_LENGTH - 1) : i + 1]
                if window.isnull().any():
                    sequences.append(None)
                else:
                    sequences.append(window.tolist())
            return pd.Series(sequences, index=group.index)
        
        sequences = team_appearances.groupby('team', group_keys=False).apply(get_sequences)
        team_appearances['sequence'] = sequences
        
        home_app = team_appearances[team_appearances['is_home'] == 1]
        away_app = team_appearances[team_appearances['is_home'] == 0]
        
        final_df = df.reset_index().merge(
            home_app[['original_index', 'sequence']],
            left_on='index', right_on='original_index'
        ).rename(columns={'sequence': 'home_sequence'})
        
        final_df = final_df.merge(
            away_app[['original_index', 'sequence']],
            left_on='index', right_on='original_index'
        ).rename(columns={'sequence': 'away_sequence'})
        
        final_df.dropna(subset=['home_sequence', 'away_sequence'], inplace=True)
        print(f"Se crearon {len(final_df)} puntos de datos para el entrenamiento.")

        home_team_ids = final_df['home_team'].map(self.team_mapping).values
        away_team_ids = final_df['away_team'].map(self.team_mapping).values
        home_seqs = np.array(final_df['home_sequence'].tolist(), dtype=np.int32)
        away_seqs = np.array(final_df['away_sequence'].tolist(), dtype=np.int32)
        targets = final_df['result'].map(self.target_mapping).values
        
        return {
            "home_team": home_team_ids,
            "away_team": away_team_ids,
            "home_sequence": home_seqs,
            "away_sequence": away_seqs,
            "target": to_categorical(targets, num_classes=3)
        }

class DLModelTrainer:
    def __init__(self, data, team_count):
        self.data = data
        self.team_count = team_count
        self.model = self._build_model()

    def _build_model(self, embedding_dim=10, lstm_units=16):
        print("Construyendo el modelo de Deep Learning...")
        home_team_in = Input(shape=(1,), name='home_team')
        away_team_in = Input(shape=(1,), name='away_team')
        home_seq_in = Input(shape=(SEQUENCE_LENGTH,), name='home_sequence')
        away_seq_in = Input(shape=(SEQUENCE_LENGTH,), name='away_sequence')
        
        team_embedding = Embedding(input_dim=self.team_count, output_dim=embedding_dim, name='team_embedding')
        home_embedded = team_embedding(home_team_in)
        away_embedded = team_embedding(away_team_in)
        
        seq_embedding = Embedding(input_dim=3, output_dim=5, name='sequence_embedding')
        home_seq_embedded = seq_embedding(home_seq_in)
        away_seq_embedded = seq_embedding(away_seq_in)
        
        lstm_layer = LSTM(lstm_units, name='lstm_layer')
        home_lstm_out = lstm_layer(home_seq_embedded)
        away_lstm_out = lstm_layer(away_seq_embedded)
        
        concatenated = Concatenate()([
            tf.keras.layers.Flatten()(home_embedded),
            tf.keras.layers.Flatten()(away_embedded),
            home_lstm_out,
            away_lstm_out
        ])
        
        dense_1 = Dense(64, activation='relu')(concatenated)
        dropout_1 = Dropout(0.5)(dense_1)
        output = Dense(3, activation='softmax', name='output')(dropout_1)
        
        model = Model(
            inputs=[home_team_in, away_team_in, home_seq_in, away_seq_in],
            outputs=output
        )
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, epochs=20, batch_size=32):
        print("\n--- Iniciando Entrenamiento del Modelo de Deep Learning ---")
        
        
        # 1. Definir todas las entradas y la salida
        home_team_X = self.data['home_team']
        away_team_X = self.data['away_team']
        home_seq_X = self.data['home_sequence']
        away_seq_X = self.data['away_sequence']
        y = self.data['target']
        
        # 2. Divide todos los arrays a la vez.
        (home_team_train, home_team_val,
         away_team_train, away_team_val,
         home_seq_train, home_seq_val,
         away_seq_train, away_seq_val,
         y_train, y_val) = train_test_split(
            home_team_X, away_team_X, home_seq_X, away_seq_X, y,
            test_size=0.2, random_state=42
        )

        # 3. Re-agrupa las entradas en listas para el entrenamiento y la validación
        X_train = [home_team_train, away_team_train, home_seq_train, away_seq_train]
        X_val = [home_team_val, away_team_val, home_seq_val, away_seq_val]

        # 4. Entrena el modelo con los datos ya organizados
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        model_path = ARTIFACTS_DIR / "deep_learning_model.h5"
        self.model.save(model_path)
        print(f"\n Modelo de Deep Learning guardado en: {model_path}")
        return history
        

if __name__ == '__main__':
    db_engine = get_engine()
    if db_engine:
        preprocessor = DLDataPreprocessor(engine=db_engine)
        dataset = preprocessor.create_sequences()
        
        if dataset['target'].shape[0] > 0:
            trainer = DLModelTrainer(data=dataset, team_count=len(preprocessor.team_mapping))
            trainer.train()
            print("\n¡Pipeline de Deep Learning completado!")
        else:
            print("No se generaron suficientes datos para el entrenamiento. Considera un historial de partidos más grande.")