# src/models/poisson_model.py

import pandas as pd
from scipy.stats import poisson
import numpy as np

from ..data_ingestion.db_config import get_engine

class PoissonModel:
    def __init__(self, engine):
        """
        Inicializa el modelo cargando los datos históricos desde la base de datos.
        """
        self.engine = engine
        self.historical_data = self._load_data()
        self.team_strengths = None

    def _load_data(self) -> pd.DataFrame:
        """Carga los datos de la tabla 'partidos' en un DataFrame."""
        print("Cargando datos históricos desde la base de datos...")
        try:
            query = "SELECT * FROM partidos;"
            df = pd.read_sql(query, self.engine)
            print(f"Se cargaron {len(df)} partidos.")
            return df
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return pd.DataFrame()

    def train(self):
        """
        Calcula las fuerzas de ataque y defensa para cada equipo.
        Este es el "entrenamiento" de nuestro modelo estadístico.
        """
        if self.historical_data.empty:
            print("No hay datos para entrenar el modelo.")
            return

        print("Calculando fuerzas de equipo (entrenamiento)...")
        df = self.historical_data.copy()

        avg_home_goals = df['home_goals'].mean()
        avg_away_goals = df['away_goals'].mean()

        #  Calcula la fuerza de ataque y defensa para cada equipo
        # Fuerza de Ataque = Promedio de goles del equipo / Promedio de goles de la liga
        # Fuerza de Defensa = Promedio de goles concedidos por el equipo / Promedio de goles de la liga
        
        home_stats = df.groupby('home_team').agg(
            avg_goals_scored_home=('home_goals', 'mean'),
            avg_goals_conceded_home=('away_goals', 'mean')
        ).reset_index().rename(columns={'home_team': 'team'})

        away_stats = df.groupby('away_team').agg(
            avg_goals_scored_away=('away_goals', 'mean'),
            avg_goals_conceded_away=('home_goals', 'mean')
        ).reset_index().rename(columns={'away_team': 'team'})
        
        team_strengths = pd.merge(home_stats, away_stats, on='team', how='outer').fillna(0)

        # Calculamos las fuerzas relativas a la media de la liga
        team_strengths['attack_strength_home'] = team_strengths['avg_goals_scored_home'] / avg_home_goals
        team_strengths['defense_strength_home'] = team_strengths['avg_goals_conceded_home'] / avg_away_goals
        team_strengths['attack_strength_away'] = team_strengths['avg_goals_scored_away'] / avg_away_goals
        team_strengths['defense_strength_away'] = team_strengths['avg_goals_conceded_away'] / avg_home_goals

        self.team_strengths = team_strengths.set_index('team')
        print("Fuerzas de equipo calculadas y almacenadas.")

    def predict(self, home_team: str, away_team: str, max_goals=10):
        """
        Predice el resultado de un partido entre dos equipos.
        """
        if self.team_strengths is None:
            print("El modelo no ha sido entrenado. Por favor, ejecuta .train() primero.")
            return None

        if home_team not in self.team_strengths.index or away_team not in self.team_strengths.index:
            print("Error: Uno o ambos equipos no se encontraron en los datos históricos.")
            return None
            
        # Obtiene las fuerzas de los equipos involucrados
        home_attack = self.team_strengths.at[home_team, 'attack_strength_home']
        home_defense = self.team_strengths.at[home_team, 'defense_strength_home']
        away_attack = self.team_strengths.at[away_team, 'attack_strength_away']
        away_defense = self.team_strengths.at[away_team, 'defense_strength_away']
        
        # Promedios de la liga
        avg_home_goals = self.historical_data['home_goals'].mean()
        avg_away_goals = self.historical_data['away_goals'].mean()

        # 2. Calcular los goles esperados (lambda) para cada equipo
        # lambda_home = Ataque_Local * Defensa_Visitante * Promedio_Goles_Local_Liga
        lambda_home = home_attack * away_defense * avg_home_goals
        lambda_away = away_attack * home_defense * avg_away_goals

        print(f"Goles esperados: {home_team} {lambda_home:.2f} - {lambda_away:.2f} {away_team}")

        # 3. Generar la matriz de probabilidades
        prob_home_win, prob_draw, prob_away_win = 0, 0, 0
        max_prob = 0
        most_likely_score = (0, 0)
        
        # Iteramos sobre los posibles goles para cada equipo
        for h_goals in range(max_goals + 1):
            for a_goals in range(max_goals + 1):
                # Probabilidad de que el local anote h_goals
                prob_h = poisson.pmf(h_goals, lambda_home)
                # Probabilidad de que el visitante anote a_goals
                prob_a = poisson.pmf(a_goals, lambda_away)
                
                # La probabilidad conjunta es el producto
                prob_match = prob_h * prob_a

                # Sumamos a la categoría correspondiente
                if h_goals > a_goals:
                    prob_home_win += prob_match
                elif h_goals == a_goals:
                    prob_draw += prob_match
                else:
                    prob_away_win += prob_match
                
                # Guardamos el marcador con la probabilidad más alta
                if prob_match > max_prob:
                    max_prob = prob_match
                    most_likely_score = (h_goals, a_goals)

        # Normalizamos para que la suma sea 100%
        total_prob = prob_home_win + prob_draw + prob_away_win
        prob_home_win /= total_prob
        prob_draw /= total_prob
        prob_away_win /= total_prob

        return {
            "prediccion": {
                "victoria_local": round(prob_home_win * 100, 2),
                "empate": round(prob_draw * 100, 2),
                "victoria_visitante": round(prob_away_win * 100, 2),
            },
            "marcador_mas_probable": f"{most_likely_score[0]} - {most_likely_score[1]}",
            "prob_marcador": round(max_prob * 100, 2)
        }

if __name__ == '__main__':
    db_engine = get_engine()

    if db_engine:
        model = PoissonModel(engine=db_engine)

        model.train()
        
        if model.team_strengths is not None:
            print("\n Equipos disponibles en la base de datos:", model.team_strengths.index.tolist())
        
        home = "COPIA UN EQUIPO DE LA LISTA AQUÍ"
        away = "COPIA OTRO EQUIPO DE LA LISTA AQUÍ"
        
        prediction = model.predict(home, away)
        
        if prediction:
            print("\n---  PREDICCIÓN DEL PARTIDO  ---")
            print(f"Partido: {home} vs {away}")
            print(f"Probabilidad Victoria Local ({home}): {prediction['prediccion']['victoria_local']}%")
            print(f"Probabilidad Empate: {prediction['prediccion']['empate']}%")
            print(f"Probabilidad Victoria Visitante ({away}): {prediction['prediccion']['victoria_visitante']}%")
            print(f"Marcador más probable: {prediction['marcador_mas_probable']} (con una probabilidad de {prediction['prob_marcador']}%)")