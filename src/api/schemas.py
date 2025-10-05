# src/api/schemas.py

from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    """El cuerpo de la solicitud para una predicción."""
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    """La respuesta que nuestra API enviará."""
    winner: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    poisson_most_likely_score: str