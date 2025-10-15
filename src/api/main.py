# src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .schemas import PredictionRequest, PredictionResponse
from .predictor import Predictor

app = FastAPI(
    title="Liga MX Prediction API",
    description="API para predecir resultados de partidos de la Liga MX usando un ensamble de modelos.",
    version="1.0.0"
)


origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://*.vercel.app",
    "https://liga-mx-predicto.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = Predictor()

@app.get("/teams", response_model=List[str])
def get_teams():
    """Devuelve una lista de todos los equipos disponibles para las predicciones."""
    # Los equipos están en el mapeo.
    return sorted(list(predictor.team_mapping.keys()))

@app.post("/predict", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    """Recibe los equipos y devuelve la predicción del modelo."""
    try:
        prediction = predictor.predict_single_match(request.home_team, request.away_team)
        return prediction
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Para cualquier error
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")