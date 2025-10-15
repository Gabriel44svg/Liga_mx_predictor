# api/main.py - Simplified API for Vercel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import os

app = FastAPI(
    title="Liga MX Prediction API",
    description="API para predecir resultados de partidos de la Liga MX",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample teams data (you can replace this with your actual data)
TEAMS = [
    "América", "Atlas", "Atlético San Luis", "Cruz Azul", "Guadalajara",
    "Juárez", "León", "Mazatlán", "Monterrey", "Necaxa", "Pachuca",
    "Puebla", "Querétaro", "Santos Laguna", "Tigres", "Tijuana",
    "Toluca", "UANL"
]

@app.get("/")
def read_root():
    return {"message": "Liga MX Predictor API", "status": "running"}

@app.get("/api/teams", response_model=List[str])
def get_teams():
    """Devuelve una lista de todos los equipos disponibles para las predicciones."""
    return sorted(TEAMS)

@app.post("/api/predict")
def predict_match(request: dict):
    """Recibe los equipos y devuelve una predicción simulada."""
    try:
        home_team = request.get("home_team", "")
        away_team = request.get("away_team", "")
        
        if not home_team or not away_team:
            raise HTTPException(status_code=400, detail="home_team y away_team son requeridos")
        
        if home_team not in TEAMS or away_team not in TEAMS:
            raise HTTPException(status_code=400, detail="Equipo no válido")
        
        if home_team == away_team:
            raise HTTPException(status_code=400, detail="El equipo local y visitante no pueden ser el mismo")
        
        # Simulated prediction (replace with your actual model)
        import random
        home_prob = round(random.uniform(30, 50), 1)
        draw_prob = round(random.uniform(20, 35), 1)
        away_prob = round(100 - home_prob - draw_prob, 1)
        
        winner = "H" if home_prob > away_prob else "A" if away_prob > home_prob else "D"
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_prob,
            "winner": winner,
            "poisson_most_likely_score": f"{random.randint(1,3)}-{random.randint(0,2)}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# This is the entry point for Vercel
handler = app
