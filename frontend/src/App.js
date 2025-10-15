// frontend/src/App.js

import React, { useState, useEffect } from 'react';
import './App.css'; 

function App() {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? '/api/teams' 
      : 'http://localhost:8000/teams';
    
    fetch(apiUrl)
      .then(response => response.json())
      .then(data => {
        setTeams(data);
        if (data.length > 1) {
          setHomeTeam(data[0]);
          setAwayTeam(data[1]);
        }
      })
      .catch(err => setError('No se pudo conectar con el servidor de predicciones.'));
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (homeTeam === awayTeam) {
      setError('El equipo local y visitante no pueden ser el mismo.');
      setPrediction(null);
      return;
    }
    setIsLoading(true);
    setError('');
    setPrediction(null);

    await new Promise(res => setTimeout(res, 500)); 

    try {
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/predict' 
        : 'http://localhost:8000/predict';
        
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Ocurrió un error en la predicción.');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1> Liga MX </h1>
        <p>Selecciona los equipos y descubre el resultado más probable.</p>
      </header>
      <main>
        <form className="prediction-form" onSubmit={handlePredict}>
          <div className="team-selector">
            <label htmlFor="home-team">Equipo Local</label>
            <select
              id="home-team"
              value={homeTeam}
              onChange={(e) => setHomeTeam(e.target.value)}
            >
              {teams.map(team => <option key={team + "-h"} value={team}>{team}</option>)}
            </select>
          </div>
          <div className="vs">VS</div>
          <div className="team-selector">
            <label htmlFor="away-team">Equipo Visitante</label>
            <select
              id="away-team"
              value={awayTeam}
              onChange={(e) => setAwayTeam(e.target.value)}
            >
              {teams.map(team => <option key={team + "-a"} value={team}>{team}</option>)}
            </select>
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analizando...' : 'Predecir '}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {prediction && (
          <div className="prediction-result">
            <h2>Resultado de la Predicción</h2>
            <div className="winner">
              <span>Resultado más probable:</span>
              <strong>
                {prediction.winner === 'H' ? `Victoria ${homeTeam}` :
                 prediction.winner === 'A' ? `Victoria ${awayTeam}` : 'Empate'}
              </strong>
            </div>
            <div className="probabilities">
              <div className="prob-bar" style={{'--prob': `${prediction.home_win_prob}%`, '--color': '#00ff7f'}}>
                <span>{homeTeam}: {prediction.home_win_prob}%</span>
              </div>
              <div className="prob-bar" style={{'--prob': `${prediction.draw_prob}%`, '--color': '#ffc107'}}>
                <span>Empate: {prediction.draw_prob}%</span>
              </div>
              <div className="prob-bar" style={{'--prob': `${prediction.away_win_prob}%`, '--color': '#f44336'}}>
                <span>{awayTeam}: {prediction.away_win_prob}%</span>
              </div>
            </div>
              <p className="poisson-score">Marcador más probable (Poisson): <strong>{prediction.poisson_most_likely_score}</strong></p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;