# src/data_ingestion/db_setup.py

from sqlalchemy import text
from db_config import get_engine

# Definimos la estructura de la tabla usando SQL
# Usamos nombres de columna en minúsculas y sin espacios por convención
CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS partidos (
    id SERIAL PRIMARY KEY,
    country VARCHAR(50),
    league VARCHAR(50),
    season VARCHAR(10),
    date DATE,
    time TIME,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    home_goals INTEGER,
    away_goals INTEGER,
    result CHAR(1),
    psc_home REAL,
    psc_draw REAL,
    psc_away REAL,
    max_ch REAL,
    max_cd REAL,
    CONSTRAINT unique_match UNIQUE (home_team, away_team, date)
);
"""

def setup_database():
    """Ejecuta el script para crear la tabla de partidos."""
    engine = get_engine()
    if engine is None:
        print("No se pudo obtener el motor de la base de datos. Abortando.")
        return

    try:
        with engine.connect() as connection:
            connection.execute(text("BEGIN;")) # Inicia una transacción
            print("Creando la tabla 'partidos' si no existe...")
            connection.execute(text(CREATE_TABLE_QUERY))
            connection.execute(text("COMMIT;")) # Confirma la transacción
            print("Tabla 'partidos' verificada/creada exitosamente.")
    except Exception as e:
        print(f"Error durante la configuración de la base de datos: {e}")

if __name__ == '__main__':
    setup_database()