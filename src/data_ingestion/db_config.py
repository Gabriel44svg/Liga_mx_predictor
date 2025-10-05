# src/data_ingestion/db_config.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener las credenciales de la base de datos
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Crear la URL de conexión para SQLAlchemy
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_engine():
    """Crea y devuelve un motor de SQLAlchemy."""
    try:
        engine = create_engine(DATABASE_URL)
        # Probar la conexión
        with engine.connect() as connection:
            print("Conexión a la base de datos establecida exitosamente.")
        return engine
    except Exception as e:
        print(f"Error al conectar con la base de datos: {e}")
        return None

if __name__ == '__main__':
    # Esto permite probar la conexión ejecutando el archivo directamente
    get_engine()