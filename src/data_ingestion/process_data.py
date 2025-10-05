# src/data_ingestion/process_data.py

import pandas as pd
from db_config import get_engine

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza y transformación de los datos.
    """
    print("Iniciando limpieza y preparación de datos...")

    # 1. Renombrar columnas a un formato estándar (snake_case)
    column_mapping = {
        'Country': 'country', 'League': 'league', 'Season': 'season',
        'Date': 'date', 'Time': 'time', 'Home': 'home_team', 'Away': 'away_team',
        'HG': 'home_goals', 'AG': 'away_goals', 'Res': 'result', 'PSCH': 'psc_home',
        'PSCD': 'psc_draw', 'PSCA': 'psc_away', 'MaxCH': 'max_ch', 'MaxCD': 'max_cd'
    }
    df = df.rename(columns=column_mapping)

    # 2. Convertir la columna 'date' a formato de fecha
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # 3. Eliminar filas donde la fecha no se pudo convertir
    df.dropna(subset=['date'], inplace=True)

    # 4. Asegurar que los goles sean enteros (manejando nulos si los hay)
    goal_cols = ['home_goals', 'away_goals']
    for col in goal_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
    # 5. Seleccionar solo las columnas que corresponden a nuestra tabla
    final_columns = list(column_mapping.values())
    df = df[final_columns]
    
    print(f"Limpieza completa. Se procesarán {len(df)} registros.")
    return df

def ingest_data_from_csv(file_path: str):
    """
    Proceso completo: leer CSV, limpiar datos e insertarlos en la BD.
    """
    engine = get_engine()
    if engine is None:
        print("No se pudo obtener el motor de la base de datos. Abortando ingesta.")
        return

    try:
        print(f"Leyendo datos desde {file_path}...")
        df = pd.read_csv(file_path)

        # Limpiar y preparar los datos
        df_cleaned = clean_and_prepare_data(df)

        # Ingesta en la base de datos
        print("Iniciando la inserción de datos en la base de datos...")
        # 'replace' borra la tabla y la vuelve a crear.
        # 'append' añadiría los nuevos datos. Usamos 'replace' para la carga inicial.
        df_cleaned.to_sql('partidos', engine, if_exists='replace', index=False)
        
        print("¡Proceso de ingesta de datos completado exitosamente!")

    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta: {file_path}")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la ingesta: {e}")

if __name__ == '__main__':
    # Asegúrate de que tu CSV esté en la carpeta /data
    csv_file = 'data/historical_data.csv' 
    ingest_data_from_csv(csv_file)