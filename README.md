# ⚽ Sistema de Predicción para la Liga MX ⚽

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white)

Este proyecto es un sistema de predicción de resultados de fútbol para la Liga MX. Utiliza una arquitectura de modelado mixto (ensamble) que combina las fortalezas de modelos estadísticos, de Machine Learning clásico y de Deep Learning para generar predicciones precisas sobre el resultado de un partido (victoria local, empate o victoria visitante).

El proyecto es un pipeline de predicción de 4 capas, donde cada capa aporta una perspectiva diferente, culminando en un meta-modelo que consolida la "inteligencia colectiva" del sistema.

1.  **Capa 1: Modelo Estadístico (Poisson)**
    * Modela la cantidad de goles esperados basándose en promedios históricos de ataque y defensa de cada equipo.

2.  **Capa 2: Modelos de Machine Learning (ML)**
    * Utiliza características de ingeniería de datos (feature engineering), como la "forma reciente" de los equipos, para entrenar tres algoritmos distintos: **Regresión Logística**, **Random Forest** y **XGBoost**.

3.  **Capa 3: Modelo de Deep Learning (DL)**
    * Implementa una Red Neuronal Recurrente (LSTM) que aprende la "identidad" de cada equipo a través de *Embeddings* y captura patrones secuenciales en su racha de resultados.

4.  **Capa 4: Meta-Modelo (Ensamble)**
    * Un modelo final de Regresión Logística que toma como entrada las probabilidades generadas por las tres capas anteriores y aprende a combinarlas para emitir la predicción definitiva del sistema.

5.  **API y Frontend**
    * Toda la lógica de predicción se expone a través de una API RESTful construida con **FastAPI**, que a su vez es consumida por una interfaz de usuario interactiva desarrollada en **React**.


---------------------------------------------------------------------------------------------------------------------------------
| Componente               | Tecnología/Librería                                                                                 |
| -------------------------| --------------------------------------------------------------------------------------------------- |
| **Backend**              | Python 3.11, FastAPI, Uvicorn                                                                       |
| **Modelado**             | Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow (Keras), SciPy, Joblib                             |
| **Base de Datos**        | PostgreSQL                                                                                          |
| **Frontend**             | React.js, JavaScript (ES6+), HTML5, CSS3                                                            |
| **Control de Versiones** | Git                                                                                                 |  
---------------------------------------------------------------------------------------------------------------------------------


* **Modelo de Ensamble Multi-capa:** Combina 4 modelos base para una mayor robustez y precisión.
* **Pipelines de Entrenamiento Modulares:** Scripts separados para el entrenamiento de cada capa del modelo.
* **Persistencia de Modelos:** Todos los modelos entrenados se guardan en la carpeta `/artifacts` para su reutilización.
* **API RESTful:** Endpoints para obtener la lista de equipos y realizar predicciones en tiempo real.
* **Interfaz de Usuario Interactiva:** Permite a cualquier usuario hacer predicciones fácilmente desde el navegador.



Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina.

# Prerrequisitos
* Git
* Python 3.10+
* Node.js y npm
* Un servidor de PostgreSQL funcionando

# 1. Configuración del Backend y Base de Datos

```bash
# 1. Clona el repositorio
git clone [https://github.com/Gabriel44svg/Liga_mx_predictor.git](https://github.com/Gabriel44svg/Liga_mx_predictor.git)
cd Liga_mx_predictor

# 2. Crea y activa un entorno virtual de Python
python -m venv venv
venv\Scripts\activate

# 3. Instala todas las dependencias de Python
pip install -r requirements.txt

# 4. Configura la base de datos
#    - Crea una base de datos en PostgreSQL (ej. 'liga_mx_db').
#llenalo con tus credenciales el .env
DB_USER=
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=
DB_NAME=

# 5. Crea la estructura de la base de datos y carga los datos iniciales
python src/data_ingestion/db_setup.py
python src/data_ingestion/process_data.py

#Ahora el entrenamiento de los modelos estos llegan a artifacts

# 1. Entrena los modelos de Machine Learning Clásico
python -m src.models.ml_pipeline

# 2. Entrena el modelo de Deep Learning
python -m src.models.dl_pipeline

# 3. Entrena el Meta-Modelo
python -m src.models.meta_model

#Frontend

# 1. Navega a la carpeta del frontend
cd frontend

# 2. Instala las dependencias de Node.js
npm install