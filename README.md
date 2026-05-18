# PhishGuard
 
> **AI-powered phishing website detection** - 30 URL/page-level features - scikit-learn - MLflow - FastAPI - Streamlit
 
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-2.x-red.svg)](https://streamlit.io/)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen)](https://abhishek-tiwari-nitrr-phishguard.streamlit.app/)

---

## Architecture

See detailed architecture diagrams here:

- [Architecture Documentation](docs/ARCHITECTURE.md)

---

## Features
 
- **Full MLOps pipeline** - 5 sequential stages, each producing typed artifacts
- **Multi-model training** - Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost with GridSearchCV
- **Experiment tracking** - MLflow with optional DagsHub remote; falls back to local `mlruns/`
- **Data drift detection** - Kolmogorov–Smirnov test per feature, saved as `report.yaml`
- **Production gating** - new model must improve F1 by ≥ 2 % to replace the incumbent
- **REST API** - FastAPI with Swagger UI, single/batch/CSV prediction endpoints
- **Streamlit dashboard** - four tabs: single prediction, batch upload, API explorer, pipeline trigger
- **Rotating logs** - 5 MB per file, 3 backup files, IST timestamps

---

## Project Structure

```
PhishGuard/
├── app.py                          # FastAPI application
├── main.py                         # Streamlit entry point
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .python-version                 # 3.11
│
├── data config/
│   └── schema.yaml                 # Column definitions & numerical column list
├── src/
│   ├── components/                 # Pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── constant/
│   │   ├── config.py               # Logging & DagsHub config
│   │   ├── training_config.py      # All pipeline constants
│   │   └── streamlit_config.py
│   ├── entity/
│   │   ├── config_entity.py        # Dataclass configs per stage
│   │   └── artifact_entity.py      # Dataclass artifacts per stage
│   ├── utils/
│   │   ├── utils.py                # IO, metrics, GridSearch helpers
│   │   └── model_utils.py          # PhishGuard wrapper (predict / predict_proba)
│   ├── pages/                      # Streamlit page modules
│   │   ├── single_prediction.py
│   │   ├── batch_prediction.py
│   │   ├── api_swagger.py
│   │   └── training_pipeline.py
│   ├── fastapi_schema.py          # Pydantic request/response models
│   ├── exception.py               # ApplicationException with file:line context
│   └── logger.py                  # Rotating file logger
│
├── production_model/
│   └── model.pkl                  # Promoted production model
├── PG_Artifacts/                  # Timestamped run artifacts (auto-generated)
│   └── DD_MM_YYYY_HH_MM_SS/
│       ├── 1. Data Ingestion/
│       ├── 2. Data Validation/
│       ├── 3. Data Transformation/
│       ├── 4. Model Trainer/
│       └── 5. Model Evaluation/
│
└── docs/
    ├── 1. High Level Design/
    ├── 2. Low Level Design/
    └── ARCHITECTURE.md
```

---

## Quick Start

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- MongoDB Atlas cluster with your phishing dataset loaded


### 1. Clone & install

```bash
git clone https://github.com/abhishek-tiwari-nitrr/PhishGuard
cd PhishGuard

# Create virtual environment
uv venv

# Activate environment
# Windows
.venv\Scripts\activate

# Install dependencies from pyproject.toml
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root and add your configuration values.

```dotenv
# MongoDB
MONGO_DB_URL = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<appName>"
DATABASE_NAME = "<database name>"
COLLECTION_NAME = "<collection name>"

# DagsHub
DAGSHUB_TRACKING_URI = "https://dagshub.com/<username>/<repo_name>.mlflow"
DAGSHUB_TOKEN = "<token>"
DAGSHUB_USERNAME = "<username>"
DAGSHUB_REPO_NAME = "<repo_name>"
LOCAL_TRACKING_URI = "mlruns"

# Render FastApi
API_URL = "<url_of_fast_api>"
```

### 3. Run the training pipeline

Trigger it from the **🚀 Train Pipeline** tab in the Streamlit UI.

### 4. Start the FastAPI server
 
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

### 5. Start the Streamlit dashboard
 
```bash
streamlit run main.py
# Opens: http://localhost:8501
```

---

## API Reference
 
Base URL: `http://localhost:8000`
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + model status |
| `GET` | `/features` | List 30 required feature names |
| `POST` | `/predict` | Single URL prediction (JSON) |
| `POST` | `/batch` | Batch prediction (JSON array) |
| `POST` | `/predict-csv` | Batch prediction (CSV upload & CSV download) |
 
Full interactive docs at `/docs` (Swagger UI)

---

## Streamlit UI
 
| Tab | Purpose |
|-----|---------|
| 🔍 Single Prediction | Fill 30 feature fields and get an instant verdict |
| 📊 Batch Prediction | Upload a CSV and download predictions |
| ⚡ API Explorer | Embedded Swagger iframe for interactive API testing |
| 🚀 Train Pipeline | Trigger the full training pipeline and watch stage logs |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML & Data | scikit-learn, pandas, numpy, scipy |
| Experiment Tracking | MLflow, DagsHub |
| API | FastAPI, Pydantic, Uvicorn |
| UI | Streamlit |
| Database | MongoDB Atlas |
| Language | Python 3.11 |
