from contextlib import asynccontextmanager
import os, sys, io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.logger import logger
from src.exception import ApplicationException
from src.constant.training_config import PRODUCTION_MODEL_FILE_PATH, SCHEMA_FILE_PATH
from src.utils.utils import load_object, read_yaml_file
import src.fastapi_schema as fs
import pandas as pd
from fastapi.responses import StreamingResponse

# Global model variable
_model = None
_schema_config = read_yaml_file(SCHEMA_FILE_PATH)
schema_columns = _schema_config.get("columns", [])
features = [list(item.keys())[0] for item in schema_columns if "Result" not in item]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.

    Startup:
        - Loads the production ML model into memory

    Shutdown:
        - Releases model resources
    """
    global _model

    try:
        if not os.path.exists(PRODUCTION_MODEL_FILE_PATH):
            logger.warning(
                f"Model not found at {PRODUCTION_MODEL_FILE_PATH}. Run main.py to train the model."
            )
        else:
            _model = load_object(PRODUCTION_MODEL_FILE_PATH)
            logger.info(f"PhishGuard model loaded from: {PRODUCTION_MODEL_FILE_PATH} ")

        yield

    except Exception as e:
        logger.error(f"Error during application lifespan: {e}")
        raise ApplicationException(e, sys)

    finally:
        _model = None
        logger.info("Model resources released")


def _require_model():
    if _model is None:
        raise HTTPException(status_code=404, detail="Model Not Found")


def _predict_one(features: fs.URLFeatures) -> fs.PredictionResponse:
    df = pd.DataFrame([features.model_dump()])
    pred = int(_model.predict(df)[0])
    label = "Legitimate" if pred == 1 else "Phishing"
    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(df)[0]
        confidence = round(float(max(proba)), 4)
    return fs.PredictionResponse(prediction=pred, label=label, confidence=confidence)


app = FastAPI(
    title="PhishGuard API",
    description=(
        "## Phishing Website Detection API\n\n"
        "PhishGuard uses a machine learning model trained on "
        "30 URL/page-level features to classify websites as:\n\n"
        "- **1 - Legitimate Website**\n"
        "- **0 - Phishing Website**\n\n"
        "### Available Endpoints\n"
        "- `GET /health` - Service health check\n"
        "- `GET /features` - List required feature names\n"
        "- `POST /predict` - Single prediction\n"
        "- `POST /batch` - Batch prediction from JSON\n"
        "- `POST /predict-csv` - Batch prediction using CSV upload\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health", tags=["Health"], summary="Liveness check", operation_id="healthCheck"
)
async def health_check():
    """
    Returns API health status and model availability.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": "1.0.0",
    }


@app.get(
    "/features",
    tags=["Info"],
    summary="List all required feature names",
    operation_id="getFeatures",
)
def get_features():
    """Returns the 30 feature names the model expects."""
    return {"features": features, "count": len(features)}


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict a single URL",
    operation_id="predictSingleURL",
    response_model=fs.PredictionResponse,
)
def predict(features: fs.URLFeatures):
    """
    Classify a single URL as Legitimate = 1 or Phishing = 0
    """
    _require_model()
    try:
        return _predict_one(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{ApplicationException(e, sys)}")


@app.post(
    "/batch",
    tags=["Prediction"],
    summary="Predict a multiple URL",
    operation_id="predictBatchURL",
    response_model=fs.BatchResponse,
)
def predict_batch(request: fs.BatchRequest):
    """
    Classify multiple URLs in a single call.
    """
    _require_model()
    try:
        results = [_predict_one(r) for r in request.records]
        phishing = sum(1 for r in results if r.prediction == 0)
        return fs.BatchResponse(
            total=len(results),
            phishing_count=phishing,
            legitimate_count=len(results) - phishing,
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{ApplicationException(e, sys)}")


@app.post(
    "/predict-csv",
    tags=["Prediction"],
    summary="Predict a multiple URL through CSV",
    operation_id="predictBatchURLCSV",
    response_class=StreamingResponse,
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV with 30 feature columns")
):
    """
    Upload a CSV file with 30 feature columns.
    """
    _require_model()
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = [col for col in features if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=500, detail=f"Missing feature columns:: {missing}"
            )

        predict = _model.predict(df)
        df["prediction"] = predict.astype(int)
        df["label"] = df["prediction"].map({1: "Legitimate", 0: "Phishing"})

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=phishguard_predictions.csv"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{ApplicationException(e, sys)}")
