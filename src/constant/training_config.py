from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

# General Config
PIPELINE_NAME: str = "PG_Pipeline"
ARTIFACT_DIR: str = "PG_Artifacts"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
RAW_FILE_NAME: str = "raw.csv"
DRIFT_REPORT_NAME: str = "report.yaml"
SCHEMA_FILE_PATH: str = os.path.join("data_config", "schema.yaml")
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
TARGET_COLUMN: str = "Result"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
RANDOM_STATE: int = 42

# MongoDB Config
MONGO_DB_DATABASE_NAME: str = os.getenv("DATABASE_NAME")
MONGO_DB_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")

# Data Ingestion Config
DATA_INGESTION_DIR_NAME: str = "1. Data Ingestion"
DATA_INGESTION_INGESTED_DIR: str = "Ingested"
DATA_INGESTION_RAW_DIR: str = "Raw"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation Config
DATA_VALIDATION_DIR_NAME: str = "2. Data Validation"
DATA_VALIDATION_VALID_DIR_NAME: str = "Valid Dataset"
DATA_VALIDATION_INVALID_DIR_NAME: str = "Invalid Dataset"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "Drift Report"

# Data Transformation Config
DATA_TRANSFORMATION_DIR_NAME: str = "3. Data Transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "Transformed"
DATA_TRANSFORMATION_PREPROCESSING_OBJECT_DIR: str = "Preprocessing Object"
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}


# Model Trainer Config
MODEL_TRAINER_DIR_NAME: str = "4. Model Trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "Trained Model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05


# Model Evaluation Config
MODEL_EVALUATION_DIR_NAME: str = "5. Model Evaluation"
# new model must improve by >= 2 %
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# Model Pusher Config
PRODUCTION_MODEL_DIR: str = "production_model"
PRODUCTION_MODEL_FILE_NAME: str = "model.pkl"

# Production Model Config
PRODUCTION_MODEL_FILE_PATH: str = os.path.join(
    PRODUCTION_MODEL_DIR, PRODUCTION_MODEL_FILE_NAME
)
