from dotenv import load_dotenv
import os

load_dotenv()

# General Config
PIPELINE_NAME: str = "PG_Pipeline"
ARTIFACT_DIR: str = "PG_Artifacts"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
RAW_FILE_NAME: str = "raw.csv"

# MongoDB Config
MONGO_DB_DATABASE_NAME: str = os.getenv("DATABASE_NAME")
MONGO_DB_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME")

# Data Ingestion Config
DATA_INGESTION_DIR_NAME: str = "Data_Ingestion"
DATA_INGESTION_INGESTED_DIR: str = "Ingested"
DATA_INGESTION_RAW_DIR: str = "Raw"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

