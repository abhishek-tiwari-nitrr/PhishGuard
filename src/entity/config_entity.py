from datetime import datetime, timezone, timedelta
from src.constant import training_config
import os

IST = timezone(timedelta(hours=5, minutes=30))


class TrainingPipelineConfig:
    """
    Configuration class for the overall training pipeline.

    Attributes:
        - pipeline_name (str): Name of the training pipeline
        - artifact_name (str): Name of the root artifact directory
        - artifact_dir (str): Full path to the timestamped artifact directory
    """

    def __init__(self, timestamp=datetime.now(IST)):
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name = training_config.PIPELINE_NAME
        self.artifact_name = training_config.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)


class DataIngestionConfig:
    """
    Configuration class for the data ingestion component.

    Attributes:
        - data_ingestion_dir (str) = Directory for storing data ingestion artifacts
        - data_ingestion_ingest_dir (str): Directory for storing data ingestion - ingested artifacts
        - data_ingestion_raw_dir (str): Directory for storing data ingestion - raw artifacts
        - training_file_path (str) = Path to the training dataset file
        - testing_file_path (str) = Path to the testing dataset file
        - train_test_split_ratio (float) = Ratio used to split the dataset into training and testing sets
        - raw_file_path (str) =  Path to the raw dataset file
        - database_name (str) =  MongoDB database name
        - collection_name (str) = MongoDB collection name
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_config.DATA_INGESTION_DIR_NAME,
        )
        self.data_ingestion_ingest_dir = os.path.join(
            self.data_ingestion_dir,
            training_config.DATA_INGESTION_INGESTED_DIR,
        )
        self.data_ingestion_raw_dir = os.path.join(
            self.data_ingestion_dir,
            training_config.DATA_INGESTION_RAW_DIR,
        )
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_config.DATA_INGESTION_INGESTED_DIR,
            training_config.TRAIN_FILE_NAME,
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_config.DATA_INGESTION_INGESTED_DIR,
            training_config.TEST_FILE_NAME,
        )
        self.train_test_split_ratio = (
            training_config.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.raw_file_path = os.path.join(
            self.data_ingestion_dir,
            training_config.DATA_INGESTION_RAW_DIR,
            training_config.RAW_FILE_NAME,
        )
        self.database_name = training_config.MONGO_DB_DATABASE_NAME
        self.collection_name = training_config.MONGO_DB_COLLECTION_NAME
