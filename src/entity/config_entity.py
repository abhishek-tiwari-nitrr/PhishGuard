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
        - data_ingestion_dir (str) = Directory for storing data ingestion artifacts.
        - data_ingestion_ingest_dir (str): Directory for storing data ingestion - ingested artifacts.
        - data_ingestion_raw_dir (str): Directory for storing data ingestion - raw artifacts.
        - training_file_path (str) = Path to the training dataset file.
        - testing_file_path (str) = Path to the testing dataset file.
        - train_test_split_ratio (float) = Ratio used to split the dataset into training and testing sets.
        - raw_file_path (str) =  Path to the raw dataset file.
        - database_name (str) =  MongoDB database name.
        - collection_name (str) = MongoDB collection name.
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


class DataValidationConfig:
    """
    Configuration class for the data validation component.

    Attributes:
        - data_validation_dir (str): Directory for storing data validation artifacts.
        - validate_data_dir (str): Directory for storing valid dataset artifacts.
        - invalidate_data_dir (str): Directory for storing invalid dataset artifacts.
        - drift_report_dir (str): Directory for storing Drift Report.
        - validate_train_file_path (str): Path to the validate training dataset file.
        - validate_test_file_path (str): Path to the validate testing dataset file.
        - invalidate_train_file_path (str): Path to the invalidate training dataset file.
        - invalidate_test_file_path (str): Path to the invalidate testing dataset file.
        - drift_report_file_path (str): Path to the data drift report file.

    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_config.DATA_VALIDATION_DIR_NAME,
        )
        self.validate_data_dir = os.path.join(
            self.data_validation_dir, training_config.DATA_VALIDATION_VALID_DIR_NAME
        )
        self.invalidate_data_dir = os.path.join(
            self.data_validation_dir, training_config.DATA_VALIDATION_INVALID_DIR_NAME
        )
        self.drift_report_dir = os.path.join(
            self.data_validation_dir,
            training_config.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
        )
        self.validate_train_file_path = os.path.join(
            self.validate_data_dir, training_config.TRAIN_FILE_NAME
        )
        self.validate_test_file_path = os.path.join(
            self.validate_data_dir, training_config.TEST_FILE_NAME
        )
        self.invalidate_train_file_path = os.path.join(
            self.invalidate_data_dir, training_config.TRAIN_FILE_NAME
        )
        self.invalidate_test_file_path = os.path.join(
            self.invalidate_data_dir, training_config.TEST_FILE_NAME
        )
        self.drift_report_file_path = os.path.join(
            self.drift_report_dir, training_config.DRIFT_REPORT_NAME
        )


class DataTransformationConfig:
    """
    Configuration class for the data transformation component.

    Attributes:
        - data_transformation_dir (str): Directory for storing data transformation artifacts.
        - transformed_data_dir (str): Directory for storing transformed dataset artifacts.
        - transformed_train_file_path (str): Path to the transformed training dataset file.
        - transformed_test_file_path (str): Path to the transformed testing dataset file.
        - transformed_object_dir (str): Directory for storing transformed object artifact.
        - transformed_object_file_path (str): Path to the Preprocessing Object File.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_config.DATA_VALIDATION_DIR_NAME,
        )
        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            training_config.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )
        self.transformed_train_file_path = os.path.join(
            self.transformed_data_dir,
            training_config.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path = os.path.join(
            self.transformed_data_dir,
            training_config.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_dir = os.path.join(
            self.data_transformation_dir,
            training_config.DATA_TRANSFORMATION_PREPROCESSING_OBJECT_DIR,
        )
        self.transformed_object_file_path = os.path.join(
            self.transformed_object_dir, training_config.PREPROCESSING_OBJECT_FILE_NAME
        )
