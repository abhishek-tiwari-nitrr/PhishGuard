from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Stores the output artifacts generated during the data ingestion phase of the machine learning pipeline.

    Attributes:
        - raw_file_path (str): Path to the raw dataset collected from the source.
        - train_file_path (str): Path to the training dataset created from the raw data based on the DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO defined in the training_config file.
        - test_file_path (str): Path to the testing dataset created from the raw data based on the DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO defined in the training_config file.

    """

    raw_file_path: str
    train_file_path: str
    test_file_path: str
