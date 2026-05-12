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


@dataclass
class DataValidationArtifact:
    """
    Stores the output artifacts generated during the data validation phase of the machine learning pipeline.

    Attributes:
        - validation_status (bool): Validation Passed/Failed.
        - valid_train_file_path (str): Path to validated train dataset.
        - valid_test_file_path (str): Path to validated test dataset.
        - invalid_train_file_path (str): Path where invalid/corrupted train dataset is stored.
        - invalid_test_file_path (str): Path where invalid/corrupted test dataset is stored.
        - drift_report_file_path (str): Path to generated data drift report.

    """

    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Stores the output artifacts generated during the data transformation phase of the machine learning pipeline.

    Attributes:
        - transformed_train_file_path (str): Path to transformed train data.
        - transformed_test_file_path (str): Path to transformed test data.
        - transformed_object_file_path (str): Path to preprocessing object.

    """

    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str


@dataclass
class ClassificationModelArtifact:
    """
    Stores evaluation metrics for a classification model.

    Attributes:
        - f1_score (float): Harmonic mean of precision and recall, providing a balance between the two metrics.
        - precision_score (float): Ratio of correctly predicted positive observations to the total predicted positives.
        - recall_score (float):  Ratio of correctly predicted positive observations to all actual positive observations.

    """

    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Stores artifacts generated during the model training phase.

    Attributes:
        - trained_model_file_path (str): File path where the trained model is saved.
        - train_metric_artifact (ClassificationModelArtifact): Evaluation metrics computed on the training dataset.
        - test_metric_artifact (ClassificationModelArtifact): Evaluation metrics computed on the testing dataset.

    """

    trained_model_file_path: str
    train_metric_artifact: ClassificationModelArtifact
    test_metric_artifact: ClassificationModelArtifact
