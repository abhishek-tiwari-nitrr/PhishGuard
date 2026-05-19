from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.constant.training_config import SCHEMA_FILE_PATH
from src.utils.utils import read_yaml_file, read_csv_file_data, write_yaml_file
from src.exception import ApplicationException
from src.logger import logger
import sys, os
import pandas as pd
from scipy.stats import ks_2samp


class DataValidation:
    """
    Handles validation of ingested datasets against the schema and performs data drift detection.

    Responsibilities:
        - Validate column count
        - Validate required numerical columns
        - Validate column names
        - Detect data drift using statistical testing
        - Save validated and invalidated datasets
        - Generate drift reports

    Attributes:
        - data_ingestion_artifacts (DataIngestionArtifact): Artifact containing paths of train and test datasets.
        - data_validation_config (DataValidationConfig): Configuration object containing validation paths and settings.
        -  _schema_config (dict): Schema configuration loaded from YAML file.
    """

    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initialize the DataValidation class.

        Args:
            data_ingestion_artifacts (DataIngestionArtifact): Artifact containing training and testing file paths.
            data_validation_config (DataValidationConfig): Configuration object containing validation directories, drift report paths, and validation settings.

        Raises:
            ApplicationException: If initialization fails.
        """
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ApplicationException(e, sys)

    def validate_no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate whether the DataFrame contains the expected number of columns defined in the schema.

        Args:
            dataframe (pd.DataFrame): Input DataFrame to validate.

        Raises:
            ApplicationException: If validation fails.

        Returns:
            bool: True if column count matches schema, otherwise False.
        """
        try:
            schema_columns = self._schema_config.get("columns", [])
            required_count = len(schema_columns)
            actual_count = len(dataframe.columns)
            logger.info(
                f"Schema columns: {required_count} | DataFrame columns: {actual_count}"
            )
            return actual_count == required_count
        except Exception as e:
            raise ApplicationException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate whether all required numerical columns exist in the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame to validate.

        Raises:
            ApplicationException: If validation fails.

        Returns:
            bool: True if all required numeric column exist, otherwise False.
        """
        try:
            numerical_columns = self._schema_config.get("numerical_columns", [])
            missing = [col for col in numerical_columns if col not in dataframe.columns]
            if missing:
                logger.info(f"Missing numerical columns: {missing}")
                return False
            return True
        except Exception as e:
            raise ApplicationException(e, sys)

    def validate_column_name(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate whether all schema-defined column names exist in the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame to validate.


        Raises:
            ApplicationException: If validation fails.

        Returns:
            bool: True if all required columns exist, otherwise False.
        """
        try:
            schema_columns = {
                list(col.keys())[0] for col in self._schema_config.get("columns", [])
            }
            df_columns = set(dataframe.columns)
            missing = schema_columns - df_columns
            if missing:
                logger.info(f"Missing columns: {missing}")
                return False
            return True
        except Exception as e:
            raise ApplicationException(e, sys)

    def detect_data_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Detect data drift between base and current datasets using the Kolmogorov-Smirnov (KS) statistical test.

        Args:
            base_df (pd.DataFrame): Reference dataset.
            current_df (pd.DataFrame): Current dataset to compare against reference dataset.
            threshold (float, optional): P-value threshold for drift detection. Defaults to 0.05.

        Raises:
            ApplicationException: If drift detection fails.

        Returns:
            bool: True if no drift is detected, otherwise False.
        """
        try:
            status = True
            report = {}
            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)
            for col in base_df.columns:
                d1, d2 = base_df[col], current_df[col]
                ks_result = ks_2samp(d1, d2)
                drift_found = float(ks_result.pvalue) < threshold
                if drift_found:
                    status = False
                report[col] = {
                    "p_value": float(ks_result.pvalue),
                    "drift_status": drift_found,
                }
                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=report,
                )
                logger.info(
                    f"Drift Report Written to {self.data_validation_config.drift_report_file_path} and Drift detected: {not status} "
                )
        except Exception as e:
            raise ApplicationException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Execute the complete data validation pipeline.

        Steps:
            1. Load training and testing datasets
            2. Validate column count
            3. Validate numerical columns
            4. Validate column names
            5. Save valid or invalid datasets
            6. Detect data drift
            7. Generate validation artifact

        Raises:
            ValueError: If validation fails by count, required numeric columns and required column names.
            ApplicationException: If validation pipeline execution fails.

        Returns:
            DataValidationArtifact:
                Artifact containing:
                    - Validation status
                    - Valid dataset paths
                    - Invalid dataset paths
                    - Drift report path

        """
        try:
            training_file_path = self.data_ingestion_artifacts.train_file_path
            testing_file_path = self.data_ingestion_artifacts.test_file_path

            train_df = read_csv_file_data(training_file_path)
            test_df = read_csv_file_data(testing_file_path)

            error_message = ""

            if not self.validate_no_of_columns(train_df):
                error_message += "Train dataframe: column count mismatch.\n"
            if not self.validate_no_of_columns(test_df):
                error_message += "Test dataframe: column count mismatch.\n"

            if not self.is_numerical_column_exist(train_df):
                error_message += "Train dataframe: numerical columns missing.\n"
            if not self.is_numerical_column_exist(test_df):
                error_message += "Test dataframe: numerical columns missing.\n"

            if not self.validate_column_name(train_df):
                error_message += "Train dataframe: column name mismatch.\n"
            if not self.validate_column_name(test_df):
                error_message += "Test dataframe: column name mismatch.\n"

            if error_message:
                os.makedirs(
                    self.data_validation_config.invalidate_data_dir, exist_ok=True
                )
                train_df.to_csv(
                    self.data_validation_config.invalidate_train_file_path,
                    index=False,
                    header=True,
                )
                test_df.to_csv(
                    self.data_validation_config.invalidate_test_file_path,
                    index=False,
                    header=True,
                )
                raise ValueError(f"Data validation failed:\n{error_message}")

            os.makedirs(self.data_validation_config.validate_data_dir, exist_ok=True)
            train_df.to_csv(
                self.data_validation_config.validate_train_file_path,
                index=False,
                header=True,
            )
            test_df.to_csv(
                self.data_validation_config.validate_test_file_path,
                index=False,
                header=True,
            )

            drift_status = self.detect_data_drift(base_df=train_df, current_df=test_df)

            return DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.validate_train_file_path,
                valid_test_file_path=self.data_validation_config.validate_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalidate_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalidate_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise ApplicationException(e, sys)
