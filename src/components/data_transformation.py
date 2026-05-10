from src.exception import ApplicationException
from src.logger import logger
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
import sys, os
from src.utils.utils import read_csv_file_data, save_numpy_array_data, save_object
from src.constant.training_config import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import numpy as np


class DataTransformation:
    """
    Handles data preprocessing and transformation for the ML pipeline.

    Responsibilities:
        - Load validated training and testing datasets
        - Split features and target columns
        - Apply preprocessing transformations
        - Transform datasets into NumPy arrays
        - Save transformed datasets and preprocessing objects

    Attributes:
        - data_validation_artifacts (DataValidationArtifact): Artifact containing validated dataset file paths.
        - data_transformation_config (DataTransformationConfig): Configuration object containing transformation directories and file paths.

    """

    def __init__(
        self,
        data_validation_artifacts: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        """
        Initialize the DataTransformation class.

        Args:
            data_validation_artifacts (DataValidationArtifact): Artifact containing validated dataset file paths.
            data_transformation_config (DataTransformationConfig): Configuration object containing transformation directories and file paths.

        Raises:
            ApplicationException: If initialization fails.
        """
        try:
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ApplicationException(e, sys)

    def get_data_transform_object(self) -> Pipeline:
        """
        Create and return the data transformation pipeline.

        Raises:
            ApplicationException: If pipeline creation fails.

        Returns:
            Pipeline: Scikit-learn preprocessing pipeline object.

        """
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(
                f"KNNImputer initialized with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            return Pipeline([("imputer", imputer)])
        except Exception as e:
            raise ApplicationException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the complete data transformation pipeline.

        Steps:
            1. Load validated datasets
            2. Split features and target column
            3. Fit preprocessing pipeline on training data
            4. Transform training and testing datasets
            5. Combine transformed features with target labels
            6. Save transformed arrays
            7. Save preprocessing object
            8. Return transformation artifacts

        Raises:
            ApplicationException: If transformation pipeline execution fails.

        Returns:
            DataTransformationArtifact:
                Artifact containing:
                    - Transformed training array path
                    - Transformed testing array path
                    - Saved preprocessing object path
        """
        try:
            train_df = read_csv_file_data(
                self.data_validation_artifacts.valid_train_file_path
            )
            test_df = read_csv_file_data(
                self.data_validation_artifacts.valid_test_file_path
            )

            # split
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1, 0)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1, 0)

            preprocessor = self.get_data_transform_object()
            preprocessor.fit(X_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            os.makedirs(
                self.data_transformation_config.transformed_data_dir, exist_ok=True
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            os.makedirs(
                self.data_transformation_config.transformed_object_dir, exist_ok=True
            )
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor,
            )

            # save_object("final/preprocessor.pkl", preprocessor)

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            )
        except Exception as e:
            raise ApplicationException(e, sys)
