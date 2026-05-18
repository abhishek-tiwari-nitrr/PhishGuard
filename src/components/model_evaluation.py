from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import ApplicationException
from src.logger import logger
import sys, os
from src.utils.utils import (
    load_numpy_array_data,
    load_object,
    get_classification_score,
    production_model_exists,
    write_yaml_file,
)


class ModelEvaluation:
    """
    Evaluates the newly trained machine learning model against the existing production model using classification metrics such as F1-score.

    Responsibilities:
        - Loading transformed test data
        - Loading the newly trained model
        - Comparing the new model with the production model (if available)
        - Deciding whether the new model should replace the production model
        - Generating and saving evaluation reports

    Attributes:
        - model_trainer_artifact (ModelTrainerArtifact): Contains trained model path and evaluation metrics.
        - data_transformation_artifact (DataTransformationArtifact): Artifact containing validated dataset file paths.
        - model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation threshold and paths.

    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        """
        Initialize the ModelEvaluation component.

        Args:
            - model_trainer_artifact (ModelTrainerArtifact): Contains trained model path and evaluation metrics.
            - data_transformation_artifact (DataTransformationArtifact): Artifact containing validated dataset file paths.
            - model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation threshold and paths.

        Raises:
            - ApplicationException: If initialization fails due to configuration or setup errors.

        """
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise ApplicationException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Performs evaluation of the newly trained model against the production model using transformed test data.

        Steps:
            1. Load transformed test dataset
            2. Load newly trained model
            3. Evaluate new model performance
            4. Compare with production model (if exists)
            5. Decide model acceptance based on threshold score
            6. Generate evaluation report
            7. Save report as YAML file
            8. Generate evaluation artifact

        Raises:
            - ApplicationException: If any error occurs during model evaluation.

        Returns:
            - ModelEvaluationArtifact:
                Artifact containing:
                    - model acceptance status
                    - performance improvement
                    - trained model path
                    - production model path
                    - evaluation metrics
                    
        """
        try:
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            new_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            new_model_metric = get_classification_score(
                y_true=y_test, y_pred=new_model.predict(X_test)
            )
            logger.info(
                f"New Model | F1 Score: {new_model_metric.f1_score:.4f} | Precision Score: {new_model_metric.precision_score:.4f} | Recall Score: {new_model_metric.recall_score:.4f}"
            )

            if production_model_exists(
                self.model_evaluation_config.production_model_file_path
            ):
                production_model = load_object(
                    self.model_evaluation_config.production_model_file_path
                )
                production_model_metric = get_classification_score(
                    y_true=y_test, y_pred=production_model.predict(X_test)
                )
                logger.info(
                    f"Production Model | F1 Score: {production_model_metric.f1_score:.4f} | Precision Score: {production_model_metric.precision_score:.4f} | Recall Score: {production_model_metric.recall_score:.4f}"
                )
                delta = new_model_metric.f1_score - production_model_metric.f1_score
                is_accepted = (
                    delta >= self.model_evaluation_config.changed_threshold_score
                )
                if is_accepted:
                    production_model_file_path = (
                        self.model_trainer_artifact.trained_model_file_path
                    )
                    logger.info(f"New model Accpetd  |  F1 = +{delta:.4f}")
                else:
                    production_model_file_path = (
                        self.model_evaluation_config.production_model_file_path
                    )
                    logger.info(f"New model Rejected  |  F1 = +{delta:.4f}")
            else:
                # first deployment
                is_accepted = True
                delta = 0.0
                production_model_metric = new_model_metric
                production_model_file_path = (
                    self.model_trainer_artifact.trained_model_file_path
                )
                logger.info("Auto accepted (first deployment)")

            report = {
                "is_model_accepted": is_accepted,
                "improved_accuracy": float(delta),
                "threshold": self.model_evaluation_config.changed_threshold_score,
                "new_model_metric": {
                    "f1_score": float(new_model_metric.f1_score),
                    "precision_score": float(new_model_metric.precision_score),
                    "recall_score": float(new_model_metric.recall_score),
                },
                "production_model_metric": {
                    "f1_score": float(production_model_metric.f1_score),
                    "precision_score": float(production_model_metric.precision_score),
                    "recall_score": float(production_model_metric.recall_score),
                },
            }

            os.makedirs(
                self.model_evaluation_config.model_evaluation_dir, exist_ok=True
            )
            write_yaml_file(
                file_path=self.model_evaluation_config.report_file_path, content=report
            )

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                improved_accuracy=float(delta),
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
                production_model_file_path=production_model_file_path,
                train_metric_artifact=new_model_metric,
                production_metric_artifact=production_model_metric,
            )
        except Exception as e:
            raise ApplicationException(e, sys)
