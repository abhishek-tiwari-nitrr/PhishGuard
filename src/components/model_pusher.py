from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.logger import logger
from src.exception import ApplicationException
import os, sys, shutil


class ModelPusher:
    """
    Handles deployment of the accepted machine learning model to the production environment.

    Responsibilities:
        - Checking whether the newly trained model was accepted
        - Creating the production model directory if it does not exist
        - Copying the trained model to the production location

    Attributes:
        - model_evaluation_artifact (ModelEvaluationArtifact): Artifact containing model evaluation results and trained model path.
        - model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation threshold and paths.

    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        """
        Initialize the ModelPusher component.

        Args:
            - model_evaluation_artifact (ModelEvaluationArtifact): Artifact containing model evaluation results and trained model path.
            - model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation threshold and paths.

        Raises:
            - ApplicationException: If initialization fails due to configuration or setup errors.
        """
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise ApplicationException(e, sys)

    def initiate_model_pusher(self) -> bool:
        """
        Copy the new model to  production_model/model.pkl  when it was accepted.

        Steps:
            1. Checks whether the model was accepted during evaluation
            2. Creates the production model directory if it does not exist
            3. Copies the trained model file to the production location

        Raises:
            - ApplicationException: Raised when any error occurs during the model deployment process.

        Returns:
            - bool: True when model pushed successfully. False when model was not accepted; production model unchanged.
        """
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logger.info(
                    "Model Pusher: skipped - new model was not accepted. Production model remains unchanged"
                )
                return False
            source = self.model_evaluation_artifact.trained_model_file_path
            destination = self.model_evaluation_config.production_model_file_path

            os.makedirs(
                self.model_evaluation_config.production_model_dir, exist_ok=True
            )
            shutil.copy2(source, destination)
            logger.info(f"Model pushed to production: {destination}")
            return True
        except Exception as e:
            raise ApplicationException(e, sys)
