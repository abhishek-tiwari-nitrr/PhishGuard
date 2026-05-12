import sys, os
from src.logger import logger
from src.exception import ApplicationException
import mlflow
import dagshub
from src.constant.config import (
    DAGSHUB_TOKEN,
    LOCAL_TRACKING_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_REPO_NAME,
    DAGSHUB_TRACKING_URI,
)
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.utils import (
    load_numpy_array_data,
    evaluate_models,
    get_classification_score,
    load_object,
    save_object,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from urllib.parse import urlparse
from src.utils.model_utils import PhishGuard


def _configure_mlflow():
    """
    Configure MLflow tracking with optional DagsHub integration.
    """
    if DAGSHUB_TRACKING_URI and DAGSHUB_TOKEN:
        try:
            logger.info("Initializing DagsHub MLflow...")

            # auth
            dagshub.auth.add_app_token(DAGSHUB_TOKEN)

            # initialize dagshub
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True
            )

            # set tracking uri
            mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
            mlflow.set_experiment("PhishGuard")
            # test connection
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()
            logger.info(f"Connected to DagsHub MLflow: {DAGSHUB_TRACKING_URI}")

        except Exception as e:
            logger.warning("DagsHub MLflow failed. Falling back to local MLflow.")
            logger.error(f"DagsHub Error: {ApplicationException(e, sys)}")

            # fallback local
            mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
            mlflow.set_experiment("PhishGuard")
            logger.info(f"Connected to Local MLflow: {LOCAL_TRACKING_URI}")
    else:
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
        mlflow.set_experiment("PhishGuard")
        logger.info(
            f"No DagsHub configuration found. Using local MLflow: {LOCAL_TRACKING_URI}"
        )


class ModelTrainer:
    """
    Handles training, evaluation, selection and logging of ML models for phishing detection.

    Responsibilities:
        - Trains multiple classification models
        - Performs hyperparameter tuning
        - Selects best performing model using F1-score
        - Logs experiments to MLflow / DagsHub
        - Saves final trained pipeline (preprocessor + model)
        - Generates classification metrics artifacts

    Attributes:
        - data_transformation (DataTransformationArtifact): Contains transformed train/test dataset paths and preprocessor path.
        - model_trainer_config (ModelTrainerConfig): Configuration for training such as model path and thresholds.

    """

    def __init__(
        self,
        data_transformation: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        Initialize the ModelTrainer component.

        Args:
            data_transformation (DataTransformationArtifact): Contains transformed train/test dataset paths and preprocessor path.
            model_trainer_config (ModelTrainerConfig): Configuration for training such as model path and thresholds.

        Raises:
            ApplicationException: If initialization fails due to configuration or setup errors.

        """
        try:
            self.data_transformation = data_transformation
            self.model_trainer_config = model_trainer_config
            _configure_mlflow()
        except Exception as e:
            raise ApplicationException(e, sys)

    def _log_model_run(
        self,
        model_name: str,
        model,
        best_params: dict,
        train_f1: float,
        test_f1: float,
        precision: float,
        recall: float,
        accuracy: float,
        is_best: bool,
        preprocessor,
    ):
        """
        Log model metrics and artifacts to MLflow.

        Args:
            model_name (str): Name of the model.
            model (Any): Trained model object.
            best_params (dict): Best hyperparameters.
            train_f1 (float): Training F1-score.
            test_f1 (float): Testing F1-score.
            precision (float): Precision score.
            recall (float): Recall score.
            accuracy (float): Accuracy score.
            is_best (bool): Whether model is best performer.
            preprocessor (Any): Fitted preprocessing pipeline.
        """
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("is_best_model", str(is_best))
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_metric("train_f1_score", train_f1)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("precision_score", precision)
            mlflow.log_metric("recall_score", recall)
            mlflow.log_metric("accuracy_score", accuracy)
            mlflow.log_metric("train_test_f1_gap", abs(train_f1 - test_f1))
            if is_best:
                pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )
                if tracking_url_type != "file":
                    mlflow.sklearn.log_model(
                        pipeline,
                        "pipeline",
                        registered_model_name=f"PhishGuard_{model_name.replace(' ', '_')}",
                    )
                else:
                    mlflow.sklearn.log_model(pipeline, "pipeline")
            else:
                mlflow.sklearn.log_model(model, "model")

            logger.info(
                f"Logged MLflow run for: {model_name} | test_f1={test_f1:.4f} | best={is_best}"
            )

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """
        Train multiple models and select the best one.

        Steps:
            1. Define models and hyperparameters
            2. Evaluate models using GridSearchCV
            3. Select best model using F1-score
            4. Log all experiments to MLflow
            5. Check overfitting/underfitting
            6. Save final pipeline

        Args:
            - X_train (np.ndarray): Training features
            - y_train (np.ndarray): Training labels
            - X_test (np.ndarray): Testing features
            - y_test (np.ndarray): Testing labels

        Raises:
            - ValueError: If no model meets expected performance threshold.

        Returns:
            - ModelTrainerArtifact: Contains trained model path and evaluation metrics.
        """
        models = {
            "Random Forest": RandomForestClassifier(verbose=0),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=0),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.05, 0.01],
                "subsample": [0.7, 0.85, 0.9],
                "n_estimators": [32, 64, 128],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [32, 64, 128],
            },
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_name = max(
            model_report,
            key=lambda model_name: model_report[model_name]["test_f1_score"],
        )
        best_model_score = model_report[best_model_name]["test_f1_score"]
        best_model = models[best_model_name]

        logger.info(
            f"Best model: {best_model_name} with Test F1 Score = {best_model_score:.4f}"
        )

        if best_model_score < self.model_trainer_config.expected_accuracy:
            raise ValueError(
                f"No model achieved the expected performance score of "
                f"{self.model_trainer_config.expected_accuracy}. "
                f"Best model score was {best_model_score:.4f} "
                f"for model {best_model_name}"
            )

        preprocessor = load_object(
            self.data_transformation.transformed_object_file_path
        )

        for model_name, metrics in model_report.items():
            self._log_model_run(
                model_name=model_name,
                model=models[model_name],
                best_params=metrics["best_params"],
                train_f1=metrics["train_f1_score"],
                test_f1=metrics["test_f1_score"],
                precision=metrics["precision_score"],
                recall=metrics["recall_score"],
                accuracy=metrics["accuracy_score"],
                is_best=(model_name == best_model_name),
                preprocessor=preprocessor,
            )

        train_metric = get_classification_score(y_train, best_model.predict(X_train))
        test_metric = get_classification_score(y_test, best_model.predict(X_test))

        score_diff = abs(train_metric.f1_score - test_metric.f1_score)
        if score_diff > self.model_trainer_config.overfitting_underfitting_threshold:
            logger.warning(
                f"Train/test F1 gap: {score_diff} exceeds threshold: {self.model_trainer_config.overfitting_underfitting_threshold} - possible overfitting."
            )

        os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)

        phish_guard_model = PhishGuard(preprocessor=preprocessor, model=best_model)
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=phish_guard_model,
        )

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Execute full model training pipeline.

        Steps:
            1. Load training and testing arrays
            2. Split features and labels
            3. Call training pipeline

        Raises:
            ApplicationException: If pipeline execution fails.

        Returns:
            ModelTrainerArtifact: Final trained model and metrics.
        """
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)
        except ApplicationException:
            raise
        except Exception as e:
            raise ApplicationException(e, sys)
