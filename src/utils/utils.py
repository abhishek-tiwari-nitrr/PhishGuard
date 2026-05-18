import yaml, sys
import pandas as pd
import numpy as np
import pickle, os
from src.exception import ApplicationException
from src.logger import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from src.entity.artifact_entity import ClassificationModelArtifact


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Raises:
        ApplicationException: If reading or parsing the YAML file fails.

    Returns:
        dict: Parsed YAML content.

    """
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ApplicationException(e, sys)


def read_csv_file_data(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame.

    Args:
        file_path (str):  Path to the CSV file.

    Raises:
        ApplicationException: If reading the CSV file fails.

    Returns:
        pd.DataFrame: DataFrame containing CSV data.

    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ApplicationException(e, sys)


def write_yaml_file(file_path: str, content: object) -> None:
    """
    Write content into a YAML file.

    Args:
        filepath (str): Destination file path for the YAML file.
        content (object): Python object to be serialized into YAML format.

    Raises:
        ApplicationException: If writing the YAML file fails.
    """
    try:
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ApplicationException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save a NumPy array to a binary `.npy` file.

    Args:
        file_path (str): Path to save Numpy Array.
        array (np.ndarray): NumPy array to save.

    Raises:
        ApplicationException: If saving the NumPy array fails.

    """
    try:
        with open(file_path, "wb") as file:
            np.save(file, array)
    except Exception as e:
        raise ApplicationException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Serialize and save a Python object using pickle.

    Args:
        file_path (str): Path to save Object.
        obj (object): Python object to serialize and save.

    Raises:
        ApplicationException: If object serialization or saving fails.

    """
    try:
        with open(file_path, "wb") as file:
            logger.info(f"Saving object to {file}")
            pickle.dump(obj, file)
    except Exception as e:
        raise ApplicationException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from a binary file.

    Args:
        file_path (str): Path to the NumPy binary file to be loaded.

    Raises:
        ApplicationException: If opening/loading the NumPy array fails.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    try:
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise ApplicationException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple classification models using GridSearchCV.

    Args:
        X_train (array): Training feature dataset.
        y_train (array): Training target labels.
        X_test (array): Testing feature dataset.
        y_test (array): Testing target labels.
        models (dict): Dictionary containing model names as keys and model objects as values.
        param (dict): Dictionary containing hyperparameter grids for each model.

    Raises:
        ApplicationException: Raised if model training or evaluation fails.

    Returns:
        dict: Dictionary containing evaluation metrics for each model.

    """
    try:
        report = {}
        for model_name, model in models.items():
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, scoring="f1_weighted")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            precision = precision_score(
                y_test, y_test_pred, average="weighted", zero_division=0
            )

            recall = recall_score(
                y_test, y_test_pred, average="weighted", zero_division=0
            )

            accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                "train_f1_score": train_f1,
                "test_f1_score": test_f1,
                "precision_score": precision,
                "recall_score": recall,
                "accuracy_score": accuracy,
                "best_params": gs.best_params_,
            }

        return report
    except Exception as e:
        raise ApplicationException(e, sys)


def get_classification_score(y_true, y_pred) -> ClassificationModelArtifact:
    """
    Calculate classification evaluation metrics.

    Args:
        y_true (array): Actual target labels.
        y_pred (array): Predicted target labels.

    Raises:
        ApplicationException: Raised if evaluation fails.

    Returns:
        ClassificationModelArtifact: Object containing F1-score, precision, and recall.

    """
    try:
        return ClassificationModelArtifact(
            f1_score=f1_score(y_true, y_pred, average="weighted"),
            precision_score=precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            recall_score=recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        )
    except Exception as e:
        raise ApplicationException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load and deserialize a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing the serialized object.

    Raises:
        ApplicationException: Raised if object cannot be loaded.

    Returns:
        object: The deserialized Python object loaded from the file.
    """
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise ApplicationException(e, sys)


def production_model_exists(file_path: str) -> bool:
    """
    Checks weather Production Model File Path Exists or not.

    Args:
        file_path (str): Production Model File Path

    Raises:
        ApplicationException: Raised if problem found

    Returns:
        bool: True if it exists. Otherwise False
    """
    try:
        if not os.path.exists(file_path):
            logger.info("No production model found - first deployment")
            return False
        else:
            logger.info("Production Model Exists")
            return True
    except Exception as e:
        raise ApplicationException(e, sys)