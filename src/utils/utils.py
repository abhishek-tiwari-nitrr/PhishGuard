import yaml, sys
import pandas as pd
import numpy as np
import pickle
from src.exception import ApplicationException
from src.logger import logger


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
