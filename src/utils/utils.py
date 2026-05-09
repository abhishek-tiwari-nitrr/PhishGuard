import yaml, sys
import pandas as pd
from src.exception import ApplicationException


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


def write_yaml_file(filepath: str, content: object) -> None:
    """
    Write content into a YAML file.

    Args:
        filepath (str): Destination file path for the YAML file.
        content (object): Python object to be serialized into YAML format.

    Raises:
        ApplicationException: If writing the YAML file fails.
    """
    try:
        with open(filepath, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ApplicationException(e, sys)
