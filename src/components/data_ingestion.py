from src.entity.config_entity import DataIngestionConfig
from src.exception import ApplicationException
import sys, os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi
import pandas as pd
import numpy as np
from src.logger import logger
from sklearn.model_selection import train_test_split
from src.entity.artifact_entity import DataIngestionArtifact

load_dotenv(override=True)
ca = certifi.where()


class DataIngestion:
    """
    Handles the complete data ingestion pipeline.

    This class is responsible for:
        - Establishing a connection with MongoDB
        - Exporting data from a MongoDB collection
        - Saving raw data into the raw data directory
        - Splitting the dataset into train and test sets
        - Saving train and test datasets into the ingested directory

    Attributes:
        - data_ingestion_config (DataIngestionConfig): Configuration object containing ingestion-related paths and settings
        - mongo_client (MongoClient): MongoDB client instance used for database connection.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize the DataIngestion class.

        Args:
            - data_ingestion_config (DataIngestionConfig): Configuration object containing database name, collection name, file paths, and split ratio.

        Raises:
            - ApplicationException: If initialization fails.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.mongo_client = None
        except Exception as e:
            raise ApplicationException(e, sys)

    def _get_mongo_client(self):
        """
        Create and return a MongoDB client instance.

        Raises:
            - ValueError: If the MONGO_DB_URL environment variable is not set.
            - ApplicationException : If MongoDB connection creation fails.

        Returns:
            MongoClient: MongoDB client instance.
        """
        if self.mongo_client is None:
            mongo_db_url = os.getenv("MONGO_DB_URL")
            if not mongo_db_url:
                raise ValueError("Please set MONGO_DB_URL value in .env")
            self.mongo_client = MongoClient(mongo_db_url, tls=True, tlsCAFile=ca)
        return self.mongo_client

    def export_data(self) -> pd.DataFrame:
        """
        Export data from the MongoDB collection into a pandas DataFrame.

        Raises:
            - ApplicationException: If data export fails.

        Returns:
            - pd.DataFrame: Cleaned DataFrame containing exported collection data.
        """
        try:
            client = self._get_mongo_client()
            collection_data = client[self.data_ingestion_config.database_name][
                self.data_ingestion_config.collection_name
            ]
            df = pd.DataFrame(list(collection_data.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            logger.info(f"Exported {len(df)} records from MongoDB")
            return df
        except Exception as e:
            raise ApplicationException(e, sys)

    def export_data_into_raw_folder(self, dataframe: pd.DataFrame):
        """
        Save the raw DataFrame into the raw data directory as a CSV file.

        Args:
            - dataframe (pd.DataFrame): DataFrame to be saved.

        Raises:
            - ApplicationException: If saving the raw data fails.

        Returns:
            - pd.DataFrame: The same input DataFrame.
        """
        try:
            raw_dir = self.data_ingestion_config.data_ingestion_raw_dir
            raw_file_path = self.data_ingestion_config.raw_file_path
            os.makedirs(raw_dir, exist_ok=True)
            dataframe.to_csv(raw_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise ApplicationException(e, sys)

    def export_data_into_ingested_folder(self, dataframe: pd.DataFrame):
        """
        Split the dataset into training and testing datasets and save them into the ingested data directory.

        Args:
            - dataframe (pd.DataFrame): Input DataFrame to split and to be saved.

        Raises:
            - ApplicationException: If train-test splitting or file export fails.
        """
        try:
            ingested_dir = self.data_ingestion_config.data_ingestion_ingest_dir
            os.makedirs(ingested_dir, exist_ok=True)
            train_data, test_data = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=self.data_ingestion_config.random_state
            )
            logger.info("Exporting train and test file path")
            train_data.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_data.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.info(f"Exported train and test file path.")
        except Exception as e:
            raise ApplicationException(e, sys)

    def initiate_data_ingestion(self):
        """
        Execute the complete data ingestion pipeline.

        Steps:
            1. Export data from MongoDB
            2. Save raw dataset into raw directory
            3. Split data into train and test datasets
            4. Save train and test datasets
            5. Create and return ingestion artifacts

        Raises:
            - ApplicationException: If the ingestion pipeline execution fails.

        Returns:
            - DataIngestionArtifact:
                    Artifact object containing paths to:
                        - Raw dataset
                        - Training dataset
                        - Testing dataset
        """
        try:
            export_data = self.export_data()
            dataframe = self.export_data_into_raw_folder(export_data)
            self.export_data_into_ingested_folder(dataframe)
            data_ingestion_artifacts = DataIngestionArtifact(
                raw_file_path=self.data_ingestion_config.raw_file_path,
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifacts
        except Exception as e:
            raise ApplicationException(e, sys)
