from datetime import datetime
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.xlsx")
    train_data_path: str = os.path.join("artifacts", "train.xlsx")
    test_data_path: str = os.path.join("artifacts", "test.xlsx")


class DataIngestion:
    def __init__(self, raw_data_path: str = "cars24_final_data.xlsx"):
        # Initialize data ingestion output paths
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path = raw_data_path  # Allow flexible data path for input

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        impliment basic data cleaning, feature engineering here except missing value imputation, scaling & encoding
        """
        logging.info("Data cleaning started")

        #drop all duplicate rows
        df.drop_duplicates(inplace=True)
        
        #perform cleaning
        
        return df

    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read raw data from the source (e.g., a local file or cloud)
            data = pd.read_excel(self.raw_data_path)
            logging.info(f"Reading data from source: {self.raw_data_path}")

            # Clean the data
            data = self.initial_data_cleaning(data)
                        
            # Ensure the directory for saving files exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the cleaned data to the specified path
            data.to_excel(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Perform train-test split (75% train, 25% test)
            logging.info("Train-test split started")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save the train and test datasets to specified paths
            train_data.to_excel(self.ingestion_config.train_data_path, index=False)
            test_data.to_excel(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed")
            
            #return train &* test data path
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)
        
        
# if __name__ == "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()