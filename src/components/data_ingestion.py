import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import customexception

@dataclass
class DataIngestionConfig:
    merged_data_path: str = os.path.join("data", "99acre_raw_data", "99acres_raw_data.csv")

class DataIngestion:
    def __init__(self, raw_data_path_1: str = "data/99acre_raw_data/flats.csv", raw_data_path_2: str = "data/99acre_raw_data/houses.csv"):
        # Initialize data ingestion output path for the merged file
        self.ingestion_config = DataIngestionConfig()
        self.raw_data_path_1 = raw_data_path_1  # Path to the first CSV file (flats)
        self.raw_data_path_2 = raw_data_path_2  # Path to the second CSV file (houses)
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read the two raw CSV files into separate dataframes
            data1 = pd.read_csv(self.raw_data_path_1)
            data2 = pd.read_csv(self.raw_data_path_2)
            
            logging.info(f"Reading data from source: {self.raw_data_path_1} and {self.raw_data_path_2}")

            # Add 'property_type' column to distinguish between flats and houses
            data1['property_type'] = 'flat'
            data2['property_type'] = 'house'

           
            # Concatenate the two dataframes into one
            data = pd.concat([data1, data2], ignore_index=True)
            logging.info(f"Both datasets concatenated successfully with shape: {data.shape}")
            
            # Save the concatenated data to the specified path
            data.to_csv(self.ingestion_config.merged_data_path, index=False)
            logging.info(f"Merged data saved at: {self.ingestion_config.merged_data_path}")

            logging.info("Data ingestion completed successfully.")
            
            # Return the path to the merged data
            return self.ingestion_config.merged_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)


# if __name__ == "__main__":
#     try:
#         # Initialize the DataIngestion class
#         obj = DataIngestion()
        
#         # Start the data ingestion process and get the path of the saved merged file
#         merged_file_path = obj.initiate_data_ingestion()
        
#         # Log the location of the merged file
#         logging.info(f"Merged file saved at: {merged_file_path}")
    
#     except Exception as e:
#         raise customexception(e, sys)