import os
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    
    def start_data_ingestion(self):
        """Step 1: Data Ingestion"""
        try:
            logging.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            # This will return merged data path (merged_data_path)
            merged_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Merged data path: {merged_data_path}")
            return merged_data_path
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)

    def start_data_cleaning(self, merged_data_path):
        """Step 2: Data Cleaning"""
        try:
            logging.info("Starting data cleaning...")
            data_cleaning = DataCleaning(merged_data_path)
            # Data cleaning will return train_data_path and test_data_path
            train_data_path, test_data_path = data_cleaning.initiate_data_cleaning()
            logging.info(f"Data cleaning completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error(f"Error during data cleaning: {str(e)}")
            raise customexception(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path):
        """Step 3: Data Transformation"""
        try:
            logging.info("Starting data transformation...")
            data_transformation = DataTransformation()
            # Data transformation will return train_arr, test_arr, categorical_cols, numerical_cols, preprocessor
            train_arr, test_arr, categorical_cols, numerical_cols, preprocessor = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
            logging.info(f"Data transformation completed. Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")
            return train_arr, test_arr, categorical_cols, numerical_cols, preprocessor
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise customexception(e, sys)

    
    # ===modify below function input when using rfe for feature selection===
    
    # def start_model_training(self, train_arr, test_arr, categorical_cols, numerical_cols, preprocessor):
    def start_model_training(self, train_arr, test_arr):        
        """Step 4: Model Training"""
        try:
            logging.info("Starting model training...")
            model_trainer_obj = ModelTrainer()
            # This will start the model training process with the transformed training and testing data
            # model_trainer_obj.initiate_model_training(train_arr, test_arr, categorical_cols, numerical_cols, preprocessor)
            model_trainer_obj.initiate_model_training(train_arr, test_arr)
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise customexception(e, sys)
        
    # High-level method to orchestrate the entire pipeline
    def start_training(self):
        """Ensemble all stages of training pipeline"""
        try:
            # Step 1: Data Ingestion
            merged_data_path = self.start_data_ingestion()
            
            # Step 2: Data Cleaning
            train_data_path, test_data_path = self.start_data_cleaning(merged_data_path)
            
            # Step 3: Data Transformation
            train_arr, test_arr, categorical_cols, numerical_cols, preprocessor = self.start_data_transformation(train_data_path, test_data_path)
            
            # Step 4: Model Training
            # ===modify function input when using rfe for feature selection===
            
            # self.start_model_training(train_arr, test_arr, categorical_cols, numerical_cols, preprocessor)
            self.start_model_training(train_arr, test_arr)
            
        except Exception as e:
            logging.error(f"Error in the training pipeline: {str(e)}")
            raise customexception(e, sys)


# To trigger the training pipeline
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.start_training()
    except Exception as e:
        logging.error(f"Error in the entire training pipeline: {str(e)}")
        sys.exit(1)