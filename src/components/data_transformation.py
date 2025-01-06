import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("data", "99acre_raw_data", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = 'price'  # Defined target column here

    def get_data_preprocessing(self, X: pd.DataFrame, y: pd.Series):
        try:
            logging.info('Preprocessing initiated')

            # Define categorical and numerical columns
            num_cols = X.select_dtypes(include=['number']).columns.tolist()
            num_cols = [col for col in num_cols if col != self.target_column_name]  # Exclude target column 'price'
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")

            # Numerical Pipeline using StandardScaler
            num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

            # Categorical Pipeline using OneHotEncoder
            cat_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

            # Create the preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[('num_pipeline', num_pipeline, num_cols),
                              ('cat_pipeline', cat_pipeline, cat_cols)]
            )
            
            return preprocessor

        except Exception as e:
            logging.error("Exception occurred in get_data_preprocessing")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train & test data started")
            logging.info(f"Train path: {train_path}, Test path: {test_path}")

            # Load the datasets
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)

            logging.info("Reading train & test data completed")            
            logging.info(f"First few rows of train_df: \n{train_df.head()}")

            # Strip whitespace from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Drop the target column from both train and test sets (just for X_train and X_test)
            logging.info("Dropping target column and preparing features and target")
            X_train = train_df.drop(columns=[self.target_column_name])  # Feature columns (without target)
            y_train = train_df[self.target_column_name]  # Target column

            X_test = test_df.drop(columns=[self.target_column_name])  # Feature columns (without target)
            y_test = test_df[self.target_column_name]  # Target column
            
            # Define categorical and numerical columns
            num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
            num_cols = [col for col in num_cols if col != self.target_column_name]  # Exclude target column 'price'
            cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Numerical Columns: {num_cols}")

            # Apply preprocessing (pass only features X_train)
            preprocessing_obj = self.get_data_preprocessing(X_train, y_train)
            X_train_transformed = preprocessing_obj.fit_transform(X_train)  # Apply transformation to feature columns only
            X_test_transformed = preprocessing_obj.transform(X_test)  # Apply transformation to feature columns only

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Concatenate transformed features with the target variable to form final datasets
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save the preprocessor object to disk
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info("Preprocessing pickle file saved")

            # Log shapes of the transformed arrays
            logging.info(f"Training data shape: {train_arr.shape}")
            logging.info(f"Testing data shape: {test_arr.shape}")

            return train_arr, test_arr, cat_cols, num_cols, preprocessing_obj

        except Exception as e:
            logging.error(f"Exception occurred in initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)