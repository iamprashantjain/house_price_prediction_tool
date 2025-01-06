# import pandas as pd
# import numpy as np
# from src.logger.logging import logging
# from src.exception.exception import customexception
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from src.utils.utils import save_object, evaluate_model
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import r2_score
# import joblib

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("data", "model.pkl")
#     # trained_model_type_path = os.path.join("data", "model_type.pkl")  # Add path to save model type


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def calculate_adjusted_r2(self, model, X, y):
#         """
#         Calculate the Adjusted R2 score for a given model.
#         """
#         # Fit the model to the data
#         model.fit(X, y)
#         r2 = r2_score(y, model.predict(X))

#         # Number of samples and features
#         n = X.shape[0]
#         p = X.shape[1]

#         # Adjusted R2 formula
#         adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#         return adjusted_r2

#     def initiate_model_training(self, train_array, test_array):
#             try:
#                 logging.info('Splitting Dependent and Independent variables from train and test data')
#                 X_train, y_train, X_test, y_test = (
#                     train_array[:, :-1],  # Features from the training data
#                     train_array[:, -1],   # Target from the training data
#                     test_array[:, :-1],   # Features from the test data
#                     test_array[:, -1]     # Target from the test data
#                 )

#                 # Define the models to be tested
#                 models = {
#                     'LinearRegression': LinearRegression(),
#                     'Lasso': Lasso(),
#                     'Ridge': Ridge(),
#                     'ElasticNet': ElasticNet(),
#                     'DecisionTreeRegressor': DecisionTreeRegressor(),
#                     'RandomForestRegressor': RandomForestRegressor(),
#                     'SVR': SVR(),
#                     'KNeighborsRegressor': KNeighborsRegressor(),
#                     'GaussianProcessRegressor': GaussianProcessRegressor(),
#                     'XGBoost': xgb.XGBRegressor(),
#                     'LightGBM': lgb.LGBMRegressor(),
#                     'CatBoost': CatBoostRegressor(verbose=0)
#                 }

#                 # Use the evaluate_model function frim utils to get the model report
#                 model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

#                 # Print the model report
#                 print(model_report)
#                 logging.info(f'Model Report: {model_report}')

#                 # Select the best model based on Adjusted R2 score
#                 best_model_name = max(model_report, key=lambda k: model_report[k]['Adjusted R2 Score'])
#                 best_model_score = model_report[best_model_name]['Adjusted R2 Score']
#                 best_model = models[best_model_name]

#                 print(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')
#                 logging.info(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')

#                 # Save the best model
#                 save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
#                 logging.info("model.pkl file saved in artifacts")
#                 # joblib.dump(best_model_name, self.model_trainer_config.trained_model_type_path)  # Save model type

#             except Exception as e:
#                 logging.info('Exception occurred during model training')
#                 raise customexception(e, sys)



# ========================= integrating mlflow in above code ==================

import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data", "model.pkl")
    # trained_model_type_path = os.path.join("data", "model_type.pkl")  # Add path to save model type


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def calculate_adjusted_r2(self, model, X, y):
        """
        Calculate the Adjusted R2 score for a given model.
        """
        # Fit the model to the data
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))

        # Number of samples and features
        n = X.shape[0]
        p = X.shape[1]

        # Adjusted R2 formula
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features from the training data
                train_array[:, -1],   # Target from the training data
                test_array[:, :-1],   # Features from the test data
                test_array[:, -1]     # Target from the test data
            )

            # Define the models to be tested
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'SVR': SVR(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'GaussianProcessRegressor': GaussianProcessRegressor(),
                'XGBoost': xgb.XGBRegressor(),
                'LightGBM': lgb.LGBMRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }

            # Start MLflow run to log the experiment
            with mlflow.start_run():

                # Use the evaluate_model function to get the model report
                model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

                # Print the model report
                print(model_report)
                logging.info(f'Model Report: {model_report}')

                # Log the model performance metrics in MLflow
                for model_name, model_metrics in model_report.items():
                    mlflow.log_metric(f'{model_name}_Adjusted_R2', model_metrics['Adjusted R2 Score'])
                    logging.info(f'Logged Adjusted R2 for {model_name}: {model_metrics["Adjusted R2 Score"]}')

                # Select the best model based on Adjusted R2 score
                best_model_name = max(model_report, key=lambda k: model_report[k]['Adjusted R2 Score'])
                best_model_score = model_report[best_model_name]['Adjusted R2 Score']
                best_model = models[best_model_name]

                print(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')
                logging.info(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')

                # Log parameters of the best model in MLflow (Optional, if needed)
                if hasattr(best_model, 'alpha'):  # Example for Lasso, Ridge, etc.
                    mlflow.log_param('alpha', best_model.alpha)

                # Log the model in MLflow
                if isinstance(best_model, xgb.XGBRegressor):
                    mlflow.xgboost.log_model(best_model, "model")
                elif isinstance(best_model, lgb.LGBMRegressor):
                    mlflow.lightgbm.log_model(best_model, "model")
                elif isinstance(best_model, CatBoostRegressor):
                    mlflow.catboost.log_model(best_model, "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

                # Save the best model to a file (for persistence)
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
                logging.info("model.pkl file saved in artifacts")

                # Optionally save the model type (commented out in your original code)
                # joblib.dump(best_model_name, self.model_trainer_config.trained_model_type_path)  # Save model type

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise customexception(e, sys)


# =============== applying feature selection: rfe =============================

# import os
# import sys
# import numpy as np
# import pandas as pd
# from src.logger.logging import logging
# from src.exception.exception import customexception
# from dataclasses import dataclass
# from src.utils.utils import save_object, evaluate_model
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostRegressor
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.feature_selection import RFE
# import joblib
# from sklearn.metrics import r2_score

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("data", "99acre_raw_data", "model.pkl")
#     trained_model_type_path = os.path.join("data", "99acre_raw_data", "model_type.pkl")  # Add path to save model type


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def calculate_adjusted_r2(self, model, X, y):
#         """
#         Calculate the Adjusted R2 score for a given model.
#         """
#         # Fit the model to the data
#         model.fit(X, y)
#         r2 = r2_score(y, model.predict(X))

#         # Number of samples and features
#         n = X.shape[0]
#         p = X.shape[1]

#         # Adjusted R2 formula
#         adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#         return adjusted_r2

#     def get_param_grid(self, model):
#         """
#         Returns a dictionary of hyperparameters for each model.
#         """
#         if isinstance(model, LinearRegression):
#             return {'fit_intercept': [True, False], 'positive': [True, False]}
#         elif isinstance(model, RandomForestRegressor):
#             return {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
#         elif isinstance(model, SVR):
#             return {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
#         elif isinstance(model, KNeighborsRegressor):
#             return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
#         elif isinstance(model, DecisionTreeRegressor):
#             return {'max_depth': [3, 5, 7, 10]}
#         elif isinstance(model, Lasso):
#             return {'alpha': [0.1, 1.0, 10.0]}
#         elif isinstance(model, Ridge):
#             return {'alpha': [0.1, 1.0, 10.0]}
#         elif isinstance(model, ElasticNet):
#             return {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
#         elif isinstance(model, xgb.XGBRegressor):
#             return {
#                 'n_estimators': [50, 100],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'max_depth': [3, 5, 7],
#                 'subsample': [0.8, 1.0],
#                 'colsample_bytree': [0.8, 1.0],
#                 'gamma': [0, 0.1, 0.2]
#             }

#     def tune_model_hyperparameters(self, model, param_grid, X_train, y_train):
#         """
#         Hyperparameter tuning using GridSearchCV.
#         """
#         # Perform Grid Search
#         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
#         grid_search.fit(X_train, y_train)

#         return grid_search.best_params_, grid_search.best_score_

#     def select_features_using_rfe(self, model, X_train, y_train, num_features_to_select=25):
#         """
#         Select features using RFE (Recursive Feature Elimination).
#         """
#         # Perform RFE for feature selection
#         selector = RFE(estimator=model, n_features_to_select=num_features_to_select)
#         selector.fit(X_train, y_train)

#         # Get the selected features based on RFE
#         selected_features = selector.support_

#         # Return the data with selected features only
#         return X_train[:, selected_features], selected_features

#     def cross_validate_model(self, model, X_train, y_train, preprocessor):
#         """
#         Perform K-fold cross-validation and return the mean and standard deviation of the R^2 scores.
#         """
#         # Apply the preprocessor to the training data to transform it
#         X_train_preprocessed = preprocessor.fit_transform(X_train)

#         # Perform cross-validation with 5 folds
#         scores = cross_val_score(model, X_train_preprocessed, y_train, cv=5, scoring='r2')

#         # Return the mean and standard deviation of the R^2 scores
#         return scores.mean(), scores.std()

    
#     def initiate_model_training(self, train_arr, test_arr, categorical_cols, numerical_cols, preprocessor):
#         try:
#             logging.info('Splitting Dependent and Independent variables from train and test data')
#             X_train, y_train, X_test, y_test = (
#                 train_arr[:, :-1],  # Features from the training data
#                 train_arr[:, -1],   # Target from the training data
#                 test_arr[:, :-1],   # Features from the test data
#                 test_arr[:, -1]     # Target from the test data
#             )

#             # Get the categorical columns' OneHotEncoded feature names
#             cat_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
#             all_feature_names = numerical_cols + list(cat_feature_names)

#             # Define the models to be tested
#             models = {
#                 'LinearRegression': LinearRegression(),
#                 'Lasso': Lasso(),
#                 'Ridge': Ridge(),
#                 'ElasticNet': ElasticNet(),
#                 'DecisionTreeRegressor': DecisionTreeRegressor(),
#                 'RandomForestRegressor': RandomForestRegressor(),
#                 'SVR': SVR(),
#                 'KNeighborsRegressor': KNeighborsRegressor(),
#                 'GaussianProcessRegressor': GaussianProcessRegressor(),
#                 'XGBoost': xgb.XGBRegressor(),
#                 'LightGBM': lgb.LGBMRegressor(),
#                 'CatBoost': CatBoostRegressor(verbose=0)
#             }
            

#             # Select features using RFE for numerical columns
#             model_to_use_for_rfe = RandomForestRegressor()  # You can choose any model here
#             X_train_rfe, selected_features_rfe = self.select_features_using_rfe(model_to_use_for_rfe, X_train, y_train)

#             # After RFE, we'll select only the OneHotEncoded features
#             selected_features = numerical_cols + list(cat_feature_names)  # Select all OHE features

#             # Evaluate all models using the 'evaluate_model' function
#             model_report = evaluate_model(X_train_rfe, y_train, X_test[:, selected_features_rfe], y_test, models)

#             # Print the model report
#             print(model_report)
#             logging.info(f'Model Report: {model_report}')

#             # Select the best model based on Adjusted R2 score
#             best_model_name = max(model_report, key=lambda k: model_report[k]['Adjusted R2 Score'])
#             best_model_score = model_report[best_model_name]['Adjusted R2 Score']
#             best_model = models[best_model_name]

#             print(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')
#             logging.info(f'Best Model Found: {best_model_name} with Adjusted R2 Score: {best_model_score}')

#             # Hyperparameter tuning for the best model
#             param_grid = self.get_param_grid(best_model)
#             best_params, best_score = self.tune_model_hyperparameters(best_model, param_grid, X_train_rfe, y_train)

#             # Log best parameters and score
#             logging.info(f'Best Hyperparameters: {best_params}')
#             logging.info(f'Best Hyperparameter Score: {best_score}')

#             # Perform cross-validation on the best model
#             cv_mean, cv_std = self.cross_validate_model(best_model, X_train_rfe, y_train, preprocessor)
#             logging.info(f'Cross-validation results: Mean R² = {cv_mean}, Std R² = {cv_std}')

#             # Fit the model with the best hyperparameters
#             best_model.set_params(**best_params)
#             best_model.fit(X_train_rfe, y_train)

#             # Save the best model
#             save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
#             logging.info("model.pkl file saved in artifacts")
#             joblib.dump(best_model_name, self.model_trainer_config.trained_model_type_path)  # Save model type

#         except Exception as e:
#             logging.info('Exception occurred during model training')
#             raise customexception(e, sys)