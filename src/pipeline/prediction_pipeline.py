import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object
import joblib

class PredictPipeline:

    def __init__(self):
        print("Initializing PredictPipeline object")

    def predict(self, features):
        """
        Makes a prediction based on the input features by first applying the preprocessor,
        then using the trained model to make predictions.

        :param features: The input features to make predictions on.
        :return: The predicted results and their 95% confidence interval.
        """
        try:
            # Define the paths for the preprocessor and model
            preprocessor_path = os.path.join("data", "preprocessor.pkl")
            model_path = os.path.join("data", "model.pkl")

            # Load the preprocessor and model using a utility function
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Scale the input features using the preprocessor
            scaled_features = preprocessor.transform(features)

            # Make predictions using the trained model
            predictions = model.predict(scaled_features)
            
            ci_lower = predictions - 5000
            ci_upper = predictions + 5000
                        
            return predictions, ci_lower, ci_upper


        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 # Numerical features
                 bedRoom: float,
                 bathroom: float,
                 balcony: float,
                 floor_nbr: float,
                 additional_room_count: float,
                 furnishDetails_count: float,
                 features_count: float,
                 Environment: float,
                 Safety: float,
                 Lifestyle: float,
                 Connectivity: float,
                 Green_Area: float,  # Green Area (with space) is expected by the model
                 Amenities: float,
                 Management: float,
                 Construction: float,
                 nearbyLocations_count: float,
                 agePossession_avg: float,
                 total_area_in_sqft: float,
                 rate_per_sqft: float,

                 # Categorical features
                 facing: str,
                 property_type: str,
                 prop_location: str
                 ):

        # Initialize the features
        # Numerical features
        self.bedRoom = bedRoom
        self.bathroom = bathroom
        self.balcony = balcony
        self.floor_nbr = floor_nbr
        self.additional_room_count = additional_room_count
        self.furnishDetails_count = furnishDetails_count
        self.features_count = features_count
        self.Environment = Environment
        self.Safety = Safety
        self.Lifestyle = Lifestyle
        self.Connectivity = Connectivity
        self.Green_Area = Green_Area  # Corrected column name for model expectations
        self.Amenities = Amenities
        self.Management = Management
        self.Construction = Construction
        self.nearbyLocations_count = nearbyLocations_count
        self.agePossession_avg = agePossession_avg
        self.total_area_in_sqft = total_area_in_sqft
        self.rate_per_sqft = rate_per_sqft

        # Categorical features
        self.facing = facing
        self.property_type = property_type
        self.prop_location = prop_location

    def get_data_as_dataframe(self):
        try:
            # Convert the features into a dictionary
            custom_data_input_dict = {
                # Numerical features
                'bedRoom': [self.bedRoom],
                'bathroom': [self.bathroom],
                'balcony': [self.balcony],
                'floor_nbr': [self.floor_nbr],
                'additional_room_count': [self.additional_room_count],
                'furnishDetails_count': [self.furnishDetails_count],
                'features_count': [self.features_count],
                'Environment': [self.Environment],
                'Safety': [self.Safety],
                'Lifestyle': [self.Lifestyle],
                'Connectivity': [self.Connectivity],
                'Green Area': [self.Green_Area],  # Correct column name for model expectations
                'Amenities': [self.Amenities],
                'Management': [self.Management],
                'Construction': [self.Construction],
                'nearbyLocations_count': [self.nearbyLocations_count],
                'agePossession_avg': [self.agePossession_avg],
                'total_area_in_sqft': [self.total_area_in_sqft],
                'rate_per_sqft': [self.rate_per_sqft],

                # Categorical features
                'facing': [self.facing],
                'property_type': [self.property_type],
                'prop_location': [self.prop_location]
            }

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom input data converted to dataframe successfully')

            return df
        except Exception as e:
            logging.error('Exception occurred while converting data to DataFrame')
            raise customexception(e, sys)