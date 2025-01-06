from datetime import datetime
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception

import os
import sys
import ast
import re
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataCleaningConfig:
    raw_data_path: str = os.path.join("data", "99acre_raw_data", "raw.xlsx")
    train_data_path: str = os.path.join("data", "99acre_raw_data", "train.xlsx")
    test_data_path: str = os.path.join("data", "99acre_raw_data", "test.xlsx")

class DataCleaning:
    def __init__(self, merged_data_path: str):
        self.merged_data_path = merged_data_path
        self.cleaning_config = DataCleaningConfig()

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning tasks
        """
        # Remove duplicates
        df = data.drop_duplicates()
        logging.info(f"Removed {len(data) - len(df)} duplicate rows.")
        
        #create a copy
        df = df.copy()
        
        # -----------------------
        #fixing missing values
        # -----------------------
                
        # Function to extract floor number from 'floorNum' for flats
        def extract_floor(flat):
            if isinstance(flat, str):
                # Ensure there's a 'of' to split on and extract the floor number before the 'of'
                if 'of' in flat:
                    # Split the string and extract only the numeric part
                    floor_number = ''.join(filter(str.isdigit, flat.split('of')[0]))
                    if floor_number:  # If there's a valid number extracted
                        return int(floor_number)
            return np.nan  # Return NaN if there's no valid floor number

        # Function to extract number of floors from 'noOfFloor' for houses
        def extract_no_of_floor(house):
            if isinstance(house, str):
                # Filter digits to handle cases like "2 Floors" or "NaN"
                numbers = ''.join(filter(str.isdigit, house))
                if numbers:  # If there's a valid number extracted
                    return int(numbers)
            return np.nan  # Return NaN if no valid number found
        
        
        # Apply the functions to create 'floor_nbr'
        df['floor_nbr'] = np.where(df['property_type'] == 'flat', df['floorNum'].apply(extract_floor), df['noOfFloor'].apply(extract_no_of_floor))
        
        
        # Function to extract area and calculate prop_area
        def calculate_prop_area(row):
            # Ensure the 'areaWithType' column is a string to avoid TypeError
            area_with_type = str(row['areaWithType'])

            # Regex to extract Carpet area, Super Built-up area, and Built-up area along with their units
            carpet_area_match = re.search(r'Carpet area[:\s]*(\d+[\.\d]*)\s*\((\d+[\.\d]*)\s*(sq\.m\.|sq\.ft.)\)', area_with_type)
            super_builtup_area_match = re.search(r'Super Built up area[:\s]*(\d+[\.\d]*)\s*\((\d+[\.\d]*)\s*(sq\.m\.|sq\.ft.)\)', area_with_type)
            builtup_area_match = re.search(r'Built Up area[:\s]*(\d+[\.\d]*)\s*\((\d+[\.\d]*)\s*(sq\.m\.|sq\.ft.)\)', area_with_type)

            # Initialize variables to store the values and units
            carpet_area = None
            carpet_unit = None
            super_builtup_area = None
            super_builtup_unit = None
            builtup_area = None
            builtup_unit = None
            
            # Extract Carpet area if present
            if carpet_area_match:
                carpet_area = float(carpet_area_match.group(1))
                carpet_unit = carpet_area_match.group(3)
            
            # Extract Super Built-up area if present
            if super_builtup_area_match:
                super_builtup_area = float(super_builtup_area_match.group(1))
                super_builtup_unit = super_builtup_area_match.group(3)
            
            # Extract Built-up area if present
            if builtup_area_match:
                builtup_area = float(builtup_area_match.group(1))
                builtup_unit = builtup_area_match.group(3)
            
            # Normalize areas to Carpet area, taking units into account
            if carpet_area is not None:
                # If Carpet area is already available, return it with its unit
                return f"{carpet_area} {carpet_unit}"
            elif super_builtup_area is not None:
                # Assuming carpet area is approximately 75% of super built up area
                carpet_area = super_builtup_area * 0.75
                return f"{carpet_area} {super_builtup_unit}"  # Use the same unit as Super Built-up area
            elif builtup_area is not None:
                # Assuming carpet area is approximately 85% of built up area
                carpet_area = builtup_area * 0.85
                return f"{carpet_area} {builtup_unit}"  # Use the same unit as Built-up area
            else:
                return np.nan  # Return NaN if no area is available    
            
            
        # Apply the function to calculate 'prop_area' for flat properties
        df['prop_area'] = df.apply(calculate_prop_area, axis=1)
        
        
        # Function to calculate the number of additional rooms
        def calculate_additional_room_count(row):
            # If the value is NaN, return 0
            if pd.isna(row['additionalRoom']):
                return 0
            # Split by comma and count the items
            else:
                rooms = row['additionalRoom'].split(',')
                return len(rooms)
            
            
        # Apply the function to calculate 'additional_room_count'
        df['additional_room_count'] = df.apply(calculate_additional_room_count, axis=1)
        
        
        #facing null values with undetermined
        df['facing'] = df['facing'].fillna('undetermined')
        
        
        
        # Function to safely evaluate the string as a list, and count the items or replace with 0 if NaN or empty
        def count_or_zero(furnish):
            if pd.isna(furnish) or furnish == "[]":
                return 0
            else:
                try:
                    # Safely evaluate the string to a list
                    furnish_list = ast.literal_eval(furnish)
                    if isinstance(furnish_list, list):  # Ensure it's a list
                        return len(furnish_list)
                    else:
                        return 0
                except (ValueError, SyntaxError):  # In case the string is not a valid list
                    return 0

        # Apply the function to the 'furnishDetails' column
        df['furnishDetails_count'] = df['furnishDetails'].apply(count_or_zero)

        
        
        # Function to safely evaluate the string as a list and count the items or replace with 0 if NaN or empty
        def count_features(features):
            if pd.isna(features) or features == "[]":
                return 0
            else:
                try:
                    # Safely evaluate the string to a list
                    features_list = ast.literal_eval(features)
                    if isinstance(features_list, list):  # Ensure it's a list
                        return len(features_list)
                    else:
                        return 0
                except (ValueError, SyntaxError):  # In case the string is not a valid list
                    return 0

        # Apply the function to the 'features' column to create 'features_count'
        df['features_count'] = df['features'].apply(count_features)
        
        #society
        df.drop(columns=['society'], inplace=True)
        
        
        # Function to extract category and rating using regex
        def extract_ratings(rating_str):
            ratings = {}
            if isinstance(rating_str, str):  # Check if it's a string
                # Convert the string representation of the list to an actual list
                try:
                    rating_list = eval(rating_str)
                except:
                    rating_list = []  # In case of invalid data
                for item in rating_list:
                    if isinstance(item, str):  # Check if the item is a string
                        match = re.match(r'([A-Za-z\s]+)(\d+(\.\d+)?)\s*out\s+of\s+5', item.strip())
                        if match:
                            category = match.group(1).strip()  # Extract category
                            score = float(match.group(2))     # Extract score
                            ratings[category] = score  # Store category and score
            return ratings

        # Apply the function to extract ratings
        df['rating_dict'] = df['rating'].apply(extract_ratings)        
        
        
        rating_df = pd.json_normalize(df['rating_dict'])

        # Concatenate with the original DataFrame
        df = pd.concat([df, rating_df], axis=1)
        
        
        #filter df
        df[['Environment', 'Safety', 'Lifestyle', 'Connectivity', 'Green Area', 'Amenities', 'Management', 'Construction']] = df[['Environment', 'Safety', 'Lifestyle', 'Connectivity', 'Green Area', 'Amenities', 'Management', 'Construction']].fillna(0)
        
        
        #nearby locations
        # Convert the string representation of a list to an actual list
        df['nearbyLocations'] = df['nearbyLocations'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Now count the number of items in the list
        df['nearbyLocations_count'] = df['nearbyLocations'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        
        #price
        df.dropna(subset=['price'], inplace=True)
        
        
        #agepossesion
        def extract_age_average(age_str):
            # First, check if the value is a string. If it's not, return NaN.
            if not isinstance(age_str, str):
                return np.nan

            # Check if the string contains a range of numbers like '5 to 10 Year Old'
            match = re.match(r'(\d+)\s*to\s*(\d+)', age_str)
            
            if match:
                # If it matches the range, calculate the average
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                return (num1 + num2) / 2
            
            # Check if the string contains only a single number with 'Year Old' or 'Year' (e.g., '10+ Year Old')
            elif re.match(r'(\d+)\+?\s*Year(s)?\s*Old', age_str):
                return int(re.match(r'(\d+)\+?\s*Year(s)?\s*Old', age_str).group(1))
            
            # If it's "undefined" or any other value, return NaN or 0 (you can decide how to handle it)
            elif age_str.lower() == "undefined":
                return 0  # You can also return 0 here if you prefer
            
            return np.nan  # If there's no match, return NaN or a default value

        # Apply the function to the 'agePossession' column
        df['agePossession_avg'] = df['agePossession'].apply(extract_age_average)        
        
        
        df.drop(columns=['agePossession'], inplace=True)
        df['agePossession_avg'] = df['agePossession_avg'].fillna(0)
        
        
        #drop floornbr
        df.dropna(subset=['floor_nbr'], inplace=True)
        
        
        #filter df
        df = df[df['price'] != 'Price on Request']
        
        
        #remove extra cols
        df.drop(columns=['floorNum', 'noOfFloor'], inplace=True)
        
        
        # Function to extract area and calculate prop_area
        def calculate_prop_area(row):
            area_with_type = str(row['areaWithType'])

            # Updated regex to capture Plot area and Built Up area with or without colon, and area inside parentheses
            plot_area_match = re.search(r'Plot area[:\s]*(\d+[\.\d]*)\s*\((\d+[\.\d]*)\s*(sq\.m\.|sq\.ft.)\)', area_with_type)
            built_up_area_match = re.search(r'Built Up area[:\s]*(\d+[\.\d]*)\s*\((\d+[\.\d]*)\s*(sq\.m\.|sq\.ft.)\)', area_with_type)

            # Initialize variables to store area values and units
            plot_area = None
            plot_unit = None
            built_up_area = None
            built_up_unit = None

            # Extract Plot area if present
            if plot_area_match:
                plot_area = float(plot_area_match.group(1))  # Area value (e.g., 160)
                plot_unit = plot_area_match.group(3)  # Unit (e.g., sq.m.)

            # Extract Built Up area if present
            if built_up_area_match:
                built_up_area = float(built_up_area_match.group(1))  # Area value (e.g., 340)
                built_up_unit = built_up_area_match.group(3)  # Unit (e.g., sq.m.)

            # If Plot area is found, return the formatted result
            if plot_area is not None:
                return f"{plot_area} {plot_unit}"
            
            # If no Plot area, check for Built Up area and return the formatted result
            if built_up_area is not None:
                return f"{built_up_area} {built_up_unit}"

            # Return NaN if no area is available
            return np.nan
        
        
        # Apply the function to fill 'prop_area' for rows where 'prop_area' is NaN
        df.loc[df['prop_area'].isna(), 'prop_area'] = df.loc[df['prop_area'].isna()].apply(calculate_prop_area, axis=1)
        
        
        #drop na rows
        df.dropna(subset=['prop_area'], inplace=True)
        
        
        # --------------------------------
        # Get the null count for each column
        null_counts = df.isnull().sum().reset_index()

        # Rename the columns for easier access
        null_counts.columns = ['Column', 'Null Count']

        # Filter the columns where Null Count is not 0
        null_counts_with_null = null_counts[null_counts['Null Count'] != 0]
        logging.info(null_counts_with_null)
        # --------------------------------
        
        cols_to_remove = null_counts_with_null['Column'].tolist()
        logging.info(cols_to_remove)
        
        #dropping cols_to_remove
        df.drop(columns=cols_to_remove, inplace=True)
        
        
        
        # -----------------------
        #fixing messy data issues
        # -----------------------
        
        df['prop_location'] = df['property_name'].str.split("in").str[1]
        df.drop(columns=['property_name'], inplace=True)
        
        
        # -----------------------
        #fixing dirty data issues
        # -----------------------
        
        
        # Function to convert price to numeric
        def convert_price_to_number(price):
            price = str(price)  # Ensure it's a string

            # Remove spaces and lower the case for uniformity
            price = price.strip().lower()

            # Check for 'crore' and 'lac' and convert accordingly
            if 'crore' in price:
                price = price.replace('crore', '').strip()
                return float(price) * 1e7  # 1 Crore = 10,000,000
            elif 'lac' in price:
                price = price.replace('lac', '').strip()
                return float(price) * 1e5  # 1 Lac = 100,000
            else:
                return float(price)  # No unit, just return the number
        
        
        df['price'] = df['price'].apply(convert_price_to_number)
        
        
        
        # Function to convert prop_area from sq.m. to sq.ft.
        def convert_area_to_sqft(area):
            # Extract the numeric value and convert to float
            area_value = float(area.split()[0])
            # Convert square meters to square feet
            area_in_sqft = area_value * 10.7639
            # Return the result in square feet as a string with the unit
            return f"{area_in_sqft:,.2f} sq.ft."

        # Apply the conversion to the 'prop_area' column
        df['total_area_in_sqft'] = df['prop_area'].apply(convert_area_to_sqft)
        
        
        #drop prop_area col
        df.drop(columns=['prop_area'], inplace=True)
        
        
        
        # Remove ₹, commas, and '/sq.ft.' from the 'rate' column, then convert to float
        df['rate_per_sqft'] = df['rate'].str.replace('₹', '', regex=False) \
                                        .str.replace('/sq.ft.', '', regex=False) \
                                        .str.replace(',', '', regex=False) \
                                        .astype(float)

        # Remove ' sq.ft.' from the 'total_area_in_sqft' column, then convert to float
        df['total_area_in_sqft'] = df['total_area_in_sqft'].str.replace(' sq.ft.', '', regex=False) \
                                                        .str.replace(',', '', regex=False) \
                                                        .astype(float)

        # Rename the 'rate' column to 'rate_per_sqft'
        df = df.drop(columns=['rate'])  # Drop the old 'rate' column
        
        
        
        #bedroom bathroom balcony
        df['bedRoom'] = df['bedRoom'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        df['bathroom'] = df['bathroom'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        df['balcony'] = df['balcony'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
        
        
        #remove extra cols
        df.drop(columns=['rating_dict', 'link', 'description', 'property_id'], inplace=True)
        
        return df

    
    def initiate_data_cleaning(self):
        logging.info("Data cleaning started")
        try:
            # Read the merged data file
            data = pd.read_csv(self.merged_data_path)
            logging.info(f"Data loaded from {self.merged_data_path} with shape: {data.shape}")

            # Perform the cleaning operations
            cleaned_data = self._clean_data(data)
            
            # Ensure the directory for saving files exists
            os.makedirs(os.path.dirname(self.cleaning_config.raw_data_path), exist_ok=True)

            # Save the cleaned data to the specified path
            cleaned_data.to_excel(self.cleaning_config.raw_data_path, index=False)
            logging.info(f"Raw cleaned data saved to: {self.cleaning_config.raw_data_path}")

            # Perform train-test split (75% train, 25% test)
            logging.info("Train-test split started")
            train_data, test_data = train_test_split(cleaned_data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save the train and test datasets to specified paths
            train_data.to_excel(self.cleaning_config.train_data_path, index=False)
            test_data.to_excel(self.cleaning_config.test_data_path, index=False)
            
            logging.info(f"Train data saved to: {self.cleaning_config.train_data_path}")
            logging.info(f"Test data saved to: {self.cleaning_config.test_data_path}")

            logging.info("Data cleaning completed successfully.")
            
            # Return train & test data paths
            return self.cleaning_config.train_data_path, self.cleaning_config.test_data_path

        except Exception as e:
            logging.error(f"Error during data cleaning: {str(e)}")
            raise customexception(e, sys)