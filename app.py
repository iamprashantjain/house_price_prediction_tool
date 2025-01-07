from flask import Flask, render_template, jsonify, request
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
import pandas as pd



#flask object
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form inputs
        bedRoom = float(request.form['bedRoom'])
        bathroom = float(request.form['bathroom'])
        balcony = float(request.form['balcony'])
        floor_nbr = float(request.form['floor_nbr'])
        additional_room_count = float(request.form['additional_room_count'])
        furnishDetails_count = float(request.form['furnishDetails_count'])
        features_count = float(request.form['features_count'])
        Environment = float(request.form['Environment'])
        Safety = float(request.form['Safety'])
        Lifestyle = float(request.form['Lifestyle'])
        Connectivity = float(request.form['Connectivity'])
        Green_Area = float(request.form['Green_Area'])
        Amenities = float(request.form['Amenities'])
        Management = float(request.form['Management'])
        Construction = float(request.form['Construction'])
        nearbyLocations_count = float(request.form['nearbyLocations_count'])
        agePossession_avg = float(request.form['agePossession_avg'])
        total_area_in_sqft = float(request.form['total_area_in_sqft'])
        rate_per_sqft = float(request.form['rate_per_sqft'])
        
        facing = request.form['facing']
        property_type = request.form['property_type']
        prop_location = request.form['prop_location']

        # Create a CustomData object
        custom_data = CustomData(
            bedRoom=bedRoom, bathroom=bathroom, balcony=balcony, floor_nbr=floor_nbr,
            additional_room_count=additional_room_count, furnishDetails_count=furnishDetails_count,
            features_count=features_count, Environment=Environment, Safety=Safety,
            Lifestyle=Lifestyle, Connectivity=Connectivity, Green_Area=Green_Area,
            Amenities=Amenities, Management=Management, Construction=Construction,
            nearbyLocations_count=nearbyLocations_count, agePossession_avg=agePossession_avg,
            total_area_in_sqft=total_area_in_sqft, rate_per_sqft=rate_per_sqft,
            facing=facing, property_type=property_type,
            prop_location=prop_location
        )

        # Convert the input data to DataFrame
        input_df = custom_data.get_data_as_dataframe()

        # Get the prediction
        pipeline = PredictPipeline()
        predictions, ci_lower, ci_upper = pipeline.predict(input_df)

        # Return the predicted price and confidence interval to the template
        return render_template('index.html', predicted_price=predictions[0], ci_lower=ci_lower[0], ci_upper=ci_upper[0])

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)