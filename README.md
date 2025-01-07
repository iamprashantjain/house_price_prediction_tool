# House Price Prediction Machine Learning Project

![alt text](image.png)

check detailed information about this project and tools used in the file: `mlops_intro.ipynb`

This repository contains the implementation of a **House Price Prediction** machine learning model using data 99Acres from Kaggle, a leading real estate platform. The project encompasses the entire Machine Learning lifecycle, from cleaning to model deployment and monitoring, utilizing state-of-the-art MLOps tools for version control, pipeline orchestration, and model management.

## Project Overview

The goal of this project is to build a machine learning model that predicts house prices based on various property features such as location, size, amenities, etc. We scraped data from MagicBricks, cleaned and preprocessed it, and built a machine learning model to predict the price of a house given the input features.

## Key Components


1. **Data Cleaning & Preprocessing**:
   - Missing values were handled, irrelevant columns were removed, and outliers were detected and treated.
   - Data types were corrected, and feature scaling and normalization were applied where necessary.

2. **Exploratory Data Analysis (EDA)**:
   - Performed detailed exploratory data analysis (EDA) to understand the underlying patterns in the data and identify key features that influence house prices.
   - Various visualizations, such as scatter plots, histograms, and heatmaps, were used to analyze relationships between features and target variables.

3. **Feature Engineering**:
   - Created new features based on existing data (e.g., extracted the floor number, building age, proximity to key locations, etc.).
   - Encoded categorical features and handled date/time variables.

4. **Model Building**:
   - Applied several machine learning models, including Linear Regression, Random Forest, and Gradient Boosting, to predict house prices.
   - Hyperparameter tuning was performed using cross-validation techniques to optimize model performance.

5. **Model Evaluation**:
   - Evaluated models based on various metrics, including **RMSE**, **MAE**, and **R-squared**.
   - Used performance plots to compare the models and determine the best-performing one.

6. **Model Deployment with MLOps**:
   - Deployed the trained model using **MLFlow** for model management and versioning.
   - Used **Docker** to containerize the application for easy deployment and scaling.
   - **Airflow** was used for orchestrating the pipeline, automating tasks such as data scraping, model training, and evaluation.
   
7. **Data Version Control (DVC)**:
   - Utilized **DVC** for version control of large datasets and model files, ensuring reproducibility and traceability of experiments.

8. **MLOps Tools**:
   - Implemented a complete MLOps pipeline for continuous integration and deployment of machine learning models.
   - Managed and tracked experiments, data, and models with **Git**, **DVC**, **MLFlow**, and **Docker**.

## Technologies Used

- **Data Collection**: Python, Requests, BeautifulSoup
- **Data Processing**: Pandas, Numpy, Scikit-learn
- **Machine Learning Models**: Scikit-learn, XGBoost, LightGBM
- **Model Evaluation**: Matplotlib, Seaborn
- **MLOps Tools**: Git, DVC, MLFlow, Docker, Airflow
- **Deployment**: Docker, Flask (for API)
  
## How to Run the Project

### Prerequisites:
- Python 3.x
- Docker
- Git
- DVC
- MLFlow
- Apache Airflow

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DVC and pull data:
   ```bash
   dvc pull
   ```

4. Train and evaluate the model:
   ```bash
   python train_model.py
   ```

5. Run the API to make predictions:
   ```bash
   python app.py
   ```

### Docker Setup:
You can also run the entire project in a Docker container:
```bash
docker build -t house-price-prediction .
docker run -p 5000:5000 house-price-prediction
```

## Future Enhancements

- Integrate more external data sources for better prediction accuracy (e.g., economic factors, neighborhood statistics).
- Implement model monitoring to track real-time performance and trigger retraining when necessary.
- Expand the deployment pipeline to support cloud-based deployment platforms (e.g., AWS, GCP, Azure).

## Contributing

Feel free to fork the repository and submit pull requests. Contributions to improving the model, adding more features, or enhancing the MLOps pipeline are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Deployment

1. AWS ElasticBeanstalk: We can deploy this project on AWS ElasticBeanStalk, create a Continous Deployment pipeline and connect our github. This way we can easily deploy the project for public use. If we make any changes to github, It will deploy the project with new updates.

2. AWS ECR + EC2: As soon as we push the code update on github, We can setup "continous Integration" pipeline which will perform unit & integrate test, Creates private docker image (ECR repo) just like dockerhub for aws "continous delivery" and deploy the image on AWS EC2 "continous deployment".

3. Azure Webapps: Same as above where we used aws ec2, we can deploy our docker image in azure container registry and deploy it to azure webapps. CI, CD, CD