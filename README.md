# AQI AI Forecast System

An intelligent Air Quality Index (AQI) forecasting system powered by automated machine learning with MongoDB model registry and real-time Streamlit dashboard.

## Overview

This project fetches historical air pollution data from OpenWeather API, engineers features from raw data, trains machine learning models, and provides AQI forecasts through an interactive web dashboard. The system leverages MongoDB to store feature sets and maintain a model registry for tracking trained models and their performance metrics.

## Features

- **Real-time Data Fetching**: Automated retrieval of historical air pollution data from OpenWeather API
- **Intelligent Feature Engineering**: Creates lag features, rolling statistics, and temporal features
- **Automated Model Training**: Multiple ML algorithms (Random Forest, Gradient Boosting, Linear Regression)
- **Model Registry**: MongoDB-backed registry tracking all trained models, metrics, and versions
- **Interactive Dashboard**: Streamlit-powered web interface for predictions and analysis
- **Forecast Generation**: Generate AQI predictions for 12-168 hours ahead
- **Feature Store**: MongoDB collection for efficient feature caching and retrieval

## Project Structure

```
├── app.py                           # Streamlit dashboard application
├── aqi_pipeline.py                  # Core ML pipeline and functions
├── requirements.txt                 # Python dependencies
├── App/
│   ├── 01_data_fetch.py            # Data fetching script
│   └── 02_feature_engineering.py   # Feature engineering implementation
├── Data/                            # Jupyter Notebooks for development
│   ├── 01_data_fetch.ipynb         # Explore and fetch air quality data
│   ├── 02_feature_engineer.ipynb   # Feature engineering and EDA
│   ├── 03_model_train.ipynb        # Model training and evaluation
│   └── 04_prediction.ipynb         # Generate predictions and forecasts
├── Dataset/
│   └── aqi_data.csv                # Historical air quality data
└── README.md                        # This file
```

## Notebooks Workflow

The project includes four Jupyter notebooks that form the complete ML workflow:

### 1. **01_data_fetch.ipynb** - Data Collection & Exploration
- Fetches historical air pollution data from OpenWeather API (Latitude: 24.8607, Longitude: 67.0011)
- Extracts pollutant measurements: CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
- Performs initial data exploration and validation
- Outputs raw dataset to `Dataset/aqi_data.csv`

### 2. **02_feature_engineer.ipynb** - Feature Engineering & EDA
- Cleans and preprocesses raw air quality data
- Creates temporal features: hour, day of week, month, weekend flag
- Generates lag features (1, 2, 3, 6, 12, 24 hours) for each pollutant
- Computes rolling statistics for trend analysis
- Produces correlation heatmap and feature visualization
- Stores engineered features in MongoDB feature store

### 3. **03_model_train.ipynb** - Model Development & Training
- Splits data into training and testing sets
- Trains multiple models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- Evaluates models using MAE, RMSE, and R² metrics
- Registers best performing model in MongoDB model registry with metadata and version tracking

### 4. **04_prediction.ipynb** - Forecasting & Evaluation
- Loads the trained model from registry
- Generates AQI forecasts for future time periods
- Visualizes predictions vs actual values
- Calculates forecast accuracy metrics
- Provides actionable insights for air quality planning

## Installation

### Prerequisites
- Python 3.8+
- MongoDB Atlas account (for model registry and feature store)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AQI
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
OPENWEATHER_API_KEY=<your-api-key>
DB_NAME=aqi_project
```

## Usage

### Run the Development Pipeline
Execute notebooks in sequence:
1. Start with `Data/01_data_fetch.ipynb` to fetch data
2. Run `Data/02_feature_engineer.ipynb` for feature engineering
3. Execute `Data/03_model_train.ipynb` to train models
4. Use `Data/04_prediction.ipynb` for forecasting

### Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` with the following capabilities:
- **Run Full Pipeline**: Execute complete data → features → training → prediction workflow
- **Train Model Only**: Train and register a new model
- **Generate Forecast**: Create predictions for specified hours (12-168)
- Interactive forecast visualization and metrics

### Programmatic Usage
```python
from aqi_pipeline import (
    fetch_data,
    build_features,
    train_model,
    generate_forecast,
    run_full_pipeline
)

# Fetch data
data = fetch_data()

# Build features
features_df = build_features(data)

# Train model
train_model()

# Generate forecast
forecast = generate_forecast(forecast_hours=72)
```

## Key Components

### aqi_pipeline.py
Core module containing:
- `fetch_data()`: Retrieves air pollution data from OpenWeather API
- `build_features()`: Generates engineered features with lags and rolling statistics
- `train_model()`: Trains and registers ML models
- `load_latest_model()`: Retrieves best model from registry
- `generate_forecast()`: Generates future AQI predictions
- `run_full_pipeline()`: Executes complete workflow

### app.py
Streamlit dashboard providing:
- Pipeline execution controls
- Model training interface
- Real-time forecast generation
- Interactive data visualization
- Performance metrics display

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.0 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.0 | Machine learning models |
| streamlit | 1.31.0 | Web dashboard |
| pymongo | 4.6.1 | MongoDB database |
| requests | 2.32.3 | API calls |
| plotly | 5.18.0 | Interactive visualizations |
| statsmodels | 0.14.1 | Time series analysis |

See `requirements.txt` for complete dependency list.

## Database Schema

### Feature Store Collection (`feature_store`)
Stores processed features with metadata:
- `datetime`: Timestamp
- Pollutant measurements: CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
- Lag features: 1, 2, 3, 6, 12, 24 hour lags
- Rolling statistics: 6, 12, 24 hour windows
- Temporal features: hour, day_of_week, month, is_weekend

### Model Registry Collection (`model_registry`)
Tracks all trained models:
- `model_name`: Model identifier
- `model_binary`: Serialized model
- `features`: List of features used
- `metrics`: MAE, RMSE, R² scores
- `trained_at`: Training timestamp
- `version`: Model version

## API Configuration

**OpenWeather API**
- Endpoint: `http://api.openweathermap.org/data/2.5/air_pollution/history`
- Location: Karachi, Pakistan (24.8607°N, 67.0011°E)
- Supports historical data queries and forecasts

## Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (Coefficient of Determination): Proportion of variance explained

## Future Enhancements

- [ ] Add ensemble model combining multiple algorithms
- [ ] Implement ARIMA/Prophet for time series forecasting
- [ ] Add real-time data streaming
- [ ] Deploy as cloud API (AWS/GCP/Azure)
- [ ] Add mobile application
- [ ] Implement automated retraining pipeline
- [ ] Add weather integration (temperature, humidity impact)
- [ ] Create alert system for unhealthy AQI levels

## Troubleshooting

**MongoDB Connection Issues**
- Verify MongoDB URI in `.env` file
- Check network connectivity to MongoDB Atlas
- Ensure IP whitelist includes your current IP

**API Rate Limiting**
- OpenWeather API has rate limits; monitor requests
- Add `time.sleep(1)` between API calls in `fetch_data()`

**Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

## License

This project is open source and available under the MIT License.

## Contact

For questions or contributions, please open an issue or contact the project maintainers.
