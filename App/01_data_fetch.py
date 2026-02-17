"""AQI pipeline module.

Provides functions:
 - fetch_data()
 - build_features()
 - train_model()
 - load_latest_model()
 - generate_forecast()
 - run_full_pipeline()

This preserves existing behavior: feature store is `feature_store` collection,
model registry is `model_registry` collection and models are stored with
fields `model_name`, `model_binary`, `features`, `metrics`, `trained_at`, `version`.
"""

from typing import Tuple
import time
import requests
import pandas as pd
import numpy as np
import pickle
import joblib
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# Configuration (keep values as in notebooks)
OPENWEATHER_API_KEY = "aa6877ac64bbbc776b89c98b61b11b54"
LAT = 24.8607
LON = 67.0011
FORECAST_HOURS = 72

MONGO_URI = 'mongodb+srv://saz_db1328:wUlewP7Ijtr80RqC@cluster0.j3swhkq.mongodb.net/'
DB_NAME = "aqi_project"

POLLUTANTS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]
DEFAULT_CSV = "../Dataset/aqi_data.csv"

# Mongo clients
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
feature_store = db["feature_store"]
model_registry = db["model_registry"]


def fetch_data(start_date: datetime = None, end_date: datetime = None, out_csv: str = DEFAULT_CSV) -> pd.DataFrame:
    """Fetch historical air pollution data from OpenWeather and save CSV.

    Defaults to last 1 year when dates not provided.
    """
    if end_date is None:
        end_date = pd.to_datetime('today').normalize()
    if start_date is None:
        start_date = end_date - pd.DateOffset(years=1)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    chunk_days = 30
    chunk_seconds = chunk_days * 24 * 60 * 60

    all_data = []
    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        params = {
            'lat': LAT,
            'lon': LON,
            'start': current_start,
            'end': current_end,
            'appid': OPENWEATHER_API_KEY
        }
        res = requests.get("http://api.openweathermap.org/data/2.5/air_pollution/history", params=params)
        res.raise_for_status()
        data = res.json()
        if 'list' in data:
            all_data.extend(data['list'])
        print(f"Fetched {len(data.get('list', []))} records from {current_start} to {current_end}")
        current_start = current_end + 1
        time.sleep(1)

    raw_df = pd.DataFrame(all_data)
    raw_df['datetime'] = pd.to_datetime(raw_df['dt'], unit='s')
    df_main = raw_df['main'].apply(pd.Series)
    df_components = raw_df['components'].apply(pd.Series)
    df = pd.concat([raw_df.drop(['main', 'components'], axis=1), df_main, df_components], axis=1)
    df = df.sort_values('datetime').reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved historical data to {out_csv}")
    return df


def recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time, lag, rolling and diff features used by model (same as notebooks)."""
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    vars_for_lags = ['aqi'] + POLLUTANTS
    for lag in LAGS:
        for p in vars_for_lags:
            df[f"{p}_lag_{lag}"] = df[p].shift(lag)

    for w in ROLLING_WINDOWS:
        for p in vars_for_lags:
            df[f"{p}_roll_mean_{w}"] = df[p].shift(1).rolling(w).mean()
            df[f"{p}_roll_std_{w}"] = df[p].shift(1).rolling(w).std()

    for p in vars_for_lags:
        df[f"{p}_diff_1"] = df[p].shift(1) - df[p].shift(2)
        df[f"{p}_diff_3"] = df[p].shift(1) - df[p].shift(4)

    return df


def build_features(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Read raw CSV, build features and push feature set to MongoDB feature_store.

    This mirrors the notebook behavior: trims initial rows with insufficient history
    (MAX_LAG = 24), creates `aqi_target`, drops NA, and writes to `feature_store`.
    """
    df = pd.read_csv(csv_path)
    if 'dt' in df.columns:
        df = df.drop(columns=['dt'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    # compute basic time features and extended features
    df = recompute_features(df)

    # Trim initial rows where lags/rollings are undefined - same as notebook MAX_LAG=24
    MAX_LAG = 24
    df_features = df.iloc[MAX_LAG:].reset_index(drop=True)
    df_features['aqi_target'] = df_features['aqi'].shift(-1)
    df_features = df_features.dropna().reset_index(drop=True)

    # persist to MongoDB feature_store
    df_mongo = df_features.copy()
    df_mongo['datetime'] = df_mongo['datetime'].astype(str)
    feature_store.delete_many({})
    feature_store.insert_many(df_mongo.to_dict('records'))
    print('Feature store saved to MongoDB')
    return df_features


def train_model() -> Tuple[object, list]:
    """Train candidate models on the feature_store and register best model.

    Returns (best_model, FEATURES).
    """
    cursor = feature_store.find({})
    data = list(cursor)
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    TARGET = 'aqi_target'
    FEATURES = [
    c for c in df.columns
    if c not in ['aqi', 'aqi_target', 'datetime']
]


    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=18, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        metrics = {
            'mae': float(mean_absolute_error(y_test, p)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, p))),
            'r2': float(r2_score(y_test, p))
        }
        results.append((name, model, metrics))
        print(name, metrics)

    best_name, best_model, best_metrics = sorted(results, key=lambda x: x[2]['rmse'])[0]
    joblib.dump(best_model, 'aqi_best_model.pkl')

    model_registry.insert_one({
        'model_name': best_name,
        'model_binary': pickle.dumps(best_model),
        'features': FEATURES,
        'metrics': best_metrics,
        'trained_at': datetime.utcnow(),
        'version': 'v1'
    })
    print('Model registered in MongoDB:', best_name)
    return best_model, FEATURES


def load_latest_model() -> Tuple[object, list]:
    """Load the latest model and its feature list from the model_registry."""
    model_doc = model_registry.find_one(sort=[('trained_at', -1)])
    if model_doc is None:
        raise RuntimeError('No model found in model_registry')
    model = pickle.loads(model_doc['model_binary'])
    FEATURES = model_doc['features']
    return model, FEATURES


def generate_forecast(hours: int = FORECAST_HOURS) -> pd.DataFrame:

    model, FEATURES = load_latest_model()

    history = pd.DataFrame(
        list(feature_store.find({}).sort('datetime', -1).limit(150))
    )

    if history.empty:
        raise RuntimeError("Feature store is empty")

    if '_id' in history.columns:
        history = history.drop(columns=['_id'])

    history['datetime'] = pd.to_datetime(history['datetime'])
    history = history.sort_values('datetime').reset_index(drop=True)

    history = recompute_features(history)
    history = history.dropna().reset_index(drop=True)

    if history.empty:
        raise RuntimeError("Insufficient historical data after feature recompute")

    predictions = []
    current_time = history['datetime'].iloc[-1]

    for step in range(hours):

        current_time += timedelta(hours=1)

        new_row = {'datetime': current_time, 'aqi': np.nan}

        for p in POLLUTANTS:
            new_row[p] = history[p].iloc[-1]

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history = recompute_features(history)

        X = history.loc[[history.index[-1]], FEATURES]

        if X.isna().any().any():
            raise RuntimeError(f"Feature NaN detected at step {step}")

        pred = model.predict(X)[0]
        history.loc[history.index[-1], 'aqi'] = pred

        predictions.append({
            'datetime': current_time,
            'predicted_aqi': float(pred)
        })

    return pd.DataFrame(predictions)


def run_full_pipeline() -> pd.DataFrame:
    """Run fetch -> build_features -> train -> forecast and return forecast DataFrame."""
    end_date = pd.to_datetime('today').normalize()
    start_date = end_date - pd.DateOffset(years=1)
    fetch_data(start_date=start_date, end_date=end_date)
    build_features()
    train_model()
    return generate_forecast()


if __name__ == '__main__':
    # simple CLI for convenience
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fetch_data', 'build_features', 'train_model', 'generate_forecast', 'run_full_pipeline'])
    args = parser.parse_args()

    if args.mode == 'fetch_data':
        fetch_data()
    elif args.mode == 'build_features':
        build_features()
    elif args.mode == 'train_model':
        train_model()
    elif args.mode == 'generate_forecast':
        print(generate_forecast().head())
    elif args.mode == 'run_full_pipeline':
        print(run_full_pipeline().head())
