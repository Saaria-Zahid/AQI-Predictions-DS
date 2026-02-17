# Generated from: 02_feature_engineer.ipynb
# Converted at: 2026-02-16T21:35:11.519Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
from pymongo import MongoClient
from datetime import datetime


df = pd.read_csv("../Dataset/aqi_data.csv")
df.head()

df.drop(columns=['dt'], inplace=True)
df.info()

import seaborn as sns
import matplotlib.pyplot as plt

num_df = df.select_dtypes(include=['number'])  # Keep only numeric columns

# Compute correlation matrix
corr_matrix = num_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Time features
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)


df.isnull().sum()

df.head()

# Lag Feature

pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
for lag in [1,2,3,6,12,24]:
    df[f'aqi_lag_{lag}'] = df['aqi'].shift(lag)
    for p in pollutants:
        df[f'{p}_lag_{lag}'] = df[p].shift(lag)

df.head()        

windows = [6, 12, 24]

for w in windows:
    df[f"aqi_roll_mean_{w}"] = df["aqi"].rolling(w).mean()
    df[f"aqi_roll_std_{w}"] = df["aqi"].rolling(w).std()

    for p in pollutants:
        df[f"{p}_roll_mean_{w}"] = df[p].rolling(w).mean()
        df[f"{p}_roll_std_{w}"] = df[p].rolling(w).std()


df.head()

# Differencing

for p in pollutants:
    df[f"{p}_diff_1"] = df[p].diff(1)
    df[f"{p}_diff_3"] = df[p].diff(3)

df["aqi_diff_1"] = df["aqi"].diff(1)
df["aqi_diff_3"] = df["aqi"].diff(3)


MAX_LAG = 24
df_features = df.iloc[MAX_LAG:].reset_index(drop=True)

df_features.head()

df_features.isnull().sum().sum()

# Target for next-hour prediction
df_features["aqi_target"] = df_features["aqi"].shift(-1)

# Drop last row (no future target)
df_features = df_features.dropna().reset_index(drop=True)

from pymongo import MongoClient

MONGO_URI = 'mongodb+srv://saz_db1328:wUlewP7Ijtr80RqC@cluster0.j3swhkq.mongodb.net/'
client = MongoClient(MONGO_URI)

db = client["aqi_project"]
collection = db["feature_store"]


df_mongo = df_features.copy()
df_mongo["datetime"] = df_mongo["datetime"].astype(str)
collection.delete_many({})  # clean old data
collection.insert_many(df_mongo.to_dict("records"))

print("Feature store saved to MongoDB")