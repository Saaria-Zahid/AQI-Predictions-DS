import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import requests
from datetime import datetime, timedelta
import time

# OpenWeatherMap API configuration
API_KEY = "aa6877ac64bbbc776b89c98b61b11b54"
LAT = 24.8607
LON = 67.0011

# Page configuration
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MongoDB configuration
@st.cache_resource
def init_mongo():
    MONGO_URI = 'mongodb+srv://saz_db1328:wUlewP7Ijtr80RqC@cluster0.j3swhkq.mongodb.net/'
    client = MongoClient(MONGO_URI)
    db = client["aqi_project"]
    return db

# Load model from MongoDB
@st.cache_resource
def load_model():
    db = init_mongo()
    model_collection = db["model_registry"]
    model_doc = model_collection.find_one({"model_name": "GradientBoostingRegressor"})
    if model_doc:
        model = pickle.loads(model_doc["model_object"])
        features = model_doc["features"]
        return model, features
    else:
        st.error("Model not found in database!")
        return None, None

# Fetch current AQI data from OpenWeatherMap API
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_current_aqi():
    """Fetch current AQI data from OpenWeatherMap API"""
    try:
        params = {
            'lat': LAT,
            'lon': LON,
            'appid': API_KEY
        }
        
        response = requests.get("http://api.openweathermap.org/data/2.5/air_pollution", params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'list' in data and len(data['list']) > 0:
            current_data = data['list'][0]
            
            # Extract components
            components = current_data['components']
            main_aqi = current_data['main']['aqi']
            dt = current_data['dt']
            
            # Create DataFrame with current data
            current_df = pd.DataFrame([{
                'datetime': pd.to_datetime(dt, unit='s'),
                'aqi': main_aqi,
                'co': components.get('co', 0),
                'no': components.get('no', 0),
                'no2': components.get('no2', 0),
                'o3': components.get('o3', 0),
                'so2': components.get('so2', 0),
                'pm2_5': components.get('pm2_5', 0),
                'pm10': components.get('pm10', 0),
                'nh3': components.get('nh3', 0)
            }])
            
            return current_df
        else:
            st.error("No current data available from API")
            return None
            
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error fetching current data: {e}")
        return None

# Fetch historical data from MongoDB
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data():
    """Fetch historical data from MongoDB for feature calculation"""
    db = init_mongo()
    collection = db["feature_store"]
    
    # Get latest 100 hours of data for robust feature calculation
    cursor = collection.find({}).sort("datetime", -1).limit(100)
    data = list(cursor)
    
    if data:
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        return df.sort_values('datetime')
    else:
        st.warning("No historical data found in feature store!")
        return None

# Combine current and historical data with feature engineering
def combine_and_engineer_features(historical_df, current_df):
    """Combine historical and current data, then engineer features"""
    if historical_df is None or current_df is None:
        return None
    
    # Combine historical and current data
    combined_df = pd.concat([historical_df, current_df], ignore_index=True)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # Remove duplicates based on datetime
    combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last').reset_index(drop=True)
    
    # Engineer features similar to the training pipeline
    # Time features
    combined_df['hour'] = combined_df['datetime'].dt.hour
    combined_df['dayofweek'] = combined_df['datetime'].dt.dayofweek
    combined_df['month'] = combined_df['datetime'].dt.month
    combined_df['is_weekend'] = combined_df['dayofweek'].isin([5,6]).astype(int)
    
    # Lag features
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for lag in [1,2,3,6,12,24]:
        combined_df[f'aqi_lag_{lag}'] = combined_df['aqi'].shift(lag)
        for p in pollutants:
            combined_df[f'{p}_lag_{lag}'] = combined_df[p].shift(lag)
    
    # Rolling features
    windows = [6, 12, 24]
    for w in windows:
        combined_df[f"aqi_roll_mean_{w}"] = combined_df["aqi"].rolling(w, min_periods=1).mean()
        combined_df[f"aqi_roll_std_{w}"] = combined_df["aqi"].rolling(w, min_periods=1).std()
        
        for p in pollutants:
            combined_df[f"{p}_roll_mean_{w}"] = combined_df[p].rolling(w, min_periods=1).mean()
            combined_df[f"{p}_roll_std_{w}"] = combined_df[p].rolling(w, min_periods=1).std()
    
    # Difference features
    for p in pollutants:
        combined_df[f"{p}_diff_1"] = combined_df[p].diff(1)
        combined_df[f"{p}_diff_3"] = combined_df[p].diff(3)
    
    combined_df["aqi_diff_1"] = combined_df["aqi"].diff(1)
    combined_df["aqi_diff_3"] = combined_df["aqi"].diff(3)
    
    # Fill NaN values for the most recent row (current data)
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    
    return combined_df

# Feature engineering for new data
def create_features_for_prediction(df, target_datetime):
    """
    Create features for a specific datetime based on historical data
    """
    # Time-based features
    hour = target_datetime.hour
    dayofweek = target_datetime.weekday()
    month = target_datetime.month
    is_weekend = 1 if dayofweek >= 5 else 0
    
    # Get the most recent values as base
    latest_row = df.iloc[-1].copy()
    
    # Create a new row for prediction
    new_row = latest_row.copy()
    new_row['datetime'] = target_datetime
    new_row['hour'] = hour
    new_row['dayofweek'] = dayofweek
    new_row['month'] = month
    new_row['is_weekend'] = is_weekend
    
    return new_row

# Predict next 72 hours
def predict_next_72_hours(model, features, df):
    """
    Predict AQI for the next 72 hours using iterative prediction
    """
    predictions = []
    prediction_times = []
    current_df = df.copy()
    
    # Start predicting from the next hour after latest data
    last_datetime = current_df['datetime'].max()
    
    for i in range(72):  # 72 hours = 3 days
        # Calculate next datetime
        next_datetime = last_datetime + timedelta(hours=i+1)
        
        # Create features for this datetime
        new_row = create_features_for_prediction(current_df, next_datetime)
        
        # Prepare features for prediction (use DataFrame to preserve feature names)
        X_pred = pd.DataFrame([new_row[features]], columns=features)
        
        # Make prediction
        pred = model.predict(X_pred)[0]
        
        # Store prediction
        predictions.append(pred)
        prediction_times.append(next_datetime)
        
        # Update the dataframe with the new prediction for next iteration
        new_row['aqi'] = pred
        
        # Add to dataframe for next prediction (this creates lag features)
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Recalculate lag and rolling features for the extended dataset
        current_df = current_df.sort_values('datetime').reset_index(drop=True)
        
        # Update lag features (only for the new row)
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        for lag in [1,2,3,6,12,24]:
            if len(current_df) > lag:
                current_df.iloc[-1, current_df.columns.get_loc(f'aqi_lag_{lag}')] = current_df.iloc[-(lag+1)]['aqi']
                for p in pollutants:
                    if f'{p}_lag_{lag}' in current_df.columns:
                        current_df.iloc[-1, current_df.columns.get_loc(f'{p}_lag_{lag}')] = current_df.iloc[-(lag+1)][p]
        
        # Update rolling features
        windows = [6, 12, 24]
        for w in windows:
            if len(current_df) >= w:
                current_df.iloc[-1, current_df.columns.get_loc(f'aqi_roll_mean_{w}')] = current_df.iloc[-w:]['aqi'].mean()
                current_df.iloc[-1, current_df.columns.get_loc(f'aqi_roll_std_{w}')] = current_df.iloc[-w:]['aqi'].std()
    
    return pd.DataFrame({
        'datetime': prediction_times,
        'predicted_aqi': predictions
    })

# AQI interpretation
def interpret_aqi(aqi_value):
    if aqi_value <= 1:
        return "Good", "#00E400", "Air quality is considered satisfactory"
    elif aqi_value <= 2:
        return "Fair", "#FFFF00", "Air quality is acceptable for most people"
    elif aqi_value <= 3:
        return "Moderate", "#FF7E00", "Unhealthy for sensitive groups"
    elif aqi_value <= 4:
        return "Poor", "#FF0000", "Unhealthy for everyone"
    else:
        return "Very Poor", "#8F3F97", "Health alert: everyone may experience more serious health effects"

# Main Streamlit app
def main():
    st.title("🌫️ AQI Prediction Dashboard")
    st.markdown("### 72-Hour Air Quality Index Forecast for Karachi, Pakistan")
    
    # Sidebar
    st.sidebar.header("🔧 Controls")
    
    # Load model and data
    with st.spinner("Loading model and fetching current data..."):
        model, features = load_model()
        
        # Fetch current AQI from API
        current_df = fetch_current_aqi()
        
        # Fetch historical data from MongoDB
        historical_df = fetch_historical_data()
        
        # Combine and engineer features
        df = combine_and_engineer_features(historical_df, current_df)
    
    if model is None or df is None:
        st.error("Failed to load model or data. Please check your MongoDB connection.")
        return
    
    # Show last update time
    if current_df is not None:
        current_time = current_df['datetime'].iloc[-1]
        st.sidebar.success(f"Current data: {current_time.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.info("✅ Live data from OpenWeatherMap API")
    else:
        st.sidebar.warning("⚠️ Using historical data only")
    
    # Refresh buttons
    col1_refresh, col2_refresh = st.sidebar.columns(2)
    with col1_refresh:
        if st.button("🔄 Refresh API"):
            st.cache_data.clear()
            st.rerun()
    
    with col2_refresh:
        if st.button("📊 Refresh DB"):
            st.cache_data.clear()
            st.rerun()
    
    # Make predictions
    with st.spinner("Generating 72-hour predictions..."):
        predictions_df = predict_next_72_hours(model, features, df)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Current AQI
    current_aqi = df['aqi'].iloc[-1]
    current_status, current_color, current_desc = interpret_aqi(current_aqi)
    
    with col1:
        st.metric(
            label="Current AQI", 
            value=f"{current_aqi:.2f}",
            help=current_desc
        )
        st.markdown(f"<div style='background-color:{current_color};padding:10px;border-radius:5px;text-align:center;color:white'><b>{current_status}</b></div>", 
                    unsafe_allow_html=True)
    
    # Next hour prediction
    next_hour_aqi = predictions_df['predicted_aqi'].iloc[0]
    next_status, next_color, next_desc = interpret_aqi(next_hour_aqi)
    
    with col2:
        st.metric(
            label="Next Hour AQI", 
            value=f"{next_hour_aqi:.2f}",
            delta=f"{next_hour_aqi - current_aqi:.2f}",
            help=next_desc
        )
        st.markdown(f"<div style='background-color:{next_color};padding:10px;border-radius:5px;text-align:center;color:white'><b>{next_status}</b></div>", 
                    unsafe_allow_html=True)
    
    # 24 hour average
    day1_avg = predictions_df['predicted_aqi'].iloc[:24].mean()
    day1_status, day1_color, _ = interpret_aqi(day1_avg)
    
    with col3:
        st.metric(
            label="24h Average AQI", 
            value=f"{day1_avg:.2f}",
            help="Average AQI for next 24 hours"
        )
        st.markdown(f"<div style='background-color:{day1_color};padding:10px;border-radius:5px;text-align:center;color:white'><b>{day1_status}</b></div>", 
                    unsafe_allow_html=True)
    
    # 72 hour average
    full_avg = predictions_df['predicted_aqi'].mean()
    full_status, full_color, _ = interpret_aqi(full_avg)
    
    with col4:
        st.metric(
            label="72h Average AQI", 
            value=f"{full_avg:.2f}",
            help="Average AQI for next 72 hours"
        )
        st.markdown(f"<div style='background-color:{full_color};padding:10px;border-radius:5px;text-align:center;color:white'><b>{full_status}</b></div>", 
                    unsafe_allow_html=True)
    
    # Time series plot
    st.subheader("📈 AQI Forecast Timeline")
    
    # Combine historical and predicted data
    historical_last_24h = df.tail(24)[['datetime', 'aqi']].copy()
    historical_last_24h['type'] = 'Historical'
    
    predicted_data = predictions_df.copy()
    predicted_data['aqi'] = predicted_data['predicted_aqi']
    predicted_data['type'] = 'Predicted'
    
    combined_data = pd.concat([
        historical_last_24h[['datetime', 'aqi', 'type']], 
        predicted_data[['datetime', 'aqi', 'type']]
    ]).reset_index(drop=True)
    
    # Create interactive plot
    fig = px.line(
        combined_data, 
        x='datetime', 
        y='aqi', 
        color='type',
        title="AQI Trend: Last 24 Hours (Historical) + Next 72 Hours (Predicted)",
        labels={'aqi': 'AQI Value', 'datetime': 'Date & Time'}
    )
    
    # Add AQI level bands
    fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1, annotation_text="Good")
    fig.add_hrect(y0=1, y1=2, fillcolor="yellow", opacity=0.1, annotation_text="Fair")
    fig.add_hrect(y0=2, y1=3, fillcolor="orange", opacity=0.1, annotation_text="Moderate")
    fig.add_hrect(y0=3, y1=4, fillcolor="red", opacity=0.1, annotation_text="Poor")
    fig.add_hrect(y0=4, y1=5, fillcolor="purple", opacity=0.1, annotation_text="Very Poor")
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')
    
    # Daily breakdown
    st.subheader("📅 Daily Breakdown")
    
    # Split predictions by day
    predictions_df['date'] = predictions_df['datetime'].dt.date
    daily_stats = predictions_df.groupby('date').agg({
        'predicted_aqi': ['mean', 'min', 'max', 'std']
    }).round(3)
    daily_stats.columns = ['Average AQI', 'Min AQI', 'Max AQI', 'Std Dev']
    
    # Add status for each day
    daily_stats['Status'] = daily_stats['Average AQI'].apply(lambda x: interpret_aqi(x)[0])
    
    st.dataframe(daily_stats, width='stretch')
    
    # Hourly heatmap
    st.subheader("🔥 Hourly AQI Heatmap")
    
    # Create hourly heatmap data
    predictions_df['hour'] = predictions_df['datetime'].dt.hour
    predictions_df['day'] = predictions_df['datetime'].dt.strftime('%a %m/%d')
    
    heatmap_data = predictions_df.pivot(index='day', columns='hour', values='predicted_aqi')
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Predicted AQI by Hour and Day",
        labels={'color': 'AQI Value'},
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, width='stretch')
    
    # Model info in sidebar
    st.sidebar.subheader("📊 Model Information")
    st.sidebar.info(f"""
    **Model**: Gradient Boosting Regressor
    **Features Used**: {len(features)}
    **R² Score**: 0.9986
    **RMSE**: 0.0346
    **MAE**: 0.0063
    """)
    
    # Download predictions
    st.sidebar.subheader("💾 Export Data")
    csv = predictions_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )
    
    # Footer
    st.markdown("---")
    st.markdown("*Data source: OpenWeatherMap API | Location: Karachi, Pakistan | Model: Gradient Boosting Regressor*")

if __name__ == "__main__":
    main()