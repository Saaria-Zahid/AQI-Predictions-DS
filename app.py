import streamlit as st
import pandas as pd
from datetime import datetime
from aqi_pipeline import (
    run_full_pipeline,
    generate_forecast,
    train_model,
    build_features
)

st.set_page_config(
    page_title="AQI AI Forecast Dashboard",
    layout="wide"
)

st.title("AI Powered AQI Forecast System")
st.caption("Automated Machine Learning Pipeline with MongoDB Registry")

# Sidebar Controls
st.sidebar.header("Controls")

forecast_hours = st.sidebar.slider(
    "Forecast Hours",
    min_value=12,
    max_value=168,
    value=72,
    step=12
)

if st.sidebar.button("Run Full Pipeline"):
    with st.spinner("Running complete pipeline..."):
        forecast = run_full_pipeline()
    st.success("Pipeline Completed")
    st.dataframe(forecast)

if st.sidebar.button("Train Model Only"):
    with st.spinner("Training model..."):
        train_model()
    st.success("Model Trained and Registered")

if st.sidebar.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        forecast = generate_forecast(forecast_hours)

    if forecast.empty:
        st.error("No forecast generated")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Forecast Data")
            st.dataframe(forecast)

        with col2:
            st.subheader("Forecast Chart")
            chart_df = forecast.set_index("datetime")
            st.line_chart(chart_df)

        st.metric(
            label="Latest Predicted AQI",
            value=round(forecast["predicted_aqi"].iloc[-1], 2)
        )
