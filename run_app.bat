@echo off
echo ===============================================
echo      🌫️ AQI Prediction Dashboard
echo ===============================================
echo.
echo Starting the AQI Prediction Dashboard...
echo This will open in your default web browser.
echo.
echo Features:
echo   ✅ Live data from OpenWeatherMap API
echo   📊 72-hour AQI predictions  
echo   📈 Interactive charts and heatmaps
echo   💾 Export predictions to CSV
echo.
echo Make sure you have installed the requirements:
echo   pip install -r requirements.txt
echo.
echo ===============================================
streamlit run app.py --server.headless true