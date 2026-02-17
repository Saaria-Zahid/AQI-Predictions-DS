"""
Configuration and setup script for AQI Prediction Dashboard
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import plotly
        import pymongo
        import pandas
        import numpy
        import sklearn
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def install_requirements():
    """Install requirements if needed"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def test_mongo_connection():
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        MONGO_URI = 'mongodb+srv://saz_db1328:wUlewP7Ijtr80RqC@cluster0.j3swhkq.mongodb.net/'
        client = MongoClient(MONGO_URI)
        db = client["aqi_project"]
        
        # Test collections
        feature_count = db["feature_store"].count_documents({})
        model_count = db["model_registry"].count_documents({})
        
        print(f"✅ MongoDB connection successful")
        print(f"   - Feature store records: {feature_count}")
        print(f"   - Model registry records: {model_count}")
        
        if feature_count == 0:
            print("⚠️  Warning: No data in feature_store. Run feature engineering first.")
        if model_count == 0:
            print("⚠️  Warning: No model in model_registry. Run model training first.")
            
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def main():
    print("🌫️ AQI Prediction Dashboard Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n📦 Installing missing packages...")
        if not install_requirements():
            print("❌ Setup failed. Please install requirements manually:")
            print("   pip install -r requirements.txt")
            return
    
    print("\n🔌 Testing MongoDB connection...")
    if not test_mongo_connection():
        print("❌ Setup incomplete. Please check MongoDB connection.")
        return
    
    print("\n🎉 Setup complete!")
    print("\nTo start the dashboard:")
    print("   streamlit run app.py")
    print("\nOr run the batch file:")
    print("   run_app.bat")

if __name__ == "__main__":
    main()