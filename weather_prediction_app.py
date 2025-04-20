import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import streamlit as st
import os

# Function to load the dataset
def load_data():
    try:
        # Attempt to load the dataset from the local repository
        data_path = "weather_data.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback to GitHub raw URL (replace with your actual GitHub raw URL)
            github_raw_url = "https://raw.githubusercontent.com/your-username/your-repo/main/weather_data.csv"
            df = pd.read_csv(github_raw_url)
        
        # Select relevant columns and handle missing values
        df = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']].dropna()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train and save the model
def train_model():
    try:
        df = load_data()
        if df is None:
            return None, None, None
        
        # Features and target
        features = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
        target = 'Temperature (C)'
        
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f"Model Mean Absolute Error: {mae:.2f} °C")
        
        # Save model and scaler
        joblib.dump(model, 'weather_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

# Streamlit app
def main():
    st.title("Weather Temperature Prediction")
    st.write("Enter weather parameters to predict the temperature.")
    
    # Load or train model
    try:
        if st.button("Train Model"):
            model, scaler, features = train_model()
        else:
            # Check if model and scaler exist
            model_path = 'weather_model.pkl'
            scaler_path = 'scaler.pkl'
            model = joblib.load(model_path) if os.path.exists(model_path) else None
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            features = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
        
        if model is None or scaler is None:
            st.warning("Model not trained. Click 'Train Model' to initialize.")
            return
        
        # User input
        humidity = st.slider("Humidity (%)", 0.0, 1.0, 0.6, step=0.01)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, step=0.1)
        pressure = st.slider("Pressure (millibars)", 900.0, 1100.0, 1013.0, step=0.1)
        
        # Predict
        if st.button("Predict Temperature"):
            input_data = np.array([[humidity, wind_speed, pressure]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            st.success(f"Predicted Temperature: {prediction:.2f} °C")
    except Exception as e:
        st.error(f"Error loading model or predicting: {e}")

if __name__ == "__main__":
    main()
