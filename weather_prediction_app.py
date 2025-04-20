import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import streamlit as st
import os

# Function to generate or load sample weather dataset
def load_data():
    # Check if dataset exists, else create a synthetic one
    if not os.path.exists('weather_data.csv'):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        data = {
            'date': dates,
            'temperature': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 2, len(dates)),
            'humidity': 60 + 20 * np.random.random(len(dates)),
            'pressure': 1013 + 10 * np.random.normal(0, 1, len(dates)),
            'wind_speed': 5 + 5 * np.random.random(len(dates))
        }
        df = pd.DataFrame(data)
        df.to_csv('weather_data.csv', index=False)
    return pd.read_csv('weather_data.csv')

# Function to train and save the model
def train_model():
    df = load_data()
    
    # Features and target
    features = ['humidity', 'pressure', 'wind_speed']
    target = 'temperature'
    
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

# Streamlit app
def main():
    st.title("Weather Temperature Prediction")
    st.write("Enter weather parameters to predict the temperature.")
    
    # Load or train model
    if os.path.exists('weather_model.pkl') and os.path.exists('scaler.pkl'):
        model = joblib.load('weather_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = ['humidity', 'pressure', 'wind_speed']
    else:
        model, scaler, features = train_model()
    
    # User input
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)
    
    # Predict
    if st.button("Predict Temperature"):
        input_data = np.array([[humidity, pressure, wind_speed]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Temperature: {prediction:.2f} °C")

if __name__ == "__main__":
    main()