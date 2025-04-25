import streamlit as st
import pickle
import numpy as np

# Load model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load LabelEncoder for Location
with open('encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Title
st.title("🌦️ Rain Prediction App")
st.write("Provide the weather and date details to predict if it will rain tomorrow.")

# Input fields
location = st.selectbox("📍 Location", label_encoder.classes_)  # shows country/city names
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-30.0, max_value=60.0, value=22.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
wind_speed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, value=0.0)
cloud_cover = st.number_input("☁️ Cloud Cover (%)", min_value=1.0, max_value=100.0, value=40.0)
pressure = st.number_input("📉 Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)

# Date inputs
year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2025)
month = st.number_input("📅 Month", min_value=1, max_value=12, value=4)
day = st.number_input("📅 Day", min_value=1, max_value=31, value=25)

# Feature engineering
temp_humidity_interaction = temperature * humidity
wind_cloud_ratio = wind_speed / cloud_cover if cloud_cover != 0 else 0

# Encode location
location_encoded = label_encoder.transform([location])[0]

# Predict
if st.button("🔍 Predict"):
    features = np.array([[ 
        location_encoded, temperature, humidity, wind_speed,
        precipitation, cloud_cover, pressure,
        temp_humidity_interaction, wind_cloud_ratio,
        year, month, day
    ]])

    prediction = model.predict(features)

    

    

    if prediction[0] == 1:
        st.success("✅ It WILL rain tomorrow.")
    else:
        st.info("☀️ It will NOT rain tomorrow.")
