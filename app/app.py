import streamlit as st
import pandas as pd
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_model

@st.cache_resource
def get_models():
    hgbr = load_model('models/hgbr_best_model.joblib')
    rfr = load_model('models/RandomForestRegressor.joblib')
    rfecv = load_model('models/RFECV_fitted.joblib')
    scaler = load_model('models/RobustScaler_fitted.joblib')

    return {'RandomForest': rfr, 'HistGradientBoosting': hgbr}, rfecv, scaler

models, rfecv, scaler = get_models()


st.title("🏠 California House Price Prediction")
st.write("""
         This app predicts the median house value in California based on several features. 
         Please provide the input features in the sidebar to get a prediction.
         """)

st.sidebar.header("Input House Features")

def get_user_input():
    longitude = st.sidebar.slider('Longitude', -124.3, -114.3, -118.5, 0.1)
    latitude = st.sidebar.slider('Latitude', 32.5, 42.0, 34.2, 0.1)
    housing_median_age = st.sidebar.slider('Housing Median Age', 1.0, 52.0, 10.0, 1.0)
    median_income = st.sidebar.slider('Median Income (in tens of thousands of $)', 0.5, 12.0, 3.5, 0.1)
    rooms_per_house = st.sidebar.slider('Number of Rooms', 1.0, 12.0, 5.0, 1.0)
    bedrooms_per_house = st.sidebar.slider('Number of Bedrooms', 1.0, 6.0, 1.0, 1.0)
    ocean_proximity = st.sidebar.selectbox(
        'Ocean Proximity', 
        ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND')
    )

    data = {
        'longitude': longitude, 
        'latitude': latitude, 
        'housing_median_age': housing_median_age, 
        'median_income': median_income, 
        'rooms_per_house': rooms_per_house, 
        'bedrooms_per_house': bedrooms_per_house, 
        'ocean_proximity': ocean_proximity 
    }

    features = pd.DataFrame(data, index=[0])

    return features


input_df = get_user_input()


ocean_prox_dummies = pd.get_dummies(input_df['ocean_proximity'], prefix='ocean_proximity')
processed_df = pd.concat([input_df.drop('ocean_proximity', axis=1), ocean_prox_dummies], axis=1)

expected_columns = [
    'longitude', 'latitude', 'housing_median_age', 'median_income', 
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
    'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN', 'rooms_per_house', 'bedrooms_per_house'
    ]

for col in expected_columns:
    if col not in processed_df.columns:
        processed_df[col] = 0

processed_df = processed_df[expected_columns]

st.subheader('Select a Model for Prediction')

model_choice = st.selectbox('Choose a regression model:', list(models.keys()))
model = models[model_choice]

if st.button('Predict House Value'):
    try:
        input_rfecv = rfecv.transform(processed_df)
        input_scaled = scaler.transform(input_rfecv)
        prediction = model.predict(input_scaled)

        st.subheader('Prediction Result')
        st.metric(label='Predicted Median House Value', 
                  value=f"${prediction[0]:,.2f}")
        
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')


st.subheader('Your Inputs')
st.write(input_df)



