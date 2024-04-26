import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('Used_Bikes.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_price(bike_name,	city,	kms_driven,	owner,	age,	power,	brand):
    features = np.array([bike_name,	city,	kms_driven,	owner,	age,	power,	brand])
    features = features.reshape(1,-1)
    emission = model.predict(features)
    return emission[0]

# Streamlit UI
st.title('BIKE PRICE PREDICTION')
st.write("""
## Input Features
ENTER THE VALUES FOR THE INPUT FEATURES TO PREDICT PRICE OF BIKE.
""")

# Input fields for user
bike_name = st.number_input('BIKE_NAME')
city = st.number_input('CITY')
kms_driven = st.number_input('KMS_DRIVEN')
owner = st.number_input('OWNER')
age = st.number_input('AGE')
power = st.number_input('POWER')
brand = st.number_input('BRAND')

# Prediction button
if st.button('Predict'):
    # Predict EMISSION
    price_prediction = predict_price(bike_name,	city,	kms_driven,	owner,	age,	power,	brand)
    st.write(f"PREDICTED PRICE OF BIKE: {price_prediction}")