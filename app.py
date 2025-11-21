# ---------------------------------------------------------
# STREAMLIT CAR PRICE PREDICTION APP (Starter Version)
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------
# Load the trained model and scaler
# ---------------------------------------------------------
rf_model = joblib.load("car_price_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------------
# Title of the App
# ---------------------------------------------------------
st.title("🚗 Car Price Prediction App")
st.write("Enter car specifications to predict the price.")

# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------
enginesize = st.slider("Engine Size", 50, 350, 150)
curbweight = st.slider("Curb Weight", 1500, 4500, 2800)
horsepower = st.slider("Horsepower", 40, 300, 120)
highwaympg = st.slider("Highway MPG", 10, 55, 30)
citympg = st.slider("City MPG", 5, 45, 25)
carwidth = st.slider("Car Width", 60, 75, 65)
carlength = st.slider("Car Length", 140, 210, 170)

# ---------------------------------------------------------
# Create input array
# ---------------------------------------------------------
input_data = np.array([[enginesize, curbweight, horsepower, highwaympg, citympg, carwidth, carlength]])

# Scale input
input_scaled = scaler.transform(input_data)

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Price"):
    pred = rf_model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Price: **${int(pred)}**")
