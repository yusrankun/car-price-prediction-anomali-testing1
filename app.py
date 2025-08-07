import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import pandas as pd
import pickle

# ==============================
# Load trained XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

import joblib
feature_names = joblib.load('feature_names.pkl')

# ==============================
# UI Layout
html_temp = """
<div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Car Price Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Built with XGBoost</h4> 
</div>
"""

desc_temp = """
### Car Price Prediction App
This app predicts the estimated price of a used car.

#### Notes:
- All inputs are based on cleaned features
- The model used is XGBoost Regressor
"""

# ==============================
# Sidebar menu
def main():
    stc.html(html_temp)
    menu = ["Home", "Predict Price"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Predict Price":
        run_prediction_app()

# ==============================
# Input form and prediction
def run_prediction_app():
    st.subheader("Input Car Features")

    col1, col2 = st.columns(2)

    # Sample user inputs
    levy = col1.number_input("Levy", value=0)
    mileage = col2.number_input("Mileage (km)", value=100000)
    prod_year = col1.slider("Production Year", 1995, 2025, 2015)
    engine_volume = col2.slider("Engine Volume (L)", 0.8, 6.5, 2.0)
    cylinders = col1.slider("Cylinders", 2, 16, 4)
    doors = col2.selectbox("Doors", [3, 5, 6])
    leather = col1.selectbox("Leather Interior", ["Yes", "No"])
    wheel = col2.selectbox("Wheel (Driving Side)", ["Left", "Right"])
    has_turbo = col1.selectbox("Has Turbo?", ["Yes", "No"])
    drive = col2.selectbox("Drive Wheels", ['Front', 'Rear', 'All'])
    fuel_gear = col1.selectbox("Fuel + Gear Type", ['Petrol_Automatic', 'Diesel_Manual', 'Rare'])
    model_encoded = col2.slider("Model Mean Price Encoding", 5000, 100000, 20000)

    # Prediction button
    if st.button("Predict"):
        # Preprocess input
        features = preprocess_input(
            levy, mileage, prod_year, engine_volume, cylinders, doors,
            leather, wheel, has_turbo, drive, fuel_gear, model_encoded
        )

        input_df = pd.DataFrame([features])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: ${int(prediction):,}")

# ==============================
# Feature preprocessing (match model input)
def preprocess_input(levy, mileage, prod_year, engine_volume, cylinders, doors,
                     leather, wheel, has_turbo, drive, fuel_gear, model_encoded):

    car_age = 2025 - prod_year
    mileage_age_interaction = mileage / (car_age + 1)
    volume_per_cylinder = engine_volume / cylinders
    volume_turbo = engine_volume * (1 if has_turbo == "Yes" else 0)
    mileage_bin = pd.cut([mileage], bins=4, labels=False)[0]

    # Encode categoricals
    leather_binary = 1 if leather == "Yes" else 0
    wheel_binary = 1 if wheel == "Right" else 0
    has_turbo_binary = 1 if has_turbo == "Yes" else 0
    drive_encoded = {"Front": 0, "Rear": 1, "All": 2}.get(drive, 0)
    fuel_gear_encoded = {"Petrol_Automatic": 0, "Diesel_Manual": 1, "Rare": 2}.get(fuel_gear, 2)

    return {
        'Levy': levy,
        'Mileage': mileage,
        'Prod. year': prod_year,
        'Engine volume': engine_volume,
        'Cylinders': cylinders,
        'Doors': doors,
        'car_age': car_age,
        'Mileage_Age_Interaction': mileage_age_interaction,
        'Model_encoded': model_encoded,
        'volume_per_cylinder': volume_per_cylinder,
        'Volume_Turbo': volume_turbo,
        'Mileage_bin': mileage_bin,
        'Leather interior': leather_binary,
        'Wheel': wheel_binary,
        'Has Turbo': has_turbo_binary,
        'Drive wheels': drive_encoded,
        'fuel_gear': fuel_gear_encoded
    }

# ==============================
if __name__ == "__main__":
    main()
