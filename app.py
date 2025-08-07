import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load model, scaler, dan fitur
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Definisikan kolom numerik (harus sama saat training)
numeric_cols = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'volume_per_cylinder', 
                'car_age', 'Mileage_Age_Interaction', 'Mileage_bin', 'Volume_Turbo']

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Buat fitur turunan
    df['volume_per_cylinder'] = df['Engine volume'] / df['Cylinders']
    df['car_age'] = 2025 - df['Prod. year']
    df['Mileage_Age_Interaction'] = df['Mileage'] / (df['car_age'] + 1)
    df['Volume_Turbo'] = df['Engine volume'] * int(df['Has Turbo'])
    df['fuel_gear'] = df['Fuel type'] + '_' + df['Gear box type']
    df['Model_encoded'] = 0  # default

    # Bin Mileage (pakai sama dengan saat training)
    df['Mileage_bin'] = pd.cut(df['Mileage'], bins=4, labels=False)

    # Encoding biner
    df['Leather interior'] = df['Leather interior'].map({'Yes': True, 'No': False})
    df['Wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})
    df['Doors'] = df['Doors'].map({'02-Mar': 3, '04-May': 5, '>5': 6})

    # Rare label encoding + one-hot (pola sama seperti training)
    df = pd.get_dummies(df)

    # Reindex agar sesuai dengan training
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scaling kolom numerik
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

def run_prediction_app():
    st.title("🚗 Car Price Prediction")

    with st.form("input_form"):
        Manufacturer = st.selectbox("Manufacturer", ["Toyota", "Mercedes-Benz", "BMW", "Rare"])
        Category = st.selectbox("Category", ["Jeep", "Sedan", "Hatchback", "Rare"])
        Fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Rare"])
        Gear_box_type = st.selectbox("Gear Box Type", ["Automatic", "Manual", "CVT", "Rare"])
        Drive_wheels = st.selectbox("Drive Wheels", ["Front", "Rear", "4x4", "Rare"])
        Leather_interior = st.selectbox("Leather Interior", ["Yes", "No"])
        Wheel = st.selectbox("Steering Wheel Side", ["Left wheel", "Right-hand drive"])
        Doors = st.selectbox("Number of Doors", ["02-Mar", "04-May", ">5"])
        Levy = st.number_input("Levy", value=0)
        Mileage = st.number_input("Mileage (km)", value=100000)
        Engine_volume = st.number_input("Engine Volume", value=2.0)
        Cylinders = st.number_input("Cylinders", value=4)
        Prod_year = st.number_input("Production Year", value=2015)
        Has_Turbo = st.checkbox("Has Turbo", value=False)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        user_input = {
            "Manufacturer": Manufacturer,
            "Category": Category,
            "Fuel type": Fuel_type,
            "Gear box type": Gear_box_type,
            "Drive wheels": Drive_wheels,
            "Leather interior": Leather_interior,
            "Wheel": Wheel,
            "Doors": Doors,
            "Levy": Levy,
            "Mileage": Mileage,
            "Engine volume": Engine_volume,
            "Cylinders": Cylinders,
            "Prod. year": Prod_year,
            "Has Turbo": Has_Turbo
        }

        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Car Price: ${prediction:,.0f}")

def main():
    run_prediction_app()

if __name__ == "__main__":
    main()
