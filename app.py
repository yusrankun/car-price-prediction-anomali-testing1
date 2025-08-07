import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan fitur
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # X.columns.tolist() saat training

# Daftar fitur numerik (harus cocok dengan training)
numeric_cols = [
    'Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders',
    'Doors', 'Wheel', 'volume_per_cylinder', 'car_age',
    'Mileage_Age_Interaction', 'Mileage_bin', 'Volume_Turbo', 'Model_encoded'
]

def preprocess_input(user_input):
    df = user_input.copy()

    # Handle Turbo
    df['Has Turbo'] = df['Engine volume'].astype(str).str.contains('Turbo', case=False)
    df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '', regex=False)
    df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')

    # Feature Engineering
    df['volume_per_cylinder'] = df['Engine volume'] / df['Cylinders']
    df['car_age'] = 2025 - df['Prod. year']
    df['Mileage_Age_Interaction'] = df['Mileage'] / (df['car_age'] + 1)
    df['Mileage_bin'] = pd.cut(df['Mileage'], bins=4, labels=False)
    df['Volume_Turbo'] = df['Engine volume'] * df['Has Turbo'].astype(int)

    # Model_encoded (pakai mean Price dummy, bisa diganti default 0 jika tidak tersedia)
    df['Model_encoded'] = 0

    # Rare Encoding tidak perlu di sini karena saat dummies nanti fitur disamakan

    # Convert binary
    df['Leather interior'] = df['Leather interior'].map({'Yes': True, 'No': False})
    df['Wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=False)

    # Reindex supaya cocok dengan training features
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scaling
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

def run_prediction_app():
    st.title("Car Price Prediction App 🚗")

    # Input form
    with st.form("car_input_form"):
        Manufacturer = st.selectbox("Manufacturer", ['Toyota', 'Mercedes-Benz', 'BMW', 'Ford', 'Rare'])
        Category = st.selectbox("Category", ['Sedan', 'Jeep', 'Hatchback', 'Microbus', 'Rare'])
        Fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Rare'])
        Gear_box = st.selectbox("Gear Box", ['Automatic', 'Manual', 'Tiptronic', 'Variator', 'Rare'])
        Drive_wheels = st.selectbox("Drive Wheels", ['Front', 'Rear', '4x4', 'Rare'])

        Levy = st.number_input("Levy", min_value=0.0, value=0.0)
        Prod_year = st.number_input("Production Year", min_value=1950, max_value=2025, value=2015)
        Engine_volume = st.text_input("Engine Volume (e.g. 2.0 or 2.0 Turbo)", "2.0")
        Mileage = st.number_input("Mileage (in km)", min_value=0, value=100000)
        Cylinders = st.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)
        Doors = st.selectbox("Doors", [3, 5, 6])
        Leather = st.selectbox("Leather Interior", ['Yes', 'No'])
        Wheel = st.selectbox("Steering Wheel", ['Left wheel', 'Right-hand drive'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.DataFrame([{
            'Manufacturer': Manufacturer,
            'Category': Category,
            'Fuel type': Fuel_type,
            'Gear box type': Gear_box,
            'Drive wheels': Drive_wheels,
            'Levy': Levy,
            'Prod. year': Prod_year,
            'Engine volume': Engine_volume,
            'Mileage': Mileage,
            'Cylinders': Cylinders,
            'Doors': Doors,
            'Leather interior': Leather,
            'Wheel': Wheel,
        }])

        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Car Price: ${prediction:,.2f}")

def main():
    run_prediction_app()

if __name__ == "__main__":
    main()
