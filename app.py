import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # penting!

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # ====== Lakukan preprocessing seperti yang dilakukan saat training ======
    df['Levy'] = df['Levy'].replace('-', np.nan)
    df['Levy'] = df['Levy'].fillna(df['Levy'].median())

    df['Has Turbo'] = df['Engine volume'].astype(str).str.contains('Turbo')
    df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '', regex=False)
    df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')

    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').str.replace(',', '').str.strip()
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
    
    df['Price'] = 0  # Dummy value to allow preprocessing
    df['volume_per_cylinder'] = df['Engine volume'] / df['Cylinders']
    df['fuel_gear'] = df['Fuel type'].astype(str) + '_' + df['Gear box type'].astype(str)
    df['car_age'] = 2025 - df['Prod. year']
    df['Mileage_Age_Interaction'] = df['Mileage'] / (df['car_age'] + 1)
    df['Mileage_bin'] = pd.cut(df['Mileage'], bins=4, labels=False)
    df['Volume_Turbo'] = df['Engine volume'] * df['Has Turbo'].astype(int)

    # Encoding kategorikal
    df['Leather interior'] = df['Leather interior'].map({'Yes': True, 'No': False})
    df['Wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})

    door_mapping = {'02-Mar': 3, '04-May': 5, '>5': 6}
    df['Doors'] = df['Doors'].map(door_mapping).astype('int')

    cat_cols = ['Manufacturer', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels']
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        df[col] = df[col].replace(rare, 'Rare')

    df = pd.get_dummies(df, drop_first=False)

    # ====== Ini yang penting! Samakan dengan feature_names training ======
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scaling numerik
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

def run_prediction_app():
    st.title("Car Price Prediction")

    # Form input pengguna
    user_input = {
        "Levy": st.number_input("Levy", value=0.0),
        "Manufacturer": st.selectbox("Manufacturer", ["Toyota", "BMW", "Mercedes-Benz", "Rare"]),
        "Model": st.text_input("Model", value="Other"),
        "Prod. year": st.number_input("Production Year", value=2015),
        "Category": st.selectbox("Category", ["Sedan", "Jeep", "Hatchback", "Rare"]),
        "Leather interior": st.selectbox("Leather interior", ["Yes", "No"]),
        "Fuel type": st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Rare"]),
        "Engine volume": st.text_input("Engine volume", value="2.0"),
        "Mileage": st.text_input("Mileage (e.g., 100000)", value="100000"),
        "Cylinders": st.number_input("Cylinders", value=4),
        "Gear box type": st.selectbox("Gear Box Type", ["Automatic", "Manual", "Tiptronic", "Rare"]),
        "Drive wheels": st.selectbox("Drive Wheels", ["Front", "Rear", "4x4", "Rare"]),
        "Doors": st.selectbox("Doors", ['02-Mar', '04-May', '>5']),
        "Wheel": st.selectbox("Wheel", ['Left wheel', 'Right-hand drive']),
        "Color": st.selectbox("Color", ['Black', 'White', 'Silver', 'Rare'])
    }

    if st.button("Predict"):
        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Car Price: {prediction:,.0f}")

def main():
    run_prediction_app()

if __name__ == '__main__':
    main()
