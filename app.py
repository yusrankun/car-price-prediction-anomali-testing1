import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import pickle

# Load pre-trained CatBoost model
with open('XGBoost_Model.pkl', 'rb') as file:
    model = pickle.load(file)

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Car Price Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Powered by Machine Learning</h4> 
               </div>"""

desc_temp = """### Car Price Prediction App
This app predicts the price of a used car based on input features.

#### Data Source
Kaggle (Car Price Prediction dataset)
"""

# Main Streamlit App
def main():
    stc.html(html_temp)
    menu = ["Home", "Price Predictor"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Price Predictor":
        run_prediction_app()

# Prediction Form
def run_prediction_app():
    st.subheader("Car Details Input")

    col1, col2 = st.columns(2)

    manufacturer = col1.selectbox("Manufacturer", ['Toyota', 'BMW', 'Mercedes-Benz', 'Rare'])
    category = col2.selectbox("Category", ['Sedan', 'Jeep', 'Hatchback', 'Rare'])
    fuel_type = col1.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Rare'])
    gear_box = col2.selectbox("Gear Box Type", ['Automatic', 'Manual', 'Tiptronic', 'CVT', 'Rare'])
    drive_wheels = col1.selectbox("Drive Wheels", ['Front', 'Rear', 'All', 'Rare'])

    prod_year = col2.slider("Production Year", 1995, 2025, 2015)
    engine_volume = col1.slider("Engine Volume (L)", 0.8, 6.5, 2.0)
    cylinders = col2.slider("Cylinders", 2, 16, 4)
    turbo = col1.selectbox("Has Turbo?", ["Yes", "No"])
    mileage = col2.number_input("Mileage (km)", value=100000)
    levy = col1.number_input("Levy", value=0)
    doors = col2.selectbox("Doors", [3, 5, 6])
    leather = col1.selectbox("Leather Interior", ["Yes", "No"])
    wheel = col2.selectbox("Wheel Position", ["Left", "Right"])
    model_encoded = col1.slider("Model Mean Price (Encoding)", 5000, 80000, 20000)

    # Predict Button
    if st.button("Predict Price"):
        features = preprocess_input(manufacturer, category, fuel_type, gear_box, drive_wheels,
                                    prod_year, engine_volume, cylinders, turbo, mileage,
                                    levy, doors, leather, wheel, model_encoded)

        prediction = model.predict([features])[0]
        st.success(f"Estimated Car Price: ${int(prediction):,}")

# Preprocessing user input for prediction
def preprocess_input(manufacturer, category, fuel_type, gear_box, drive_wheels,
                     prod_year, engine_volume, cylinders, turbo, mileage,
                     levy, doors, leather, wheel, model_encoded):

    car_age = 2025 - prod_year
    has_turbo = 1 if turbo == "Yes" else 0
    volume_per_cylinder = engine_volume / cylinders
    mileage_age_interaction = mileage / (car_age + 1)
    volume_turbo = engine_volume * has_turbo
    fuel_gear = fuel_type + "_" + gear_box
    mileage_bin = pd.cut([mileage], bins=4, labels=False)[0]

    # Encoding categorical variables manually
    def encode_cat(value, mapping):
        return mapping.get(value, mapping['Rare'])

    manu_map = {'Toyota': 0, 'BMW': 1, 'Mercedes-Benz': 2, 'Rare': 3}
    cat_map = {'Sedan': 0, 'Jeep': 1, 'Hatchback': 2, 'Rare': 3}
    fuel_gear_map = {'Petrol_Automatic': 0, 'Diesel_Manual': 1, 'Rare': 2}
    drive_map = {'Front': 0, 'Rear': 1, 'All': 2, 'Rare': 3}

    encoded = [
        levy,
        mileage,
        prod_year,
        engine_volume,
        cylinders,
        doors,
        car_age,
        mileage_age_interaction,
        model_encoded,
        volume_per_cylinder,
        volume_turbo,
        mileage_bin if not pd.isna(mileage_bin) else 0,
        encode_cat(manufacturer, manu_map),
        encode_cat(category, cat_map),
        1 if leather == "Yes" else 0,
        1 if wheel == "Right" else 0,
        has_turbo,
        encode_cat(drive_wheels, drive_map),
        encode_cat(fuel_gear, fuel_gear_map)
    ]
    return encoded

if __name__ == '__main__':
    main()
