import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="PredictionPage")
st.title("Prediction")
with open("pages/df.pkl", "rb") as file:
    df = pickle.load(file)
with open("pages/pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)
st.title("Hello!")

property_type = st.selectbox("Property_Type", df["property_type"].unique())
sector = st.selectbox("Sector", df["sector"].sort_values().unique())
bedroom = float(st.number_input("Number of bedroom", min_value=1, max_value=100))
bathroom = float(st.number_input("Number of bathroom", min_value=1, max_value=100))
balcony = st.selectbox("Balcony", df["balcony"].unique())
agepossession = st.selectbox("AgePossession", df["agePossession"].unique())
built_up_area = float(st.number_input("Built_up_area"))
servantroom = float(st.selectbox("ServantRoom", df["servant room"].unique()))
storeroom = float(st.selectbox("StoreRoom", df["store room"].unique()))
furnish = st.selectbox("FurnishingType", df["furnishing_type"].unique())
luxury = st.selectbox("Luxury", df["luxury_category"].unique())
floor = st.selectbox("Floor", df["floor_category"].unique())
# Create a DataFrame using user inputs

if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "property_type": property_type,
                "sector": sector,
                "bedRoom": bedroom,
                "bathroom": bathroom,
                "balcony": balcony,
                "agePossession": agepossession,
                "built_up_area": built_up_area,
                "servant room": servantroom,
                "store room": storeroom,
                "furnishing_type": furnish,
                "luxury_category": luxury,
                "floor_category": floor,
            }
        ]
    )

    prediction = pipeline.predict(input_data)  # Pass actual DataFrame
    base = np.expm1(prediction[0])
    st.text(base)
    low = base - 0.22
    high = base + 0.22
    st.text(f"Predicted Price: {low} to {base}")  # Display result
