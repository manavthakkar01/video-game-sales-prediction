# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("vg_sales_model.pkl")
platform_encoder = joblib.load("platform_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

st.set_page_config(page_title="Video Game Sales Predictor")

st.title("🎮 Video Game Global Sales Prediction")

st.sidebar.header("Enter Game Details")

platform = st.sidebar.selectbox("Platform", platform_encoder.classes_)
genre = st.sidebar.selectbox("Genre", genre_encoder.classes_)

year = st.sidebar.number_input("Release Year", min_value=1980, max_value=2025, step=1)
na_sales = st.sidebar.number_input("NA Sales", min_value=0.0)
eu_sales = st.sidebar.number_input("EU Sales", min_value=0.0)
jp_sales = st.sidebar.number_input("JP Sales", min_value=0.0)
other_sales = st.sidebar.number_input("Other Sales", min_value=0.0)

if st.sidebar.button("Predict Global Sales"):
    platform_encoded = platform_encoder.transform([platform])[0]
    genre_encoded = genre_encoder.transform([genre])[0]

    input_data = np.array([[platform_encoded, genre_encoded, year,
                            na_sales, eu_sales, jp_sales, other_sales]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Global Sales: {prediction[0]:.2f} million units")

st.markdown("---")
st.markdown("Developed by **Manav Thakkar**")
