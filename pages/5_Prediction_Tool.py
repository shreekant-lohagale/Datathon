import streamlit as st
from utils import load_csv, sanitize_dataframe
import numpy as np

st.title("🧮 Prediction Tool (Demo)")

df = load_csv("processed_data.csv") or load_csv("output_Pune_builderfloor.csv")

if df is None:
    st.error("Dataset not found.")
    st.stop()

df = sanitize_dataframe(df)

st.write("### Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (sq ft)", 200, 5000, 1000)
with col2:
    locality_score = st.number_input("Locality Score (0–10)", 0.0, 10.0, 6.5)
with col3:
    builder_exp = st.number_input("Builder Experience (years)", 0.0, 30.0, 5.0)

if st.button("Predict"):
    predicted = (
        3000 * area +
        100000 * (locality_score / 10) +
        20000 * np.log1p(builder_exp)
    )
    st.success(f"Estimated Price (demo): ₹ {predicted:,.0f}")
