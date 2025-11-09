import streamlit as st
from utils import load_csv, sanitize_dataframe

st.title("🤖 Model Overview")

df = load_csv("processed_data.csv")
if df is None:
    st.error("Please generate processed_data.csv first.")
    st.stop()

df = sanitize_dataframe(df)

st.write("### Placeholder Model Section")
st.info("Here you can fit models, train/test split, evaluate RMSE, R², etc.")
