import streamlit as st
import plotly.express as px
from utils import load_csv, sanitize_dataframe

st.title("🔗 Correlation Heatmap")

df = load_csv("processed_data.csv") or load_csv("output_Pune_builderfloor.csv")
if df is None:
    st.error("Dataset missing.")
    st.stop()

df = sanitize_dataframe(df)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    st.warning("No numeric columns available.")
else:
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
