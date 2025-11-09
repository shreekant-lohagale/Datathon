import streamlit as st
from utils import load_csv, sanitize_dataframe
import plotly.express as px

st.title("📊 EDA – Exploratory Data Analysis")

df = load_csv("processed_data.csv") or load_csv("output_Pune_builderfloor.csv")

if df is None:
    st.error("No dataset found in /data folder.")
    st.stop()

df = sanitize_dataframe(df)

st.write("### Dataset Preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

st.write("### Visual Explorer")

c1, c2, c3 = st.columns(3)
with c1:
    x_col = st.selectbox("X", all_cols)
with c2:
    y_col = st.selectbox("Y (numeric)", numeric_cols)
with c3:
    color = st.selectbox("Color", [None] + all_cols)

# ✅ Narwhals-safe plotting
x_series = df[x_col]
y_series = df[y_col]
color_series = df[color] if color else None

fig = px.scatter(
    x=x_series,
    y=y_series,
    color=color_series,
    labels={"x": x_col, "y": y_col},
)
st.plotly_chart(fig, use_container_width=True)

st.write("### Distribution")
dist_col = st.selectbox("Distribution Column", numeric_cols)

fig2 = px.histogram(x=df[dist_col], nbins=40)
st.plotly_chart(fig2, use_container_width=True)
