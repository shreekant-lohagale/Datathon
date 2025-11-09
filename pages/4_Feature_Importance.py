import streamlit as st
from utils import load_csv, sanitize_dataframe
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.title("🔥 Feature Importance")

df = load_csv("processed_data.csv")
if df is None:
    st.error("Dataset unavailable.")
    st.stop()

df = sanitize_dataframe(df)

target = st.selectbox("Select target column", df.select_dtypes(include="number").columns)

feature_cols = st.multiselect(
    "Select feature columns",
    df.select_dtypes(include="number").columns.drop(target)
)

if st.button("Train Random Forest"):
    X = df[feature_cols]
    y = df[target]

    model = RandomForestRegressor()
    model.fit(X, y)

    importance = model.feature_importances_

    fig = px.bar(
        x=feature_cols,
        y=importance,
        labels={"x": "Feature", "y": "Importance Score"},
        title="Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)
