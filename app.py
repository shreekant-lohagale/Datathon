import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import clean_dataframe, engineer_features, load_model

# =========================
#   PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Datathon — Price Prediction App",
    layout="wide",
    page_icon="📊",
)

# =========================
#   CUSTOM CSS THEME
# =========================
st.markdown("""
<style>
body {
    background-color: #0d0f17;
    color: #ffffff;
}
.sidebar .sidebar-content {
    background-color: #11131c;
}
.block-container {
    padding: 2rem;
}
.stButton>button {
    background-color: #5b78f6 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 1rem;
}
.stDownloadButton>button {
    background-color: #2ecc71 !important;
    color: black !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
}
.dataframe {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
#   HEADER
# =========================
st.title("🏠 Pune Builder Floor Price Prediction")
st.markdown(
    "Upload a dataset and get predicted property prices with an optimized ML pipeline."
)

# =========================
#   MODEL LOAD
# =========================
model = load_model()

# =========================
#   FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📌 Preview Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        # ============================
        #   CLEANING + FEATURE ENGINEERING
        # ============================
        df_clean = clean_dataframe(df)
        df_fe = engineer_features(df_clean)

        st.subheader("🔧 Processed Data Sample")
        st.dataframe(df_fe.head(), use_container_width=True)

        # ============================
        #   PREDICTION
        # ============================
        st.subheader("🎯 Generating Predictions")
        predictions = model.predict(df_fe)

        df_output = df_clean.copy()
        df_output["Predicted_price"] = predictions

        st.success("✅ Predictions generated successfully!")

        st.subheader("📊 Predicted Values Preview")
        st.dataframe(df_output.head(), use_container_width=True)

        # ============================
        #   VISUALIZATION
        # ============================
        fig = px.histogram(
            df_output,
            x="Predicted_price",
            nbins=30,
            title="Prediction Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#5b78f6"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # ============================
        #   DOWNLOAD BUTTON
        # ============================
        st.download_button(
            "⬇️ Download Predictions CSV",
            df_output.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error while processing file: {e}")

else:
    st.info("Upload a CSV file to start prediction.")
