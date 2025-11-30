# ============================
#  FINAL WORKING STREAMLIT APP
# ============================

import os
os.environ["PLOTLY_NARWHALS_ENABLED"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================
#  --- PROFESSIONAL UI THEME ---
# ============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background: radial-gradient(circle at top, #0a0f1f, #05070d);
    color: #E8E8E8 !important;
}

section.main > div { 
    padding-top: 1rem;
}

div.block-container {
    background: rgba(255,255,255,0.03);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
}

h1, h2, h3, h4 {
    color: #38bdf8;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg,#38bdf8,#0ea5e9);
    color: white;
    border-radius: 12px;
    padding: 10px 22px;
    border: none;
    font-size: 16px;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#7dd3fc,#38bdf8);
}

.stTabs [data-baseweb="tab"] {
    font-size: 17px;
    padding: 10px 18px;
}

.dataframe tbody tr {
    background: rgba(255,255,255,0.02) !important;
}

</style>
""", unsafe_allow_html=True)


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="🏠 Housing Price Explorer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Housing Price Explorer – Datathon 2025")
st.caption("Modern UI • Clean EDA • Ready for Model Integration")


# ============================
#  HELPERS
# ============================
DATA_DIR = Path("data")


def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        base = str(c).strip()
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}.{count}"
        while name in out:
            count += 1
            name = f"{base}.{count}"
        seen[base] = count + 1
        out.append(name)
    return out


def sanitize_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.columns = make_unique_columns(df.columns)

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.replace(",", "", regex=False)
            except:
                pass
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except:
                pass

    return df


def load_csv_safely(filename: str):
    p = DATA_DIR / filename
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)
        df.columns = make_unique_columns(df.columns)
        df = sanitize_dataframe(df)
        return df

    except Exception as e:
        st.warning(f"Failed to load {filename}: {e}")
        return None


# ============================
# LOAD DATA
# ============================
raw_df = load_csv_safely("output_Pune_builderfloor.csv")
proc_df = load_csv_safely("processed_data.csv")


# ============================
# TABS
# ============================
tab1, tab2, tab3 = st.tabs(["📁 Data", "📊 EDA", "🤖 Predict"])

# ----------------------------
# TAB 1 — DATA PREVIEW
# ----------------------------
with tab1:
    st.subheader("📁 Available Datasets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### output_Pune_builderfloor.csv")
        if raw_df is not None:
            st.write("Shape:", raw_df.shape)
            st.dataframe(raw_df.head(), use_container_width=True)
        else:
            st.info("File missing.")

    with col2:
        st.markdown("### processed_data.csv")
        if proc_df is not None:
            st.write("Shape:", proc_df.shape)
            st.dataframe(proc_df.head(), use_container_width=True)
        else:
            st.info("File missing.")


# ----------------------------
# TAB 2 — EDA
# ----------------------------
with tab2:
    st.subheader("📊 Quick Visuals (Auto-cleaned data)")

    df = proc_df if proc_df is not None else raw_df

    if df is None:
        st.info("Upload CSV files into /data folder.")
    else:
        df = sanitize_dataframe(df)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X-axis", all_cols)
        with c2:
            y_col = st.selectbox("Y-axis (numeric)", [c for c in num_cols if c != x_col])
        with c3:
            color = st.selectbox("Color Group", [None] + all_cols)

        try:
            if x_col in num_cols:
                fig = px.scatter(df, x=x_col, y=y_col, color=color)
            else:
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg, x=x_col, y=y_col)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Plot error: {e}")

        st.divider()
        dist_col = st.selectbox("Distribution Column", num_cols)

        fig2 = px.histogram(df, x=dist_col, nbins=40)
        st.plotly_chart(fig2, use_container_width=True)


# ----------------------------
# TAB 3 — DEMO PREDICTION
# ----------------------------
with tab3:
    st.subheader("🤖 Predict House Price (Demo Model)")

    cA, cB, cC = st.columns(3)
    area = cA.number_input("Area (sq ft)", value=1000.0)
    locality = cB.number_input("Locality Score (0–10)", value=5.0)
    builder = cC.number_input("Builder Experience (years)", value=5.0)

    if st.button("Predict Price"):
        price = (
            3000 * area +
            100000 * (locality / 10) +
            20000 * np.log1p(builder)
        )
        st.success(f"Estimated Price: ₹ {price:,.0f}")

    st.divider()
    st.write("### Batch CSV Prediction")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            df2 = pd.read_csv(up)
            df2 = sanitize_dataframe(df2)

            st.write("Preview:", df2.head())

            df2["pred_price"] = (
                3000 * df2.iloc[:, 0].astype(float) +
                100000 * (df2.iloc[:, 1].astype(float) / 10) +
                20000 * np.log1p(df2.iloc[:, 2].astype(float))
            )

            st.download_button(
                "Download Predictions",
                df2.to_csv(index=False).encode(),
                "predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

