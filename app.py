# ==============================
# FINAL PRODUCTION APP.PY
# Housing Price Explorer – Datathon 2025
# Fully redesigned premium UI theme
# ==============================

import os
os.environ["PLOTLY_NARWHALS_ENABLED"] = "0"  # Disable issue-causing narwhals mode

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ----------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------
st.set_page_config(
    page_title="🏠 Housing Price Explorer",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS Theme
st.markdown("""
<style>

body {
    background-color: #0d1117;
}

[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Card-like sections */
.block-container {
    padding-top: 2rem;
}

section {
    background: #161b22;
    padding: 1.5rem;
    border-radius: 14px;
    border: 1px solid #30363d;
    margin-bottom: 1.5rem;
}

/* Titles */
h1, h2, h3 {
    color: #e6edf3 !important;
    font-weight: 700;
}

/* Metric boxes */
.metric-box {
    background: #21262d;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
DATA_DIR = Path("data")

def make_unique_columns(cols):
    """Ensure column names are unique (col, col.1, col.2...)."""
    seen = {}
    new_cols = []
    for c in cols:
        base = str(c).strip()
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}.{count}"
        while name in new_cols:
            count += 1
            name = f"{base}.{count}"
        seen[base] = count + 1
        new_cols.append(name)
    return new_cols


def sanitize_df(df):
    """Clean columns, convert numeric strings, fix duplicates."""
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    df.columns = make_unique_columns(df.columns)

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.replace(",", "")
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except:
                pass
    return df


def load_csv(filename):
    """Load CSV if exists safely."""
    f = DATA_DIR / filename
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return sanitize_df(df)


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
raw_df = load_csv("output_Pune_builderfloor.csv")
proc_df = load_csv("processed_data.csv")


# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.title("🏠 Housing Price Explorer – Datathon 2025")
st.caption("A clean UI to preview data, explore EDA, and try demo predictions.")


# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📂 Data", "📊 EDA", "🤖 Predict"])


# ==========================================================
# 📂 TAB 1 — DATA
# ==========================================================
with tab1:
    st.subheader("Available Datasets")

    col1, col2 = st.columns(2)

    # RAW DATA
    with col1:
        st.markdown("### 🟦 output_Pune_builderfloor.csv")
        if raw_df is not None:
            st.markdown(f"<div class='metric-box'>Rows: {raw_df.shape[0]} | Columns: {raw_df.shape[1]}</div>", unsafe_allow_html=True)
            st.dataframe(raw_df.head(), use_container_width=True)

            with st.expander("Describe"):
                st.write(raw_df.describe(include="all").transpose())

            with st.expander("Columns"):
                st.write(list(raw_df.columns))

        else:
            st.warning("File not found in /data folder.")

    # PROCESSED DATA
    with col2:
        st.markdown("### 🟩 processed_data.csv")
        if proc_df is not None:
            st.markdown(f"<div class='metric-box'>Rows: {proc_df.shape[0]} | Columns: {proc_df.shape[1]}</div>", unsafe_allow_html=True)
            st.dataframe(proc_df.head(), use_container_width=True)

            with st.expander("Describe"):
                st.write(proc_df.describe(include="all").transpose())

            with st.expander("Columns"):
                st.write(list(proc_df.columns))

        else:
            st.warning("File not found in /data folder.")



# ==========================================================
# 📊 TAB 2 — EDA
# ==========================================================
with tab2:
    st.subheader("Quick Visual EDA")

    df = proc_df if proc_df is not None else raw_df

    if df is None:
        st.info("Upload datasets into /data folder to begin.")
    else:
        df = sanitize_df(df)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        c1, c2, c3 = st.columns(3)

        with c1:
            x_col = st.selectbox("X-axis", all_cols)

        with c2:
            y_col = st.selectbox("Y-axis (numeric)", num_cols)

        with c3:
            color = st.selectbox("Color", [None] + all_cols)

        # Plot
        try:
            if x_col in num_cols:
                fig = px.scatter(df, x=x_col, y=y_col, color=color)
            else:
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg, x=x_col, y=y_col, color=color)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Plot error: {e}")

        st.divider()

        # Distribution
        dist_col = st.selectbox("Distribution Column", num_cols)
        try:
            fig2 = px.histogram(df, x=dist_col, nbins=40)
            st.plotly_chart(fig2, use_container_width=True)
        except:
            st.error("Histogram error.")



# ==========================================================
# 🤖 TAB 3 — PREDICT
# ==========================================================
with tab3:
    st.subheader("Demo Price Prediction")

    cA, cB, cC = st.columns(3)

    area = cA.number_input("Area (sq ft)", value=1000.0)
    loc = cB.number_input("Locality Score (0–10)", value=6.5)
    exp = cC.number_input("Builder Experience (years)", value=5.0)

    if st.button("Predict Price"):
        price = (
            3000 * area +
            100000 * (loc / 10) +
            20000 * np.log1p(exp)
        )
        st.success(f"Estimated Price: ₹ {price:,.0f}")

    st.divider()

    st.write("### Batch Prediction (CSV Upload)")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        try:
            dfu = pd.read_csv(file)
            dfu = sanitize_df(dfu)

            st.write("Preview:", dfu.head())

            # Column mapping
            def find(df, keys):
                for k in keys:
                    if k in df.columns:
                        return k
                return None

            colA = find(dfu, ["area", "sqft", "sq_ft"])
            colL = find(dfu, ["locality_score", "locality"])
            colE = find(dfu, ["builder_experience", "experience"])

            if not (colA and colL and colE):
                st.error("CSV missing required columns.")
            else:
                dfu["pred_price_demo"] = (
                    3000 * dfu[colA].astype(float) +
                    100000 * (dfu[colL].astype(float) / 10) +
                    20000 * np.log1p(dfu[colE].astype(float))
                )

                st.download_button(
                    "Download Predictions",
                    dfu.to_csv(index=False).encode(),
                    "predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")
