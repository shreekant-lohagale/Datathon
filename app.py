# app.py
import os
# Turn off Plotly's Narwhals strict mode BEFORE importing plotly.express
os.environ["PLOTLY_NARWHALS_ENABLED"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Housing Price Explorer", page_icon="🏠", layout="wide")
st.title(" Housing Price Prediction – Datathon 2025")
st.caption("Data preview, quick EDA, and a demo prediction UI (wire to your model later).")

# -------------------------------
# Helpers
# -------------------------------
DATA_DIR = Path("data")

def make_unique_columns(cols):
    """Ensure column names are unique by appending .1, .2 etc."""
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

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, normalize, dedupe columns, convert numeric-like strings."""
    df = df.copy()

    # Clean column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.columns = make_unique_columns(df.columns)

    # Try numeric conversions
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

def load_csv_safely(filename: str) -> pd.DataFrame | None:
    """Load CSV and sanitize; compatible with older Pandas."""
    p = DATA_DIR / filename
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)   # <-- FIXED (no mangle_dupe_cols)
        df.columns = make_unique_columns(df.columns)
        df = sanitize_dataframe(df)
        return df

    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
        return None

# -------------------------------
# Load datasets
# -------------------------------
raw_df  = load_csv_safely("output_Pune_builderfloor.csv")
proc_df = load_csv_safely("processed_data.csv")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Data", "EDA", "Predict"])

# ========================
# 📂 TAB 1 — DATA PREVIEW
# ========================
with tab1:
    st.subheader("Available Datasets (loaded from /data)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### output_Pune_builderfloor.csv")
        if raw_df is not None:
            st.write("Shape:", raw_df.shape)
            st.dataframe(raw_df.head(), use_container_width=True)
            with st.expander("Describe"):
                st.write(raw_df.describe(include="all").transpose())
            with st.expander("Columns"):
                st.write(list(raw_df.columns))
        else:
            st.info("Not found in /data.")

    with col2:
        st.markdown("### processed_data.csv")
        if proc_df is not None:
            st.write("Shape:", proc_df.shape)
            st.dataframe(proc_df.head(), use_container_width=True)
            with st.expander("Describe"):
                st.write(proc_df.describe(include="all").transpose())
            with st.expander("Columns"):
                st.write(list(proc_df.columns))
        else:
            st.info("Not found in /data.")

# ========================
# 📊 TAB 2 — EDA
# ========================
with tab2:
    st.subheader("Quick Visuals")

    df = proc_df if proc_df is not None else raw_df
    if df is None:
        st.info("Upload CSV files into /data folder.")
    else:
        df = sanitize_dataframe(df)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        st.write("### Choose columns")
        c1, c2, c3 = st.columns(3)

        with c1:
            x_col = st.selectbox("X (numeric or categorical)", all_cols)

        with c2:
            y_options = [c for c in num_cols if c != x_col]
            y_col = st.selectbox("Y (numeric)", y_options)

        with c3:
            color = st.selectbox("Color (optional)", [None] + all_cols)

        # -------- Plot (NO TRENDLINE)
        try:
            if x_col in num_cols:
                fig = px.scatter(df, x=x_col, y=y_col, color=color)
            else:
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg, x=x_col, y=y_col, color=color)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Plot failed: {e}")

        st.divider()
        st.write("### Distribution")

        dist_col = st.selectbox("Numeric column", num_cols)
        try:
            fig2 = px.histogram(df, x=dist_col, nbins=40)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Histogram failed: {e}")

# ========================
# 🤖 TAB 3 — PREDICT (DEMO)
# ========================
with tab3:
    st.subheader("Predict House Price (Demo)")

    cA, cB, cC = st.columns(3)
    with cA:
        area_val = st.number_input("Area (sq ft)", value=1000.0)
    with cB:
        locality_score = st.number_input("Locality Score (0–10)", value=6.5)
    with cC:
        builder_exp = st.number_input("Builder Experience (years)", value=5.0)

    if st.button("Predict (demo)"):
        price = (
            3000 * area_val +
            100000 * (locality_score / 10) +
            20000 * np.log1p(builder_exp)
        )
        st.success(f"Estimated Price: ₹ {price:,.0f}")

    st.divider()
    st.write("### Batch Prediction")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            udf = pd.read_csv(up)
            udf.columns = make_unique_columns(udf.columns)
            udf = sanitize_dataframe(udf)
            st.write("Preview:", udf.head())

            need_map = {
                "area": ["area", "sqft", "carpet_area"],
                "locality_score": ["locality_score", "localityscore"],
                "builder_experience": ["builder_experience", "builderexp"],
            }

            def find(df, names):
                for n in names:
                    if n in df.columns:
                        return n
                return None

            col_a = find(udf, need_map["area"])
            col_l = find(udf, need_map["locality_score"])
            col_b = find(udf, need_map["builder_experience"])

            if not (col_a and col_l and col_b):
                st.warning("Missing required columns in CSV.")
            else:
                udf["pred_price_demo"] = (
                    3000 * udf[col_a].astype(float) +
                    100000 * (udf[col_l].astype(float) / 10) +
                    20000 * np.log1p(udf[col_b].astype(float))
                )

                st.download_button(
                    "Download Predictions",
                    udf.to_csv(index=False).encode(),
                    "predictions_demo.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")
