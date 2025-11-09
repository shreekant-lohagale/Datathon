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
st.title("🏠 Housing Price Prediction – Datathon 2025")
st.caption("Data preview, quick EDA, and a demo prediction UI (wire to your model later).")

# -------------------------------
# Helpers
# -------------------------------
DATA_DIR = Path("data")

def make_unique_columns(cols):
    """Return a list of column names made unique by appending .1, .2, ... when needed."""
    seen = {}
    out = []
    for c in cols:
        base = str(c)
        # normalize basic whitespace
        base = base.strip()
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}.{count}"
        while name in out:
            count += 1
            name = f"{base}.{count}"
        seen[base] = count + 1
        out.append(name)
    return out

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces, lower-case, dedupe columns; coerce obvious numerics."""
    df = df.copy()
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # make lowercase for consistency (optional; comment out if you prefer original case)
    df.columns = [c.lower() for c in df.columns]
    # ensure unique
    df.columns = make_unique_columns(df.columns)
    # try to coerce number-like strings
    for c in df.columns:
        if df[c].dtype == object:
            # remove commas in numbers like "1,234"
            try:
                df[c] = df[c].str.replace(",", "", regex=False)
            except Exception:
                pass
            # attempt numeric conversion (non-numeric remain object)
            try:
                conv = pd.to_numeric(df[c], errors="ignore")
                df[c] = conv
            except Exception:
                pass
    return df

def load_csv_safely(filename: str) -> pd.DataFrame | None:
    """Load CSV from /data, sanitize columns, and return df or None."""
    p = DATA_DIR / filename
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
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
tab1, tab2, tab3 = st.tabs(["📂 Data", "📊 EDA", "🤖 Predict"])

# ========================
# 📂 TAB 1 — DATA PREVIEW
# ========================
with tab1:
    st.subheader("Available Datasets (loaded from /data)")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**output_Pune_builderfloor.csv**")
        if raw_df is not None:
            st.write("Shape:", raw_df.shape)
            st.dataframe(raw_df.head(), use_container_width=True)
            with st.expander("Describe (all dtypes)"):
                st.write(raw_df.describe(include="all").transpose())
            with st.expander("Column names (after sanitization)"):
                st.write(list(raw_df.columns))
        else:
            st.info("Not found in /data.")

    with c2:
        st.markdown("**processed_data.csv**")
        if proc_df is not None:
            st.write("Shape:", proc_df.shape)
            st.dataframe(proc_df.head(), use_container_width=True)
            with st.expander("Describe (all dtypes)"):
                st.write(proc_df.describe(include="all").transpose())
            with st.expander("Column names (after sanitization)"):
                st.write(list(proc_df.columns))
        else:
            st.info("Not found in /data.")

# ========================
# 📊 TAB 2 — EDA
# ========================
with tab2:
    st.subheader("Quick Visuals")

    # Prefer processed if present
    df = proc_df if proc_df is not None else raw_df
    if df is None:
        st.info("No data available to visualize. Put your CSVs in the /data folder.")
    else:
        # Final safeguard: ensure uniqueness (again) right before plotting
        df = sanitize_dataframe(df)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        st.write("**Choose columns to visualize**")
        c1, c2, c3 = st.columns(3)

        with c1:
            x_col = st.selectbox("X (numeric or categorical)", options=all_cols, index=0)
        with c2:
            y_col = st.selectbox("Y (numeric)", options=num_cols if num_cols else [None])
        with c3:
            color = st.selectbox("Color (optional)", options=[None] + all_cols)

        if y_col:
            try:
                if x_col in num_cols:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color, trendline="ols")
                else:
                    agg = df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
                    fig = px.bar(agg, x=x_col, y=y_col, color=color)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot failed: {e}")

        st.divider()
        st.write("**Distribution**")
        dist_col = st.selectbox("Choose a numeric column", options=num_cols if num_cols else all_cols)
        try:
            fig2 = px.histogram(df, x=dist_col, nbins=40)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Histogram failed: {e}")

# ========================
# 🤖 TAB 3 — PREDICT (Demo)
# ========================
with tab3:
    st.subheader("Predict House Price (Demo)")

    st.write(
        "This demo uses a placeholder formula. Replace it with your trained model "
        "(e.g., load a pickle/joblib and call `model.predict` on the features)."
    )

    # If your dataset has likely features, we can default to them; else use manual inputs.
    # We'll look for common columns:
    candidate_locality = None
    candidate_area = None
    candidate_builder_exp = None

    if proc_df is not None:
        cols = set(proc_df.columns)
        for c in ["locality_score", "locality score", "localityscore"]:
            if c in cols: candidate_locality = c; break
        for c in ["area", "area.1", "sqft", "carpet_area"]:
            if c in cols: candidate_area = c; break
        for c in ["builder_experience", "builder experience", "builderexp"]:
            if c in cols: candidate_builder_exp = c; break

    cA, cB, cC = st.columns(3)
    with cA:
        area_val = st.number_input("Area (sq ft)", min_value=0.0, value=1000.0, step=10.0)
    with cB:
        locality_score = st.number_input("Locality score (0–10)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    with cC:
        builder_exp = st.number_input("Builder experience (years)", min_value=0.0, value=5.0, step=0.5)

    if st.button("Predict (demo)"):
        # Completely made-up weighting just for a demo UI
        price = (
            3000.0 * area_val +
            100000.0 * (locality_score / 10.0) +
            20000.0 * np.log1p(builder_exp)
        )
        st.success(f"Estimated price (demo): ₹ {price:,.0f}")

    st.divider()
    st.write("**Batch prediction (CSV upload)**")

    up = st.file_uploader("Upload CSV for batch demo", type=["csv"])
    if up is not None:
        try:
            udf = pd.read_csv(up)
            udf = sanitize_dataframe(udf)
            st.write("Preview:", udf.head())

            # Demo formula requires columns; let users map or fallback
            need_map = {
                "area": ["area", "area.1", "sqft", "carpet_area"],
                "locality_score": ["locality_score", "locality score", "localityscore"],
                "builder_experience": ["builder_experience", "builder experience", "builderexp"],
            }

            def find_first(df, candidates):
                for name in candidates:
                    if name in df.columns:
                        return name
                return None

            col_area = find_first(udf, need_map["area"])
            col_loc  = find_first(udf, need_map["locality_score"])
            col_bexp = find_first(udf, need_map["builder_experience"])

            if not (col_area and col_loc and col_bexp):
                st.info(
                    "Your CSV should include columns for area, locality_score, and builder_experience "
                    f"(any of {need_map}). You can rename columns in your file to match."
                )
            else:
                udf["pred_price_demo"] = (
                    3000.0 * pd.to_numeric(udf[col_area], errors="coerce").fillna(0) +
                    100000.0 * (pd.to_numeric(udf[col_loc], errors="coerce").fillna(0) / 10.0) +
                    20000.0 * np.log1p(pd.to_numeric(udf[col_bexp], errors="coerce").fillna(0))
                )

                st.download_button(
                    "Download predictions (CSV)",
                    udf.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_demo.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
