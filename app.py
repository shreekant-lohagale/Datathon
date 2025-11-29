# app.py
import os

# Turn off Plotly's Narwhals strict mode BEFORE importing plotly.express
os.environ["PLOTLY_NARWHALS_ENABLED"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# =========================
# ⚙️ PAGE CONFIG & THEME
# =========================
st.set_page_config(
    page_title="Pune Housing – Datathon 2025",
    page_icon="🏠",
    layout="wide",
)

# Custom CSS for a professional dark theme
st.markdown(
    """
    <style>
    html, body, [data-testid="block-container"] {
        background: radial-gradient(circle at top left, #111827 0, #020617 45%, #000000 100%) !important;
        color: #e5e7eb !important;
    }

    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
    }

    .tagline {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Glass cards */
    .card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 1rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin-bottom: 1rem;
    }

    .card h3, .card h4 {
        margin-top: 0;
        color: #e5e7eb;
    }

    .metric-card {
        text-align: left;
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f9fafb;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617 50%, #020617);
        border-right: 1px solid rgba(55, 65, 81, 0.9);
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.4rem;
    }

    .sidebar-sub {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        padding: 0.45rem 1.3rem;
        background: linear-gradient(90deg, #0f172a, #020617);
        color: #e5e7eb;
        font-weight: 500;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.7);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 999px !important;
        color: #9ca3af !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.9rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e5e7eb !important;
        border: 1px solid rgba(148, 163, 184, 0.6) !important;
    }

    /* Dataframe tweak */
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 📂 DATA LOADING HELPERS
# =========================
DATA_DIR = Path(__file__).parent / "data"


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
    """Clean column names & convert numeric-like strings."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.columns = make_unique_columns(df.columns)

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.replace(",", "", regex=False)
            except Exception:
                pass
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df


def load_csv_safely(filename: str):
    """Load CSV from /data and sanitize; compatible with older Pandas."""
    p = DATA_DIR / filename
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)
        df.columns = make_unique_columns(df.columns)
        df = sanitize_dataframe(df)
        return df
    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
        return None


# =========================
# 🧭 SIDEBAR – PROJECT INFO
# =========================
# with st.sidebar:
#     st.markdown('<div class="sidebar-title">🏠 Pune Housing – Datathon 2025</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="sidebar-sub">Exploring factors that drive builder-floor prices in Pune and building a simple prediction demo.</div>',
#         unsafe_allow_html=True,
#     )

#     st.markdown("---")
#     st.markdown("**Role:** Member 3 – Analyst & Storyteller")
#     st.markdown(
#         "- Initial EDA on raw data  \n"
#         "- Highlight key locations & drivers  \n"
#         "- Present the final narrative and visuals"
#     )

#     st.markdown("---")
#     st.caption("Tip: make sure `data/output_Pune_builderfloor.csv` and `data/processed_data.csv` are committed to the repo.")

# =========================
# 🎯 HEADER
# =========================
st.markdown('<div class="main-title">Housing Price Explorer – Pune</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">From raw listings to insights: quick EDA, price patterns, and a demo prediction UI for Datathon 2025.</div>',
    unsafe_allow_html=True,
)

# =========================
# 📥 LOAD DATA
# =========================
raw_df = load_csv_safely("output_Pune_builderfloor.csv")
proc_df = load_csv_safely("processed_data.csv")

# Pick a reference df for metrics
base_df = proc_df if proc_df is not None else raw_df

# Top summary cards
if base_df is not None:
    n_rows, n_cols = base_df.shape
    num_cols = base_df.select_dtypes(include=np.number).shape[1]
    cat_cols = n_cols - num_cols

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Listings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{n_rows}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Features</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{n_cols}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m3:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Numeric / Categorical</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{num_cols} / {cat_cols}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No data found in `/data`. Make sure your CSV files are present in the deployed repo.")

# =========================
# 🧱 TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📂 Data", "📊 EDA", "🤖 Predict (Demo)"])

# -------------------------
# TAB 1 – DATA PREVIEW
# -------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Raw Dataset – output_Pune_builderfloor.csv</div>', unsafe_allow_html=True)
        if raw_df is not None:
            st.write("Shape:", raw_df.shape)
            st.dataframe(raw_df.head(), use_container_width=True)
            with st.expander("Describe (numeric)"):
                st.write(raw_df.describe(include="all").transpose())
            with st.expander("Columns"):
                st.write(list(raw_df.columns))
        else:
            st.info("`output_Pune_builderfloor.csv` not found in `/data`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Processed Dataset – processed_data.csv</div>', unsafe_allow_html=True)
        if proc_df is not None:
            st.write("Shape:", proc_df.shape)
            st.dataframe(proc_df.head(), use_container_width=True)
            with st.expander("Describe (numeric)"):
                st.write(proc_df.describe(include="all").transpose())
            with st.expander("Columns"):
                st.write(list(proc_df.columns))
        else:
            st.info("`processed_data.csv` not found in `/data` (this comes from Member 1’s pipeline).")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2 – EDA
# -------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Visual Exploration</div>', unsafe_allow_html=True)

    df = proc_df if proc_df is not None else raw_df
    if df is None:
        st.info("No dataset available. Add CSV files to the `/data` folder in your repo.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        df = sanitize_dataframe(df)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X (numeric or categorical)", all_cols)
        with c2:
            y_options = [c for c in num_cols if c != x_col]
            y_col = st.selectbox("Y (numeric)", y_options)
        with c3:
            color = st.selectbox("Color (optional)", [None] + all_cols)

        st.write("")

        try:
            if x_col in num_cols:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color,
                    template="plotly_dark",
                )
            else:
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(
                    agg,
                    x=x_col,
                    y=y_col,
                    color=color if color and color in agg.columns else None,
                    template="plotly_dark",
                )
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot failed: {e}")

        st.markdown("---")
        st.markdown("#### Distribution Explorer")

        if num_cols:
            dist_col = st.selectbox("Choose a numeric column", num_cols)
            try:
                fig2 = px.histogram(
                    df,
                    x=dist_col,
                    nbins=40,
                    template="plotly_dark",
                )
                fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Histogram failed: {e}")
        else:
            st.info("No numeric columns available for histogram.")

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB 3 – PREDICT (DEMO)
# -------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Single Listing – Demo Prediction</div>', unsafe_allow_html=True)
    st.caption("This is a simple rule-based demo. In your final version, plug in the actual model from Member 2.")

    cA, cB, cC = st.columns(3)
    with cA:
        area_val = st.number_input("Area (sqft)", value=1000.0, min_value=100.0, step=50.0)
    with cB:
        locality_score = st.number_input("Locality Score (0–10)", value=6.5, min_value=0.0, max_value=10.0, step=0.5)
    with cC:
        builder_exp = st.number_input("Builder Experience (years)", value=5.0, min_value=0.0, step=1.0)

    if st.button("Predict (Demo)"):
        price = (
            3000 * area_val +
            100000 * (locality_score / 10.0) +
            20000 * np.log1p(builder_exp)
        )
        st.success(f"Estimated Price (Demo): ₹ {price:,.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Batch Prediction – CSV (Demo)</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with columns like `area`, `locality_score`, `builder_experience` to get demo predictions.")

    up = st.file_uploader("Upload CSV file", type=["csv"])
    if up:
        try:
            udf = pd.read_csv(up)
            udf.columns = make_unique_columns(udf.columns)
            udf = sanitize_dataframe(udf)
            st.write("Preview:", udf.head())

            need_map = {
                "area": ["area", "sqft", "carpet_area"],
                "locality_score": ["locality_score", "localityscore"],
                "builder_experience": ["builder_experience", "builderexp", "builder_experience_years"],
            }

            def find_col(df_, names):
                for n in names:
                    if n in df_.columns:
                        return n
                return None

            col_a = find_col(udf, need_map["area"])
            col_l = find_col(udf, need_map["locality_score"])
            col_b = find_col(udf, need_map["builder_experience"])

            if not (col_a and col_l and col_b):
                st.warning(
                    "Missing required columns. "
                    "Need something like: `area`, `locality_score`, `builder_experience`."
                )
            else:
                udf["pred_price_demo"] = (
                    3000 * udf[col_a].astype(float) +
                    100000 * (udf[col_l].astype(float) / 10.0) +
                    20000 * np.log1p(udf[col_b].astype(float))
                )

                st.success("Predictions generated. Click below to download.")
                st.download_button(
                    "Download Predictions CSV",
                    udf.to_csv(index=False).encode("utf-8"),
                    "predictions_demo.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(f"Error while processing file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
