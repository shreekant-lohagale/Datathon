# utils.py
import os
os.environ["PLOTLY_NARWHALS_ENABLED"] = "0"

import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")

def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        base = str(c).strip().lower()
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}.{count}"
        while name in out:
            count += 1
            name = f"{base}.{count}"
        seen[base] = count + 1
        out.append(name)
    return out


def sanitize_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.columns = make_unique_columns(df.columns)

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.replace(",", "", regex=False)
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = sanitize_dataframe(df)
        return df
    except Exception:
        return None
