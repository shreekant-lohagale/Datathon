# 🏠 Housing Price Prediction – Datathon 2025

An interactive, multi-page **Streamlit Dashboard** for:

* 🧹 Data Cleaning
* 📊 Exploratory Data Analysis
* 🔗 Correlation Study
* 🤖 Model Integration (coming soon)
* 🔥 Price Prediction (single + batch)

Built for **Datathon 2025**, this project provides a complete workflow from loading raw housing datasets to performing EDA and generating demo predictions — all with a clean, modern UI.

---

## 🚀 Features

### ✅ 1. Clean & Modern Multi-Page Streamlit App

Pages include:

| Page                   | Function                                                |
| ---------------------- | ------------------------------------------------------- |
| **Home**               | Project intro, navigation, dark theme                   |
| **EDA**                | Scatter plots & distributions (no statsmodels required) |
| **Correlation Matrix** | Visual relationship mapping (optional)                  |
| **Model Training**     | Placeholder for your ML models                          |
| **Feature Importance** | Random Forest-based feature scoring                     |
| **Prediction Tool**    | Single + batch prediction UI                            |

---

## 📂 Folder Structure

```
Datathon/
│
├── app.py                      ← Main app (merged & fixed)
├── styles.css                  ← Dark theme + UI enhancements
├── utils.py                    ← Shared preprocessing utilities
│
├── data/                       ← Place CSV files here
│   ├── output_Pune_builderfloor.csv
│   └── processed_data.csv
│
├── pages/                      ← Modular Streamlit pages
│   ├── 1_EDA.py
│   ├── 2_Correlation.py
│   ├── 3_Model.py
│   ├── 4_Feature_Importance.py
│   └── 5_Prediction_Tool.py
│
├── models/
│   └── model.ipynb             ← Your ML notebook
│
└── README.md
```

---

## 💻 Tech Stack

* **Python 3.10+**
* **Streamlit** – Multi-page web UI
* **Pandas & NumPy** – Data cleaning and processing
* **Plotly** – Interactive charts (scatter, histograms, bar)
* **Scikit-Learn** – Training + feature importance (future pages)
* **Custom CSS** – Full dark-mode premium UI
* **Narwhals-safe Plotly** (ensures no duplicate-column errors)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Then activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run App

```bash
streamlit run app.py
```

---

## 📊 Dataset Overview

This project uses real estate data from **Pune builder-floor housing**.

### Raw Dataset

`data/output_Pune_builderfloor.csv`

### Processed Dataset

`data/processed_data.csv`

Key features include:

* Area (sqft)
* Price
* Locality score
* Builder experience
* Property age
* Furnishing status
* Facing direction
* Total floors / floor number
* Price negotiable
* Security deposit
* Amenities

---

## ✨ UI Highlights

* 🌑 Full **dark mode** aesthetic
* 🎛️ Clean card-based layouts
* 📊 Modern Plotly visuals
* 🔄 Smooth interactions
* 🧼 Auto-sanitized CSV loading (duplicate columns fixed)
* 📱 Responsive layout

---

## 🔍 EDA Features

* Scatter plots
* Bar plots (for categorical X)
* Histograms
* Auto-cleaned columns
* No dependency on `statsmodels`

---

## 🤖 Prediction Tool (Demo)

Supports:

### ✔ Single Prediction

Input:

* Area (sqft)
* Locality score (0–10)
* Builder experience (years)

Outputs an estimated price:

```
₹ 12,34,567
```

### ✔ Batch Prediction

Upload a CSV with the required fields.
App automatically detects:

* `area`
* `locality_score`
* `builder_experience`

Outputs a downloadable prediction CSV.

---