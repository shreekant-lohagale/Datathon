# 🏠 Housing Price Prediction – Datathon 2025  
A multi-page Streamlit dashboard for **Exploratory Data Analysis**, **Correlation Study**, **Modeling**, **Feature Importance**, and **Price Prediction** based on housing, demographic, and economic factors.

This project was built for **Datathon 2025** and provides a complete workflow from **data cleaning → EDA → modeling → predictions**, all inside a modern interactive web UI.

---

## 🚀 Features

### ✅ 1. Multi-Page Streamlit Application  
The app is divided into the following pages:

| Page | Description |
|------|-------------|
| **Home** | Overview of the project, navigation, theme loader |
| **EDA** | Scatter plots, distributions, interactive visual exploration |
| **Correlation Heatmap** | Correlation matrix to identify significant relationships |
| **Model** | Train/evaluate ML models (placeholder for now) |
| **Feature Importance** | Random Forest-based feature scoring |
| **Prediction Tool** | Single & batch prediction demo (CSV supported) |

---

## 📂 Folder Structure

```

Datathon/
│
├── app.py                      ← Main landing page
├── styles.css                  ← Premium dark theme styling
├── utils.py                    ← Shared loaders + cleaning utilities
│
├── data/
│   ├── output_Pune_builderfloor.csv
│   └── processed_data.csv
│
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Correlation.py
│   ├── 3_Model.py
│   ├── 4_Feature_Importance.py
│   └── 5_Prediction_Tool.py
│
└── README.md

````

---

## 💻 Tech Stack

- **Python**
- **Streamlit** (Multi-page app)
- **Pandas / NumPy** (Data processing)
- **Plotly** (Interactive charts)
- **Scikit-Learn** (Modeling and feature importance)
- **Custom CSS** (Dark premium UI)
- **Narwhals-safe Plotly workflow** (Prevents duplicate-column errors)

---

## ⚙️ Installation & Setup

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
````

### ✅ 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Dataset

The project uses **housing dataset from Pune (builder floors)** containing features like:

* Location
* Area
* Price
* Property age
* Amenities
* Locality score
* Builder experience
* Furnishing / facing details
* Security deposit
* Price negotiable
* New/resale

Processed dataset is stored inside:

```
/data/processed_data.csv
```

Raw dataset is stored inside:

```
/data/output_Pune_builderfloor.csv
```

---

## ✨ UI Highlights

* Dark-mode premium theme
* Glassmorphism sidebar
* Animated transitions
* Modern typography
* Rounded cards & polished layout
* Clean Plotly visualization blocks

---

## 🔥 Screenshots (Optional)

Add your screenshots here:

```
![Homepage](screenshots/home.png)
![EDA](screenshots/eda.png)
![Correlation](screenshots/correlation.png)
![Prediction Tool](screenshots/predict.png)
```

---

## ✅ Model (Coming Soon)

* Linear Regression
* Random Forest
* Gradient Boosting
* Hyperparameter tuning
* RMSE / MAE / R² evaluation

---

## 🧮 Prediction Tool

Supports:

✅ Single prediction → Users input area, locality score, builder experience
✅ Batch predictions → Upload a CSV

Outputs predicted price:

```
₹ XX,XX,XXX
```

---

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Push project to GitHub
2. Go to: [https://share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set **main file** = `app.py`
5. Deploy

OR deploy on:

* **Render**
* **Railway**
* **Vercel (Streamlit WASM)**

---

