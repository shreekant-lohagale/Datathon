# Housing Price Prediction

A machine learning project to predict housing prices in Pune, built for the Datathon 2025. This application uses a Random Forest Regressor model and serves predictions via a Flask API to a simple HTML/CSS/JS frontend.



## 📋 Project Overview

This project is a solution for the "Housing Price Prediction" problem. It analyzes property features, location scores, and amenities from a dataset of Pune builder floors to predict their market price.

The workflow is as follows:
1.  **Data Cleaning & EDA:** The raw CSV data is loaded, cleaned, and analyzed.
2.  **Feature Engineering:** New features like `amenities_score` and `price_per_sqft` are created to improve model accuracy.
3.  **Model Training:** A `RandomForestRegressor` is trained on the processed data.
4.  **Model Deployment:** The trained model is saved and served via a Flask REST API.
5.  **Frontend:** A simple HTML/JS/CSS interface allows users to input property details and receive a price prediction.

## ✨ Features

* **Real-time Price Prediction:** A web form to input property details and get an estimated price.
* **Data-Driven Model:** Trained on the `output_Pune_builderfloor.csv` dataset.
* **REST API Backend:** A clean Flask backend that serves the model as an API.
* **Simple UI:** A no-frills HTML/CSS/JS interface to interact with the model.

## 💻 Tech Stack

### Backend
* **Python 3.x**
* **Flask:** For the web server and REST API.
* **Scikit-learn:** For building and training the `RandomForestRegressor`.
* **Pandas:** For all data loading, cleaning, and feature engineering.
* **Joblib:** For saving and loading the trained model assets.

### Frontend
* **HTML5:** For the main structure of the web app.
* **CSS3:** For all styling.
* **JavaScript (ES6+):** For populating dropdowns and handling API calls (`fetch`).

## 🚀 How to Run

1.  **Clone this repository.**

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install Flask pandas scikit-learn joblib
    ```

4.  **Prepare the Model Assets:**
    * First, you must run the script to train and save the model:
    ```bash
    python save_assets.py
    ```
    * This will create the necessary `.joblib` files (`rf_model.joblib`, `model_columns.joblib`, etc.).

5.  **Run the Flask application:**
    ```bash
    python app.py
    ```

6.  **Open your browser:**
    * Navigate to `http://127.0.0.1:5000/` to use the app.
