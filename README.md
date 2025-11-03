# 🏡 House Price Predictor — Linear & Polynomial Regression (Gradio App)

A machine learning project that predicts house prices based on key property features using **Linear Regression** and **Polynomial Regression**.  
This project is built with **scikit-learn** and **Gradio**, providing an interactive web interface for real-time predictions.

---

## 🚀 Overview

In this project, we predict the **sale price of residential homes** using features such as:
- Living area (square feet)
- Year built
- Overall material and finish quality

Two models are trained and compared:
1. **Linear Regression**
2. **Polynomial Regression (degree=2)**

The final app allows users to select a model, input property details, and instantly get predicted prices.

---

## 🧩 Features

✅ Data preprocessing and feature engineering  
✅ Model training and comparison (Linear vs Polynomial Regression)  
✅ Model serialization using `joblib`  
✅ Interactive UI built with **Gradio**  
✅ Easy to deploy on Hugging Face Spaces  

---

## 📊 Dataset

This project uses the [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), which contains detailed property data like lot size, overall quality, year built, etc.

For simplicity, we use a subset of key predictive features:
- `GrLivArea` — Living area in square feet  
- `YearBuilt` — Construction year  
- `OverallQual` — Overall material and finish quality (1–10)  

You can replace or extend this with additional features as needed.

---

## 🧠 Model Training

The training script (`train_model.py`) performs the following steps:

1. Loads and preprocesses data  
2. Splits dataset into training and testing sets  
3. Trains both Linear and Polynomial Regression models  
4. Evaluates and compares their performance using **R² score** and **RMSE**  
5. Saves trained models as `.pkl` files for later use

---

## 🖥️ Gradio App

The app (`app.py`) provides a clean, minimal interface for live predictions.

### 🧱 UI Inputs
- **Living Area (sqft)** — numeric input  
- **Year Built** — numeric input  
- **Overall Quality** — slider (1–10)  
- **Model Type** — radio button (Linear or Polynomial)  

### 🧾 Output
- Displays predicted house price (in USD)

---

## 📦 Folder Structure

