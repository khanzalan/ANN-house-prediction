# 🏠 California Housing Price Prediction — ANN Regression

A deep learning regression project using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras to predict median house values in California.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikit-learn)

---

## 📌 Project Overview

This project trains a fully connected ANN to predict the **median house value** (in $100,000s) for California census block groups. It demonstrates a clean end-to-end ML pipeline: data loading → preprocessing → model building → training → evaluation → visualization.

---

## 📊 Dataset

**California Housing Dataset** (via `sklearn.datasets.fetch_california_housing`)

- **Samples:** 20,640
- **Features:** 8 numeric features (income, house age, rooms, population, lat/lon, etc.)
- **Target:** Median house value in $100,000s
- **Source:** 1990 US Census

---

## 🏗️ Model Architecture

```
Input(8) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
```

| Layer | Units | Activation | Parameters |
|-------|-------|------------|------------|
| Dense | 64    | ReLU       | 576        |
| Dense | 32    | ReLU       | 2,080      |
| Dense | 1     | Linear     | 33         |
| **Total** | | | **2,689** |

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Epochs:** 50 · **Batch size:** 32 · **Validation split:** 10%

---

## 📈 Results

| Metric | Score |
|--------|-------|
| MAE    | ~0.32 (~$32,000) |
| R²     | ~0.76 |

Training and validation curves, plus an Actual vs Predicted scatter plot, are saved as `results.png` after running the notebook.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/california-housing-ann.git
cd california-housing-ann
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Housing_Price_Prediction_ANN.ipynb
```

---

## 📁 Project Structure

```
california-housing-ann/
├── Housing_Price_Prediction_ANN.ipynb   # Main notebook
├── requirements.txt                      # Dependencies
├── results.png                           # Generated after running notebook
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — ANN model
- **scikit-learn** — Dataset, preprocessing, metrics
- **NumPy** — Numerical operations
- **Matplotlib** — Visualization
