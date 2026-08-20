# 🌊 AI Marine Monitoring System

An **AI-based Marine Monitoring and Risk Prediction System** that combines Machine Learning, Deep Learning, Reinforcement Learning, geospatial data, and a web-based monitoring interface for intelligent maritime analysis.

## 🚢 Project Overview

The system integrates multiple AI techniques to monitor marine conditions, predict environmental hazards, analyze vessel risk, and optimize vessel routes.

### Major Components

* 🌊 **XGBoost** — High-wave and storm condition prediction
* 📈 **LSTM** — Marine/environmental time-series forecasting
* 🚢 **Random Forest** — Vessel accident-risk prediction
* 🧠 **Reinforcement Learning (PPO)** — Vessel route optimization
* 🗺️ **GeoData** — Coastline, land, and port geographic information
* ⚙️ **Backend** — API and AI model integration
* 💻 **Frontend** — Web-based marine monitoring dashboard

---

## 🏗️ System Architecture

```text
                         MARINE DATA
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          XGBoost            LSTM        Random Forest
             │                │                │
             ▼                ▼                ▼
       High-Wave /        Environmental    Vessel Risk
       Storm Prediction   Forecasting      Prediction
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                    Reinforcement Learning
                          PPO Model
                              │
                              ▼
                     Route Optimization
                              │
                              ▼
                         Backend API
                              │
                              ▼
                  Web Monitoring Dashboard
```

---

## 🤖 AI Modules

### 1. XGBoost — High-Wave / Storm Prediction

The XGBoost module analyzes oceanographic and environmental data to identify conditions associated with high waves and potentially hazardous marine conditions.

**Directory:**

```text
XGBoost/
```

**Main functions:**

* Environmental data preprocessing
* Feature engineering
* XGBoost model training
* High-wave prediction
* Model evaluation
* Prediction visualization

---

### 2. LSTM — Marine Forecasting

The LSTM module processes sequential/time-series data and learns temporal patterns for marine and environmental forecasting.

**Directory:**

```text
LSTM/
```

**Main functions:**

* Time-series preprocessing
* Data normalization
* LSTM model training
* Future value prediction
* Actual vs predicted visualization
* Performance evaluation

**Important files:**

```text
train_lstm.py
plot_lstm.py
lstm_model.keras
lstm_scaler.pkl
lstm_predictions.csv
lstm_results.csv
```

---

### 3. Random Forest — Vessel Risk Prediction

The Random Forest module analyzes vessel-related features and predicts vessel accident-risk conditions.

**Main files:**

```text
random_forest_model.pkl
random_forest_vessel_risk_v2.pkl
train_vessel_risk_rf_v2.py
```

The trained model can classify vessel situations according to the risk patterns learned from the training dataset.

---

### 4. Reinforcement Learning — Route Optimization

The Reinforcement Learning module uses **Proximal Policy Optimization (PPO)** to learn effective vessel route-selection strategies.

**Directory:**

```text
RL/
```

**Main functions:**

* Route environment creation
* RL training
* PPO model training
* Route evaluation
* Baseline comparison
* Trajectory analysis
* Route performance visualization

The trained PPO model is located under:

```text
RL/rl_model/
```

---

### 5. GeoData — Geographic Information

The GeoData module provides geographic information required for maritime visualization and spatial analysis.

**Directory:**

```text
GeoData/
```

It contains:

* Coastline data
* Land boundaries
* Port locations
* Shapefiles
* Geographic reference files

---

## 🌐 Web Application

The project contains a web-based monitoring interface consisting of a backend and frontend.

### Backend

```text
backend/
```

The backend provides the API/server layer responsible for connecting the AI models and application interface.

### Frontend

```text
frontend/
```

The frontend provides the user-facing monitoring dashboard and route visualization interface.

---

## 📁 Project Structure

```text
ML_training/
│
├── backend/
│   ├── main.py
│   └── .gitignore
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── monitoring.html
│   ├── monitoring.js
│   ├── monitoring.css
│   ├── route.html
│   ├── route.js
│   └── route.css
│
├── GeoData/
│   ├── coastline/
│   ├── land/
│   └── ports/
│
├── LSTM/
│   ├── train_lstm.py
│   ├── plot_lstm.py
│   ├── lstm_model.keras
│   ├── lstm_scaler.pkl
│   ├── lstm_predictions.csv
│   └── lstm_results.csv
│
├── RL/
│   ├── train_rl.py
│   ├── route_environment.py
│   ├── evaluate_rl.py
│   ├── analyze_*.py
│   ├── create_*.py
│   └── rl_model/
│
├── XGBoost/
│   ├── train_storm_xgboost.py
│   ├── plot_xgboost.py
│   ├── download_ndbc_dataset.py
│   ├── data/
│   └── *.pkl
│
├── evaluate_models.py
├── train_models.py
├── train_vessel_risk_rf_v2.py
├── random_forest_model.pkl
├── random_forest_vessel_risk_v2.pkl
├── xgboost_model.pkl
├── evaluation_results.csv
├── model_results.csv
└── .gitignore
```

---

## 🛠️ Technologies Used

| Category               | Technologies           |
| ---------------------- | ---------------------- |
| Programming            | Python, JavaScript     |
| Machine Learning       | Random Forest, XGBoost |
| Deep Learning          | LSTM, TensorFlow/Keras |
| Reinforcement Learning | PPO                    |
| Data Processing        | Pandas, NumPy          |
| Visualization          | Matplotlib             |
| Geospatial Data        | Shapefiles, GeoData    |
| Backend                | Python API             |
| Frontend               | HTML, CSS, JavaScript  |
| Version Control        | Git, GitHub            |

---

## 🎯 Objectives

1. Develop an AI-based system for intelligent marine monitoring.
2. Predict potentially hazardous marine conditions.
3. Forecast environmental conditions using sequential data.
4. Analyze vessel-related accident risk.
5. Optimize vessel routes using reinforcement learning.
6. Integrate multiple AI models into a unified monitoring platform.
7. Provide an interactive web-based interface for maritime analysis.

---

## 🔄 System Workflow

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
AI Model Training
       ↓
Model Evaluation
       ↓
Prediction / Route Optimization
       ↓
Backend Integration
       ↓
Web Dashboard
       ↓
Marine Monitoring & Decision Support
```

---

## 📊 Model Evaluation

The project includes evaluation and analysis outputs for the implemented AI models.

Evaluation includes metrics and results such as:

* MAE
* RMSE
* R²
* MAPE
* Prediction results
* Feature importance
* Route comparison
* Trajectory analysis
* Visualization results

---

## 📌 Large Dataset Handling

Some raw and training datasets are intentionally excluded from this GitHub repository because they exceed GitHub's standard individual-file size limit.

The following large RL datasets remain available in the local development environment but are excluded from Git tracking:

```text
RL/zenodo_raw_filtered.csv
RL/rl_training_dataset.csv
RL/rl_training_dataset_v2.csv
```

The source code required for processing, training, evaluation, and analysis is included in this repository.

---

## 🚀 Future Enhancements

* Real-time AIS data integration
* Live vessel tracking
* Real-time weather and ocean data
* Collision-risk prediction
* Dynamic route replanning
* Real-time maritime alerts
* Cloud deployment
* Model monitoring
* Mobile-friendly dashboard
* Integration with additional maritime data sources

---

## 👨‍💻 Project

**AI Marine Monitoring System**

An academic Artificial Intelligence and Machine Learning project for:

**Marine Monitoring → Environmental Prediction → Vessel Risk Analysis → Intelligent Route Optimization**

---

## 📄 License

This project is developed for academic and educational purposes.
