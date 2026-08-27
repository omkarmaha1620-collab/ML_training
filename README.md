# 🌊 AI Marine Monitoring System

An **AI-based Marine Monitoring and Risk Prediction System** that combines Machine Learning, Deep Learning, Reinforcement Learning, geospatial data, and a web-based monitoring interface for intelligent maritime analysis.

---

## 🚢 Project Overview

The system integrates multiple AI techniques to:

- Monitor marine and environmental conditions
- Predict potentially hazardous marine conditions
- Forecast environmental time-series data
- Analyze vessel accident risk
- Optimize vessel routes using Reinforcement Learning
- Visualize marine information through a web-based dashboard

### Major Components

| Component | Technology | Purpose |
|---|---|---|
| 🌊 Environmental Hazard Prediction | XGBoost | High-wave / storm condition prediction |
| 📈 Environmental Forecasting | LSTM | Marine/environmental time-series forecasting |
| 🚢 Vessel Risk Prediction | Random Forest | Vessel accident-risk prediction |
| 🧠 Route Optimization | PPO | Intelligent vessel route optimization |
| 🗺️ Geographic Analysis | GeoData | Coastline, land, and port information |
| ⚙️ Backend | Python API | AI model and application integration |
| 💻 Frontend | HTML, CSS, JavaScript | Web-based monitoring dashboard |

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
