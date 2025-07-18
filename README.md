# 🌍 Air Quality & Health Risk Predictor

An interactive **Streamlit web app** that predicts your **health risk level** (Low, Moderate, High) based on **real-time AQI, PM2.5, and personal health details** using a **Machine Learning model**.

---

## 🚀 Features

- **Live AQI Fetching**: Uses the WAQI API to get real-time AQI for selected cities.
- **ML-Based Predictions**: Predicts health risk using a trained **Random Forest** model.
- **Health Risk Score (0–100)**: Converts model probabilities into a single, interpretable score.
- **Interactive Visuals**:
  - **Gauge Meter (Speedometer)** showing your risk level.
  - **Radar Chart** comparing AQI, PM2.5, age, asthma, heart disease, and outdoor work impact.
- **City Insights**:
  - AQI trend charts.
  - Top 10 polluted cities ranking.
- **Feature Importance**: Explains which factors most affect your prediction.

---

## 🛠️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/chitralekha28/air_quality_risk_predictor.git
   cd air_quality_risk_predictor
