# main.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load sample dataset for trend and importance display
@st.cache_data
def load_data():
    df = pd.read_csv("data/air_quality_data.csv")
    df = df.dropna(subset=["AQI", "Datetime"])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

data = load_data()

# ----------------- UI -----------------

st.title("🌍 Air Quality & Health Risk Predictor")
st.markdown("Get your personalized health risk based on your city's air quality and your health conditions.")

# Select City
city = st.selectbox("🏙️ Select your city", sorted(data["City"].unique()))
city_data = data[data["City"] == city]

# Show AQI Info
if not city_data.empty:
    current_aqi = city_data.sort_values("Datetime", ascending=False).iloc[0]["AQI"]
    st.success(f"📍 Current AQI in {city}: {int(current_aqi)}")

    st.markdown("📈 **AQI Trend in " + city + "**")
    recent = city_data.sort_values("Datetime").tail(30)
    st.line_chart(recent.set_index("Datetime")["AQI"])

    top_cities = data.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    st.markdown("🏆 **Top 10 Cities by AQI**")
    st.bar_chart(top_cities)

    # AQI status label
    if current_aqi <= 50:
        aqi_status = "🟢 Good"
    elif current_aqi <= 100:
        aqi_status = "🟡 Moderate"
    elif current_aqi <= 150:
        aqi_status = "🟠 Unhealthy for Sensitive Groups"
    else:
        aqi_status = "🔴 Unhealthy"
    st.markdown(f"**Air Quality Level**: {aqi_status}")
else:
    st.error("No data available for the selected city.")

# ----------------- User Health Inputs -----------------
st.markdown("### 🧍 Your Health Conditions")

breathing_issue = st.checkbox("Do you have breathing issues?")
heart_problem = st.checkbox("Any heart-related problem?")
diabetes = st.checkbox("Are you diabetic?")
outdoor_time = st.slider("How many hours do you spend outdoors daily?", 0, 24, 2)

# Convert to binary
has_breathing = 1 if breathing_issue else 0
has_heart = 1 if heart_problem else 0
has_diabetes = 1 if diabetes else 0

# ----------------- Enhanced Prediction -----------------
if st.button("🔍 Predict Health Risk"):
    input_data = pd.DataFrame([[
        current_aqi, pm25, age, has_breathing, has_heart, has_diabetes, outdoor_time
    ]], columns=["AQI", "PM2.5", "Age", "BreathingIssue", "HeartDisease", "Diabetes", "OutdoorHours"])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data).max()
    risk_label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("🧠 Predicted Health Risk Level: " + risk_label)
    st.markdown(f"🔍 **Prediction Confidence**: `{round(confidence * 100, 2)}%`")

    if risk_label.lower() == "high":
        st.error("📋 **Advice**: High risk. Avoid outdoor activity. Consult a doctor if symptoms worsen.")
    elif risk_label.lower() == "moderate":
        st.warning("📋 **Advice**: Moderate risk. Stay indoors more often and wear a mask.")
    else:
        st.success("📋 **Advice**: Low risk. Maintain precautions and hydrate well.")

    # ----------------- Feature Importance -----------------
    st.markdown("### 📊 Feature Importance")
    importances = model.feature_importances_
    feature_names = ["AQI", "PM2.5", "Age", "BreathingIssue", "HeartDisease", "Diabetes", "OutdoorHours"]
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="coolwarm", ax=ax)
    st.pyplot(fig)
