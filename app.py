import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.graph_objects as go

# --- Load ML model, scaler, and label encoder ---
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# --- Fetch Live AQI ---
def get_live_aqi(city):
    API_TOKEN = "01c9cc7a2575250d9b09e469449d69b502cdd509"  # Replace with your WAQI token
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            return data["data"]["aqi"], data["data"].get("iaqi", {}).get("pm25", {}).get("v", None)
        else:
            return None, None
    except:
        return None, None

# --- Load Local AQI Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/air_quality_data.csv")
    df = df.dropna(subset=['AQI', 'Datetime'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df[['City', 'Datetime', 'AQI']]
    latest = df.sort_values(by='Datetime', ascending=False).groupby('City').first().reset_index()
    return latest, df

latest_data, full_data = load_data()

# --- App UI ---
st.set_page_config(page_title="Air Quality & Health Risk Predictor", page_icon="🌫️")
st.title("🌍 Air Quality & Health Risk Predictor")
st.markdown("Get your personalized health risk based on your city's air quality and your health conditions.")

# --- City Selection ---
city = st.selectbox("🏙️ Select your city", sorted(latest_data['City'].unique()))
aqi_value, live_pm25 = get_live_aqi(city)

if aqi_value is None:
    aqi_value = latest_data[latest_data['City'] == city]['AQI'].values[0]
    st.warning("⚠️ Live AQI not available, using latest dataset value.")
else:
    st.success("✅ Live AQI fetched from API.")

st.write(f"### 📍 Current AQI in {city}: `{int(aqi_value)}`")

# --- AQI Trend Chart ---
st.subheader(f"📈 AQI Trend in {city}")
city_data = full_data[full_data['City'] == city]
daily = city_data.groupby(city_data['Datetime'].dt.date)['AQI'].mean().reset_index()
daily.columns = ['Date', 'Average AQI']
st.line_chart(daily.set_index('Date'))

# --- Top Cities Chart ---
st.subheader("🏆 Top 10 Cities by AQI")
top_cities = latest_data[['City', 'AQI']].sort_values(by='AQI', ascending=False).head(10)
st.bar_chart(top_cities.set_index('City'))

# --- AQI Level Advice ---
def interpret_aqi(aqi):
    if aqi <= 50:
        return "🟢 Good", "Enjoy your day!"
    elif aqi <= 100:
        return "🟡 Satisfactory", "Safe for most, avoid dusty areas."
    elif aqi <= 200:
        return "🟠 Moderate", "May affect sensitive individuals."
    elif aqi <= 300:
        return "🟣 Poor", "Limit outdoor activities, consider mask."
    elif aqi <= 400:
        return "🔴 Very Poor", "Avoid going outdoors, use purifier."
    else:
        return "⚫ Severe", "Stay indoors, wear N95 mask."

level, advice_env = interpret_aqi(aqi_value)
st.write(f"**Air Quality Level:** {level}")
st.info(advice_env)

# --- User Inputs ---
st.subheader("🧍 Your Health Details")
age = st.slider("Your Age", 1, 100, 25)
asthma = st.checkbox("I have asthma")
heart = st.checkbox("I have a heart condition")
outdoor = st.checkbox("I work outdoors daily")
pm25 = st.slider("PM2.5 value (approximate)", 0, 500, int(live_pm25) if live_pm25 else 120)

# --- Predict Risk ---
if st.button("Check My Health Risk"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        "AQI": aqi_value,
        "PM2.5": pm25,
        "Age": age,
        "BreathingIssue": int(asthma),
        "HeartDisease": int(heart),
        "OutdoorHours": 8 if outdoor else 0
    }])

    # Scale continuous features
    cols_to_scale = ['AQI', 'PM2.5', 'Age', 'OutdoorHours']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale].astype('float64'))

    # Predict
    prediction = model.predict(input_df)[0]
    risk_label = label_encoder.inverse_transform([prediction])[0]

    # Probability -> Risk Score (0–100)
    proba = model.predict_proba(input_df)[0][1]
    risk_score = int(proba * 100)

    # --- Risk Gauge Meter ---
    st.subheader("📊 Health Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Score (0–100)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if risk_score > 70 else "orange" if risk_score > 40 else "green"},
               'steps': [
                   {'range': [0, 40], 'color': "lightgreen"},
                   {'range': [40, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "pink"}
               ]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Radar Chart for Factors ---
    st.subheader("🕸️ Risk Factor Profile")
    factors = ["AQI", "PM2.5", "Age", "Asthma", "Heart", "Outdoor Hours"]
    normalized_vals = [
        aqi_value/500*100,
        pm25/500*100,
        age/100*100,
        int(asthma)*100,
        int(heart)*100,
        (8 if outdoor else 0)/8*100
    ]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_vals,
        theta=factors,
        fill='toself',
        name='Your Risk Profile',
        line_color='crimson'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- Final Risk Output ---
    st.success(f"🧠 Predicted Health Risk Level: **{risk_label}**")

    # Confidence Table
    proba_all = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        "Risk Level": label_encoder.inverse_transform(np.arange(len(proba_all))),
        "Confidence (%)": np.round(proba_all * 100, 2)
    }).sort_values(by="Confidence (%)", ascending=False)
    st.subheader("🔍 Prediction Confidence")
    st.dataframe(prob_df.set_index("Risk Level"))

    # Advice
    advice_dict = {
        "Low Risk": "You're safe. Stay hydrated and enjoy your day.",
        "Moderate Risk": "Limit outdoor activities. If sensitive, wear a mask.",
        "High Risk": "Stay indoors. Use air purifiers and wear N95 mask if outside."
    }
    st.warning(f"📋 Advice: {advice_dict.get(risk_label, 'Stay cautious and monitor AQI updates.')}")

    # --- Feature Importance ---
    if hasattr(model, "feature_importances_"):
        st.subheader("📊 Feature Importance")
        features = ['AQI', 'PM2.5', 'Age', 'BreathingIssue', 'HeartDisease', 'OutdoorHours']
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')

        fig, ax = plt.subplots()
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        ax.set_xlabel("Importance Score")
        ax.set_title("Features Influencing Prediction")
        st.pyplot(fig)
