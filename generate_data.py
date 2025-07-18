import pandas as pd
import numpy as np
import random

def generate_row():
    aqi = random.randint(50, 500)
    pm25 = random.randint(20, 300)
    age = random.randint(10, 90)
    asthma = random.choice([0, 1])
    heart = random.choice([0, 1])
    outdoor = random.choice([0, 1])

    # Rule-based risk scoring
    score = 0
    if aqi > 300: score += 3
    elif aqi > 200: score += 2
    elif aqi > 100: score += 1
    if pm25 > 150: score += 2
    if age > 60: score += 1
    if asthma: score += 2
    if heart: score += 2
    if outdoor: score += 1

    if score >= 7:
        risk = "High"
    elif score >= 4:
        risk = "Medium"
    else:
        risk = "Low"

    return [aqi, pm25, age, asthma, heart, outdoor, risk]

# Generate data
data = [generate_row() for _ in range(1000)]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["AQI", "PM2.5", "Age", "Asthma", "Heart", "Outdoor", "Risk_Level"])

# Save to CSV
df.to_csv("ml_risk_data.csv", index=False)
print("✅ Dataset created as ml_risk_data.csv")
