# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Generate Synthetic Data ---
np.random.seed(42)
num_samples = 5000

data = {
    "AQI": np.random.randint(30, 500, num_samples),
    "PM2.5": np.random.randint(10, 400, num_samples),
    "Age": np.random.randint(1, 90, num_samples),
    "BreathingIssue": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    "HeartDisease": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
    "OutdoorHours": np.random.randint(0, 10, num_samples),
}

df = pd.DataFrame(data)

# --- Assign Risk Labels based on thresholds ---
def assign_risk(row):
    score = (
        (row['AQI'] / 100) +
        (row['PM2.5'] / 80) +
        (row['Age'] / 60) +
        (row['BreathingIssue'] * 2) +
        (row['HeartDisease'] * 2) +
        (row['OutdoorHours'] / 5)
    )
    if score > 10:
        return "High Risk"
    elif score > 5:
        return "Moderate Risk"
    else:
        return "Low Risk"

df['RiskLabel'] = df.apply(assign_risk, axis=1)

# --- Features and Target ---
features = ['AQI', 'PM2.5', 'Age', 'BreathingIssue', 'HeartDisease', 'OutdoorHours']
target = 'RiskLabel'

X = df[features].copy()  # Deep copy to avoid SettingWithCopyWarning
y = df[target]

# --- Scale Continuous Features ---
scaler = StandardScaler()
cols_to_scale = ['AQI', 'PM2.5', 'Age', 'OutdoorHours']
X[cols_to_scale] = X[cols_to_scale].astype('float64')
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# --- Encode Target Labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
print("\nModel Performance:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Save Model and Encoders ---
joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model trained and saved successfully as 'model.pkl'!")
