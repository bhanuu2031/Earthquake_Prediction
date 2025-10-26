import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_PATH = "data/processed/final_dataset.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

np.random.seed(42)

# Number of sensor nodes & samples per node
N_NODES = 20
SAMPLES_PER_NODE = 7500

start_time = datetime(2025, 1, 1)
rows = []

for node in range(1, N_NODES + 1):
    base_vibration = np.random.uniform(0.2, 0.8)
    base_tilt = np.random.uniform(0.1, 0.6)
    base_moisture = np.random.uniform(0.3, 0.7)
    base_rain = np.random.uniform(0.1, 0.5)

    for t in range(SAMPLES_PER_NODE):
        timestamp = start_time + timedelta(minutes=t * 5)
        vibration = base_vibration + np.random.normal(0, 0.15)
        tilt = base_tilt + np.random.normal(0, 0.1)
        moisture = base_moisture + np.random.normal(0, 0.05)
        rainfall = base_rain + np.random.normal(0, 0.1)

        # Clip values between 0–1
        vibration = np.clip(vibration, 0, 1)
        tilt = np.clip(tilt, 0, 1)
        moisture = np.clip(moisture, 0, 1)
        rainfall = np.clip(rainfall, 0, 1)

        # --- probabilistic alert generation ---
        risk_score = (
            0.5 * vibration
            + 0.3 * tilt
            + 0.1 * rainfall
            + 0.1 * (1 - moisture)
        )

        # Add randomness to make it realistic
        risk_score += np.random.normal(0, 0.05)

        # Higher risk_score → more likely to trigger alert
        alert_prob = 1 / (1 + np.exp(-8 * (risk_score - 0.6)))
        alert = np.random.rand() < alert_prob

        rows.append(
            [node, timestamp, vibration, moisture, tilt, rainfall, int(alert)]
        )

df = pd.DataFrame(
    rows, columns=["node", "timestamp", "vibration", "moisture", "tilt", "rainfall", "alert"]
)

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Generated realistic synthetic dataset → {OUTPUT_PATH} ({len(df)} rows)")
