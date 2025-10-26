import numpy as np
import pandas as pd
import time
import os

def generate_sensor_data(node_id, duration_min=60):
    """
    Simulates IoT sensor data for vibration, moisture, tilt, rainfall.
    Each node represents one location.
    """
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=duration_min, freq="1min")

    # Generate random baseline sensor readings
    vibration = np.random.normal(0, 0.3, duration_min)
    moisture = np.random.uniform(25, 45, duration_min)
    tilt = np.random.normal(0, 1, duration_min)
    rainfall = np.random.choice([0, 0, 1, 2, 5, 10], duration_min)

    # Inject a simulated “earthquake event” spike
    event_time = np.random.randint(10, duration_min - 10)
    vibration[event_time:event_time+5] += np.linspace(2, 5, 5)
    tilt[event_time:event_time+5] += np.linspace(1, 3, 5)
    moisture[event_time:event_time+5] += np.linspace(10, 15, 5)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "vibration": vibration,
        "moisture": moisture,
        "tilt": tilt,
        "rainfall": rainfall
    })

    # Save file
    os.makedirs("data/synthetic", exist_ok=True)
    out_path = f"data/synthetic/node_{node_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Node {node_id} data generated → {out_path} ({df.shape})")

    return df


if __name__ == "__main__":
    for i in range(3):  # simulate 3 nodes
        generate_sensor_data(i+1)
