import random
import time
import pandas as pd
import os

def stream_sensor_data(node_id):
    """
    Simulates real-time streaming of sensor readings.
    Appends a new record every 5 seconds to CSV.
    """
    os.makedirs("data/stream", exist_ok=True)
    path = f"data/stream/node_{node_id}.csv"

    # write header once
    with open(path, "w") as f:
        f.write("timestamp,vibration,moisture,tilt,rainfall\n")

    print(f"🚀 Streaming started for Node {node_id} → {path}")
    while True:
        vibration = random.uniform(0, 1)
        moisture = random.uniform(20, 50)
        tilt = random.uniform(-2, 2)
        rainfall = random.choice([0, 1, 2, 5])
        timestamp = pd.Timestamp.now()

        with open(path, "a") as f:
            f.write(f"{timestamp},{vibration},{moisture},{tilt},{rainfall}\n")

        print(f"📡 Node {node_id}: {timestamp}")
        time.sleep(5)  # wait 5 s before next reading

if __name__ == "__main__":
    stream_sensor_data(1)
