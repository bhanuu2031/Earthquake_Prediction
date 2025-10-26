import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# --- File paths ---
EQ1 = "data/raw/Earthquakes.csv"
EQ2 = "data/raw/Earthquakes_Worldwide.csv"
LS  = "data/raw/landslide.csv"
SENSOR_PATH = "data/synthetic/"

# --- Step 1: Load datasets ---
eq1 = pd.read_csv(EQ1)
eq2 = pd.read_csv(EQ2)
ls  = pd.read_csv(LS)

print("✅ Loaded Datasets:")
print("EQ1:", eq1.shape, "EQ2:", eq2.shape, "Landslides:", ls.shape)

# --- Step 2: Basic cleaning ---
def clean_dataframe(df):
    df = df.dropna(axis=0, how='any')
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.strip()
    return df

eq1 = clean_dataframe(eq1)
eq2 = clean_dataframe(eq2)
ls  = clean_dataframe(ls)

# --- Step 3: Convert timestamps ---
for df in [eq1, eq2, ls]:
    for col in df.columns:
        if "time" in col or "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)  # <-- UTC standardized

# --- Step 4: Merge all earthquake datasets ---
merged_eq = pd.concat([eq1, eq2], ignore_index=True)
merged_eq = merged_eq.drop_duplicates(subset=["latitude", "longitude", "time"], keep="first")

# --- Step 5: Merge earthquake + landslide info ---
merged_eq["event_type"] = "earthquake"
ls["event_type"] = "landslide"
all_events = pd.concat([merged_eq, ls], ignore_index=True)

# --- Step 6: Filter region (India bounds) ---
if "latitude" in all_events.columns and "longitude" in all_events.columns:
    all_events = all_events[
        (all_events["latitude"].between(6, 38)) &
        (all_events["longitude"].between(68, 98))
    ]

print("✅ Filtered Events for India:", all_events.shape)

# --- Step 7: Read synthetic IoT sensor data ---
sensor_dfs = []
for f in os.listdir(SENSOR_PATH):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(SENSOR_PATH, f))
        df["node"] = f.split(".")[0]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)  # <-- also UTC
        sensor_dfs.append(df)

sensors = pd.concat(sensor_dfs, ignore_index=True)
print("✅ Combined Synthetic Sensors:", sensors.shape)

# --- Step 8: Align timestamps with events ---
sensors["alert"] = 0

# Make sure sensors timestamps are timezone-naive for consistent comparison
sensors["timestamp"] = sensors["timestamp"].dt.tz_localize(None)

for _, event in all_events.iterrows():
    event_time = pd.to_datetime(event.get("time"), errors="coerce")
    if pd.isna(event_time):
        continue

    # Drop timezone info if present
    if event_time.tzinfo is not None:
        event_time = event_time.tz_localize(None)

    window_start = event_time - pd.Timedelta(minutes=30)
    window_end = event_time

    sensors.loc[
        (sensors["timestamp"] >= window_start) &
        (sensors["timestamp"] <= window_end),
        "alert"
    ] = 1

# --- Step 9: Normalize numeric columns ---
scaler = MinMaxScaler()
numeric_cols = ["vibration", "moisture", "tilt", "rainfall"]
numeric_cols = [col for col in numeric_cols if col in sensors.columns]

if numeric_cols:
    sensors[numeric_cols] = scaler.fit_transform(sensors[numeric_cols])

# --- Step 10: Save processed dataset ---
os.makedirs("data/processed", exist_ok=True)
sensors.to_csv("data/processed/final_dataset.csv", index=False)

print("💾 Final processed dataset saved to data/processed/final_dataset.csv")
print(sensors.head())
