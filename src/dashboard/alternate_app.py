import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# -----------------------------
# 🌍 Setup
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="🌋 Earthquake Prediction Dashboard", layout="wide")

st.title("🌋 Real-Time Earthquake Prediction Dashboard")
st.markdown("This dashboard simulates **multiple IoT sensor nodes** across different regions and sends them to your ML model API in real time.")

# -----------------------------
# ⚙️ Control Panel
# -----------------------------
st.sidebar.header("⚙️ Controls")

refresh_rate = st.sidebar.slider("⏱ Update every (seconds)", 1, 10, 3)
threshold = st.sidebar.slider("🎛️ Detection Threshold", 0.3, 0.9, 0.5)
num_nodes = st.sidebar.slider("🏙️ Number of Sensor Nodes", 3, 10, 5)

st.sidebar.markdown("---")
export_btn = st.sidebar.button("📂 Export Session Data to CSV")

# -----------------------------
# 🧩 Initialize Session
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "location", "vibration", "moisture", "tilt", "rainfall", "prediction", "score"])

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# -----------------------------
# 🌍 Simulated Sensor Locations
# -----------------------------
LOCATIONS = [
    {"city": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"city": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"city": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"city": "Pune", "lat": 18.5204, "lon": 73.8567},
]

# limit by num_nodes
LOCATIONS = LOCATIONS[:num_nodes]

# -----------------------------
# 📊 Layout Setup
# -----------------------------
placeholder = st.empty()
st.sidebar.subheader("🔔 Alert History")

# -----------------------------
# 🚀 Start Simulation
# -----------------------------
if st.sidebar.button("▶️ Start Multi-Node Simulation"):
    st.sidebar.success("Simulation running... Close tab to stop.")
    while True:
        new_records = []

        for loc in LOCATIONS:
            new_data = {
                "timestamp": datetime.now(),
                "location": loc["city"],
                "vibration": round(random.uniform(0.2, 1.0), 2),
                "moisture": round(random.uniform(0.1, 0.9), 2),
                "tilt": round(random.uniform(0.1, 0.8), 2),
                "rainfall": round(random.uniform(0.1, 1.0), 2),
            }

            # Send to API
            try:
                response = requests.post(API_URL, json=[{
                    "vibration": new_data["vibration"],
                    "moisture": new_data["moisture"],
                    "tilt": new_data["tilt"],
                    "rainfall": new_data["rainfall"]
                }])
                if response.status_code == 200:
                    res = response.json()
                    new_data["prediction"] = res["prediction"]
                    new_data["score"] = res["score"]
                else:
                    new_data["prediction"] = "API Error"
                    new_data["score"] = 0.0
            except Exception:
                new_data["prediction"] = "API Offline"
                new_data["score"] = 0.0

            new_records.append(new_data)

        # Add to session
        df_new = pd.DataFrame(new_records)
        st.session_state.data = pd.concat([st.session_state.data, df_new], ignore_index=True)
        st.session_state.data = st.session_state.data.tail(200)

        # Log alerts
        alerts_now = df_new[df_new["score"] > threshold]
        for _, a in alerts_now.iterrows():
            st.session_state.alerts.append(f"[{a['timestamp'].strftime('%H:%M:%S')}] ⚠️ {a['location']} — Score: {a['score']:.2f}")

        # -----------------------------
        # 🗺️ Map Visualization
        # -----------------------------
        map_df = df_new.copy()
        map_df["color"] = map_df["score"].apply(lambda x: "red" if x > 0.7 else ("orange" if x > 0.4 else "green"))
        map_display = pd.DataFrame({
            "lat": [l["lat"] for l in LOCATIONS],
            "lon": [l["lon"] for l in LOCATIONS],
            "color": map_df["color"].tolist(),
        })

        # -----------------------------
        # 📈 Risk Trend & Sensor Data
        # -----------------------------
        with placeholder.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📈 Risk Trend (Rolling Average)")
                rolling = st.session_state.data.groupby("timestamp")["score"].mean().rolling(10).mean()
                st.line_chart(rolling)

                st.subheader("📊 Latest Sensor Readings")
                st.dataframe(df_new, use_container_width=True)

            with col2:
                st.subheader("🗺️ Multi-node Map")
                st.map(map_display)

                st.metric("🌡️ Last Prediction", df_new.iloc[-1]["prediction"])
                st.metric("🔢 Confidence Score", f"{df_new.iloc[-1]['score']:.2f}")

                if df_new.iloc[-1]["score"] > threshold:
                    st.error("⚠️ High risk of seismic activity!")
                else:
                    st.success("✅ Normal conditions.")

                st.markdown("### 🔔 Recent Alerts")
                st.write("\n".join(st.session_state.alerts[-5:]) if st.session_state.alerts else "No recent alerts.")

        # -----------------------------
        # 📂 Export
        # -----------------------------
        if export_btn:
            st.session_state.data.to_csv("session_data.csv", index=False)
            st.sidebar.success("Session data exported as session_data.csv ✅")

        time.sleep(refresh_rate)
