import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
import geocoder
import streamlit.components.v1 as components

# -----------------------------
# 🌍 Setup & Config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="🌋 Earthquake Prediction Dashboard", layout="wide")

st.title("🌋 Real-Time Earthquake Prediction Dashboard")
st.markdown("This dashboard simulates live IoT sensor readings and sends them to your ML model API in real time.")

# -----------------------------
# 📍 Browser-based Location Detection
# -----------------------------
st.sidebar.subheader("📍 Current Location")

# Inject JavaScript to get browser location
get_location_js = """
<script>
function sendLocation() {
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            const data = {lat: lat, lon: lon};
            window.parent.postMessage(data, "*");
        },
        (err) => {
            window.parent.postMessage({error: err.message}, "*");
        }
    );
}
sendLocation();
</script>
"""
components.html(get_location_js, height=0)

# Placeholder for location
if "lat" not in st.session_state:
    st.session_state.lat = None
    st.session_state.lon = None

# Listen for browser messages (handled automatically by Streamlit runtime)
location_data = st.query_params
if "lat" in location_data and "lon" in location_data:
    st.session_state.lat = float(location_data["lat"][0])
    st.session_state.lon = float(location_data["lon"][0])

# Fallback: IP-based lookup if GPS not available
if not st.session_state.lat or not st.session_state.lon:
    try:
        g = geocoder.ip("me")
        if g.latlng:
            st.session_state.lat, st.session_state.lon = g.latlng
        else:
            st.session_state.lat, st.session_state.lon = (28.6139, 77.2090)  # fallback: Delhi
    except Exception:
        st.session_state.lat, st.session_state.lon = (28.6139, 77.2090)

# Show map
st.sidebar.map(pd.DataFrame({"lat": [st.session_state.lat], "lon": [st.session_state.lon]}))

# -----------------------------
# ⚙️ Simulation parameters
# -----------------------------
refresh_rate = st.sidebar.slider("⏱ Update every (seconds)", 1, 10, 3)
seq_length = 10  # number of readings per sequence

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "vibration", "moisture", "tilt", "rainfall", "prediction", "score"])

# -----------------------------
# 📊 Streamlit Dashboard Layout
# -----------------------------
placeholder = st.empty()

# -----------------------------
# 🚀 Real-time simulation loop
# -----------------------------
st.sidebar.subheader("▶️ Simulation Control")
if st.sidebar.button("Start Simulation"):
    st.sidebar.success("Simulation running... Close tab to stop.")

    while True:
        # Simulate random sensor readings
        new_data = {
            "timestamp": datetime.now(),
            "vibration": round(random.uniform(0.2, 1.0), 2),
            "moisture": round(random.uniform(0.1, 0.9), 2),
            "tilt": round(random.uniform(0.1, 0.8), 2),
            "rainfall": round(random.uniform(0.1, 1.0), 2),
        }

        df = pd.DataFrame([new_data])

        # Send to API
        try:
            response = requests.post(API_URL, json=df[["vibration", "moisture", "tilt", "rainfall"]].to_dict(orient="records"))
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

        # Append data
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)

        # Keep only recent N readings
        st.session_state.data = st.session_state.data.tail(100)

        # -----------------------------
        # 📈 Display dashboard
        # -----------------------------
        with placeholder.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Sensor Readings (Real-time)")
                st.line_chart(
                    st.session_state.data.set_index("timestamp")[["vibration", "moisture", "tilt", "rainfall"]]
                )

            with col2:
                st.metric("🌡 Latest Prediction", new_data["prediction"])
                st.metric("🔢 Confidence Score", f"{new_data['score']:.2f}")
                if new_data["prediction"] == "Possible Event Detected":
                    st.error("⚠️ Alert: Potential seismic activity detected!")
                else:
                    st.success("✅ Normal conditions.")

            st.divider()
            st.dataframe(st.session_state.data.tail(10), use_container_width=True)

        time.sleep(refresh_rate)
