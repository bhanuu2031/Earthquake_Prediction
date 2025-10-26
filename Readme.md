# 🌋 Earthquake Prediction System — Real-Time ML + IoT Dashboard

An end-to-end **Earthquake Prediction and Monitoring System** that combines **IoT sensor simulation**, a **hybrid ML model (BiLSTM + Random Forest)**, and a **real-time Streamlit dashboard** to visualize seismic risks in different regions.

---

## 🚀 Project Overview

This project predicts potential earthquake-like events based on simulated or real IoT sensor readings such as **vibration, moisture, tilt, and rainfall**.

The system consists of three major components:

1. 🧠 **ML Model API (Flask)** — Receives sensor data and returns predictions.  
2. 📊 **Streamlit Dashboard** — Visualizes live sensor readings, risk levels, and alerts.  
3. 🌍 **Data Enhancements (Step 9)** — Integrates USGS earthquake feed, LLM summaries, and persistent alert history.

---

## 🧩 System Architecture

IoT Sensors (Simulated / Real)
↓
Flask API (PyTorch Model)
↓
Streamlit Dashboard (Frontend)
↓
SQLite DB + Optional LLM Insights

yaml
Copy code

---

## ⚙️ Features

### ✅ **Core Functionality**
- Real-time simulation of IoT sensor data.
- Predicts “Normal” vs. “Possible Event Detected”.
- Displays live metrics and trend charts.

### 🗺️ **Visual Dashboard**
- Live line charts for vibration, tilt, rainfall, and moisture.
- Risk-level map showing multi-location sensors:
  - 🟥 Red = High risk  
  - 🟨 Yellow = Moderate  
  - 🟩 Green = Normal
- Alerts and confidence score indicators.

### 🧭 **Optional Enhancements**
| Feature | Description |
|----------|-------------|
| 🗺️ **Multi-node map** | Multiple sensors shown with live alert pins. |
| 🔔 **Alert history** | Persistent storage of recent alerts (SQLite). |
| 🎛️ **Threshold slider** | Manually adjust detection sensitivity. |
| 📂 **Data export** | Download current session as CSV. |
| 📊 **Risk trend** | Rolling average of event probability. |

### 💡 **Step 9 — Advanced Additions**
| Enhancement | Description |
|--------------|-------------|
| 🤖 **LLM summaries** | Uses LangChain/OpenAI to summarize seismic activity in natural language. |
| 🌍 **USGS API integration** | Real-time overlay of actual global earthquakes. |
| 💾 **SQLite persistence** | Stores historical data and alerts. |
| 📈 **Trend visualization** | Risk-over-time chart with moving averages. |

---

## 🧠 Machine Learning Model

- Model: **BiLSTMClassifier**
- Input features: `vibration`, `moisture`, `tilt`, `rainfall`
- Framework: PyTorch
- Output: Probability score (0–1) → Threshold = 0.5 for event detection

---

## 📦 Installation

### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/earthquake_prediction.git
cd earthquake_prediction

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🧠 Run the ML Model API
python src/api/predict_api.py
Runs on: http://127.0.0.1:8000/predict

Example request:
curl.exe -X POST http://127.0.0.1:8000/predict ^
     -H "Content-Type: application/json" ^
     -d "[{\"vibration\":0.52,\"moisture\":0.36,\"tilt\":0.28,\"rainfall\":0.42}]"

Response Example
{
  "prediction": "Normal",
  "score": 0.0569
}

📊 Run the Dashboard
streamlit run src/dashboard/app.py

This will open your browser at:
Local: http://localhost:8501

🌍 Future Enhancements (Planned)
Feature	Description
🛰️ IoT Hardware Integration	Collect real sensor data using ESP32 / Raspberry Pi.
🧮 Hybrid Ensemble Model	Combine BiLSTM temporal prediction with Random Forest decision refinement.
📱 Mobile Alerts	Push notifications via Telegram / Twilio API.
🧠 Explainable AI	Use SHAP / LIME to visualize feature importance.
☁️ Cloud Deployment	Deploy API on Render or AWS, dashboard on Streamlit Cloud.

🧾 Project Structure

earthquake_prediction/
│
├── src/
│   ├── models/
│   │   ├── train_and_eval_hybrid.py     # ML model training
│   │   └── bilstm_best.pth              # Saved model weights
│   ├── api/
│   │   └── predict_api.py               # Flask API for predictions
│   └── dashboard/
│       └── app.py                       # Streamlit live dashboard
│
├── data/                                # Datasets (if any)
├── models/                              # Saved checkpoints
├── requirements.txt
├── README.md
└── sample.json                          # Example test input

🧭 Tech Stack
Layer	       Technology
Backend	       Flask, PyTorch
Frontend	   Streamlit
Data	       Pandas, NumPy
Visualization  Plotly / Streamlit native charts
Database	   SQLite
Optional AI	   OpenAI / LangChain
External Feed  USGS API

🧰 Requirements
nginx
Copy code
torch
flask
streamlit
pandas
numpy
requests
geocoder
langchain
openai
sqlite3

👨‍💻 Author
Chitrabhanu Srivastava
Aamya Pandey

🏁 License
MIT License © 2025 Chitrabhanu Srivastava

Feel free to connect