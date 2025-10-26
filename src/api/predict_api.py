import os
import sys
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------
# 🧭 Setup
# ----------------------------------
# Ensure project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.train_and_eval_hybrid import LSTMClassifier  # ✅ use your actual model class name

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("models", "bilstm_best.pth")
N_FEATURES = 4

# ----------------------------------
# 🧠 Load Model
# ----------------------------------
print(f"💻 Using device: {DEVICE}")
print(f"📦 Loading model from: {MODEL_PATH}")

model = LSTMClassifier(n_features=N_FEATURES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Ideally, use the same scaler used during training (saved with joblib)
scaler = MinMaxScaler()

# ----------------------------------
# 🚀 API Endpoint
# ----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        df = pd.DataFrame(data)

        # Validate input
        required_features = ["vibration", "moisture", "tilt", "rainfall"]
        if not all(f in df.columns for f in required_features):
            return jsonify({"error": "Missing required features"}), 400

        # Scale the features
        df[required_features] = scaler.fit_transform(df[required_features])

        # Prepare input (1, seq_len, n_features)
        X = np.expand_dims(df[required_features].values, axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        # Run prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(X_tensor)).cpu().item()

        label = "Possible Event Detected" if pred > 0.5 else "Normal"

        return jsonify({
            "prediction": label,
            "score": float(pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------
# ▶️ Run Server
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
