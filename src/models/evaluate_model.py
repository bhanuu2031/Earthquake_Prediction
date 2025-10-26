import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============ CONFIG ============
DATA_PATH = "data/processed/final_dataset.csv"
MODEL_PATH = "models/bilstm_best.pth"
SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
FEATURES = ["vibration", "moisture", "tilt", "rainfall"]


# ============ MODEL CLASS ============
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        return self.fc(out).view(-1)


# ============ LOAD DATA ============
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["alert"] = df["alert"].fillna(0).astype(int)
    return df


# ============ BUILD SEQUENCES ============
def build_sequences(df, features, seq_len=SEQ_LEN):
    seqs, labels = [], []
    for node in df["node"].unique():
        node_df = df[df["node"] == node].sort_values("timestamp")
        arr = node_df[features].values
        alerts = node_df["alert"].values
        for i in range(len(node_df) - seq_len):
            seqs.append(arr[i : i + seq_len])
            labels.append(alerts[i + seq_len - 1])
    return np.stack(seqs), np.array(labels)


# ============ MAIN ============
if __name__ == "__main__":
    print("📦 Loading dataset...")
    df = load_data()

    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    print("🔁 Building sequences...")
    X_seq, y_seq = build_sequences(df, FEATURES)
    _, X_test, _, y_test = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

    # Load model
    model = LSTMClassifier(len(FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("🧠 Evaluating model...")
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_prob = torch.sigmoid(model(X_tensor)).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    # ======== Confusion Matrix ========
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Alert", "Alert"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - BiLSTM")
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plt.close()

    # ======== ROC Curve ========
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - BiLSTM")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
    plt.close()

    # ======== Precision-Recall Curve ========
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.plot(recall, precision, color="green", lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve - BiLSTM")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
    plt.close()

    print("✅ All evaluation graphs saved in:", SAVE_DIR)
