import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample

# ============ CONFIG ============
DATA_PATH = "data/processed/final_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SEQ_LEN = 10
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 10
DROPOUT = 0.3


# ============ LOAD DATA ============
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["alert"] = df["alert"].fillna(0).astype(int)
    return df


# ============ SEQUENCE BUILDER ============
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


# ============ SIMPLE BILSTM ============
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).view(-1)


# ============ TRAIN LSTM ============
def train_lstm(X_train, y_train, X_val, y_val, n_features):
    model = LSTMClassifier(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    best_f1, patience = 0, 0
    f1_scores, losses = [], []

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # evaluate
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(out)
                true.append(yb.numpy())
        preds, true = np.concatenate(preds), np.concatenate(true)
        y_bin = (preds >= 0.5).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(true, y_bin, average="binary")
        auc = roc_auc_score(true, preds)
        f1_scores.append(f1)

        print(f"[LSTM] Epoch {epoch+1}/{EPOCHS} | F1={f1:.3f} | AUC={auc:.3f} | Loss={avg_loss:.4f}")
        if f1 > best_f1:
            best_f1, patience = f1, 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "bilstm_best.pth"))
        else:
            patience += 1
        if patience >= PATIENCE:
            print(f"⏸ Early stop at epoch {epoch+1}")
            break

    # ===== Save training curve =====
    plt.figure(figsize=(8,5))
    plt.plot(f1_scores, label="F1-score", color="green")
    plt.plot(losses, label="Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves - BiLSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"))
    plt.close()

    return best_f1


# ============ MAIN PIPELINE ============
if __name__ == "__main__":
    print("📦 Loading dataset...")
    df = load_data()

    features = ["vibration", "moisture", "tilt", "rainfall"]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # --- RandomForest baseline ---
    print("\n🌲 Training RandomForest...")
    X, y = df[features], df["alert"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    rf_f1 = precision_recall_fscore_support(y_test, y_pred, average="binary")[2]
    rf_auc = roc_auc_score(y_test, y_prob)
    print(classification_report(y_test, y_pred))
    print(f"✅ RandomForest F1={rf_f1:.3f} | AUC={rf_auc:.3f}")
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))

    # --- BiLSTM model ---
    print("\n🔁 Building sequences for BiLSTM...")
    X_seq, y_seq = build_sequences(df, features)
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=SEED)

    # optional balance
    X_pos, X_neg = X_train[y_train == 1], X_train[y_train == 0]
    y_pos, y_neg = y_train[y_train == 1], y_train[y_train == 0]
    if len(X_pos) < len(X_neg):
        X_pos_up, y_pos_up = resample(X_pos, y_pos, replace=True, n_samples=len(X_neg), random_state=SEED)
        X_train = np.concatenate([X_pos_up, X_neg])
        y_train = np.concatenate([y_pos_up, y_neg])
    print("✅ Balanced train set:", np.bincount(y_train.astype(int)))

    print("\n🧠 Training BiLSTM...")
    lstm_f1 = train_lstm(X_train, y_train, X_val, y_val, n_features=len(features))
    print(f"✅ BiLSTM Best F1={lstm_f1:.3f}")

    # --- Compare and Save Best ---
    if lstm_f1 > rf_f1:
        print("\n🏆 Best Model: BiLSTM")
        best_model = "bilstm_best.pth"
    else:
        print("\n🏆 Best Model: RandomForest")
        best_model = "rf_model.pkl"

    print(f"💾 Saved best model → {os.path.join(MODELS_DIR, best_model)}")
