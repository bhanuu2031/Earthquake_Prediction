import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Config ----------
DATA_PATH = "data/processed/final_dataset.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- Hyperparameters ----------
SEQ_LEN = 30  # reduced for more dynamic learning
PRED_HORIZON = 5
BATCH_SIZE = 32
EPOCHS_LSTM = 150
LR = 1e-3
DROPOUT = 0.3
PATIENCE = 10

# ---------- Utilities ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    ts_cols = [c for c in df.columns if "time" in c or "date" in c or "timestamp" in c]
    if ts_cols:
        df.rename(columns={ts_cols[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("No timestamp/date column found in dataset")
    if "alert" not in df.columns:
        df["alert"] = 0
    df["alert"] = df["alert"].fillna(0).astype(int)
    return df

def build_sequences(df, features, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON):
    seqs, labels = [], []
    node_ids = df['node'].unique() if 'node' in df.columns else [None]
    for node in node_ids:
        node_df = df[df['node'] == node] if node is not None else df
        node_df = node_df.sort_values("timestamp").reset_index(drop=True)
        arr = node_df[features].values
        alerts = node_df['alert'].values
        N = len(node_df)
        if N < seq_len + pred_horizon:
            continue
        for i in range(N - seq_len - pred_horizon + 1):
            X = arr[i:i + seq_len]
            future_alert = alerts[i + seq_len:i + seq_len + pred_horizon]
            y = 1 if future_alert.sum() > 0 else 0
            seqs.append(X)
            labels.append(y)
    return np.stack(seqs), np.array(labels)

# ---------- Model ----------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attn(out)
        return self.fc(context).squeeze(1)

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        prob = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

# ---------- Training ----------
def train_lstm(X_train, y_train, X_val, y_val, n_features):
    model = LSTMClassifier(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)
    criterion = FocalLoss()

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=False)

    best_f1, patience = 0, 0
    for epoch in range(EPOCHS_LSTM):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(out)
                true.append(yb.numpy())
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        best_thresh, best_local_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(true, preds > t)
            if f1 > best_local_f1:
                best_local_f1, best_thresh = f1, t

        auc = roc_auc_score(true, preds)
        scheduler.step(best_local_f1)

        if best_local_f1 > best_f1:
            best_f1 = best_local_f1
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_best.pth"))
            torch.save({"threshold": best_thresh}, os.path.join(MODELS_DIR, "best_thresh.pt"))
        else:
            patience += 1

        print(f"[LSTM] Epoch {epoch+1}/{EPOCHS_LSTM} F1={best_local_f1:.4f} AUC={auc:.4f} Th={best_thresh:.2f}")
        if patience >= PATIENCE:
            print(f"⏸ Early stopping at epoch {epoch+1}")
            break

    return model

# ---------- Main ----------
if __name__ == "__main__":
    print("📦 Loading dataset...")
    df = load_data()

    features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in features if c != "alert"]

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    print("🔍 Feature correlations with 'alert':")
    print(df.select_dtypes(include=[np.number]).corr()['alert'].sort_values(ascending=False))

    print("🔁 Building sequences...")
    X, y = build_sequences(df, features)
    print(f"✅ Sequences: {X.shape}, Labels: {y.shape}, Positives: {y.sum()}")
    print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    # 🔁 Random undersampling (safer for sequential data)
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    y_pos = y_train[y_train == 1]
    y_neg = y_train[y_train == 0]
    X_neg_down, y_neg_down = resample(X_neg, y_neg, replace=False, n_samples=len(X_pos), random_state=SEED)
    X_train = np.concatenate([X_pos, X_neg_down])
    y_train = np.concatenate([y_pos, y_neg_down])
    print("✅ Applied Random Undersampling:", np.bincount(y_train))

    print("🧠 Training BiLSTM with Attention + Focal Loss...")
    model = train_lstm(X_train, y_train, X_val, y_val, n_features=len(features))

    print("📊 Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_best.pth")))
    threshold_path = os.path.join(MODELS_DIR, "best_thresh.pt")
    threshold = torch.load(threshold_path)["threshold"] if os.path.exists(threshold_path) else 0.5

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))).cpu().numpy()
    y_bin = (preds >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_bin, average='binary', zero_division=0)
    auc = roc_auc_score(y_test, preds)
    print(f"\n✅ Final Test Results: Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

    pd.DataFrame({
        "y_true": y_test,
        "y_pred_prob": preds,
        "y_pred_label": y_bin
    }).to_csv("data/processed/predictions_summary.csv", index=False)
    print("📁 Saved predictions → data/processed/predictions_summary.csv")
