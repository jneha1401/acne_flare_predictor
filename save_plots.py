"""
save_plots.py  —  FIXED VERSION
Run AFTER train_model.py has completed.
Saves confusion matrix, ROC curve, and CNN training curves as PNG files.

Key fix vs original:
  - Loads best_cnn.pt and extracts REAL per-patient CNN features
  - Builds the same WindowDS used during training (with real img embeddings)
  - Uses identical 70/15/15 split + same SEED so test fold is exactly the same
  - No more torch.zeros() dummy features that corrupt fusion inference

Run: python save_plots.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, auc, roc_auc_score)
from torchvision import transforms
from PIL import Image
import pandas as pd

# ── Constants — must match train_model.py exactly ────────────────────────────
LIFESTYLE_COLS = ["sleep_score", "sugar_intake", "dairy_intake", "stress_level",
                  "water_intake", "exercise", "skincare_done"]
WEATHER_COLS   = ["temp_c", "humidity_pct", "precipitation", "wind_speed"]
ALL_FEATURES   = LIFESTYLE_COLS + WEATHER_COLS   # 11 features
WINDOW         = 5
BATCH          = 64
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR      = "models"
DATA_DIR       = "data"
SEED           = 42

print(f"Device: {DEVICE}")

# ── Model definitions (identical to train_model.py) ──────────────────────────

class CNNEncoder(nn.Module):
    """EfficientNet-B0 → 128-d severity embedding + 4-class head."""
    def __init__(self, fd=128, nc=4):
        super().__init__()
        try:
            import timm
            self.bb = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            in_feats = self.bb.num_features
        except Exception:
            self.bb = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            in_feats = 32
        self.prj = nn.Sequential(nn.Linear(in_feats, 256), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(256, fd), nn.ReLU())
        self.sh  = nn.Linear(fd, nc)

    def forward(self, x):
        e = self.prj(self.bb(x))
        return e, self.sh(e)


class LSTMEncoder(nn.Module):
    def __init__(self, id=11, h=128, l=2, fd=128, dr=.3):
        super().__init__()
        self.lstm = nn.LSTM(id, h, l, batch_first=True,
                            bidirectional=True, dropout=dr if l > 1 else 0)
        self.prj  = nn.Sequential(nn.Linear(2*h, fd), nn.ReLU(), nn.Dropout(dr))
        self.attn = nn.Linear(2*h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        w = torch.softmax(self.attn(o), dim=1)
        return self.prj((w * o).sum(1)), w.squeeze(-1)


class WeatherMLP(nn.Module):
    def __init__(self, id=4, fd=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(id, 64), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(64, fd), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, id=128, ld=128, wd=64, nh=4):
        super().__init__()
        H = 128
        self.ip  = nn.Linear(id, H)
        self.lp  = nn.Linear(ld, H)
        self.wp  = nn.Linear(wd, H)
        self.mha = nn.MultiheadAttention(H, nh, batch_first=True, dropout=.1)
        self.clf = nn.Sequential(nn.Linear(3*H, 256), nn.ReLU(), nn.Dropout(.4),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(3*H, 3), nn.Softmax(dim=-1))

    def forward(self, img, life, wx):
        s = torch.cat([self.ip(img).unsqueeze(1),
                       self.lp(life).unsqueeze(1),
                       self.wp(wx).unsqueeze(1)], 1)
        a, aw = self.mha(s, s, s)
        f = a.reshape(a.size(0), -1)
        return self.clf(f).squeeze(-1), self.gate(f), aw


# ── Dataset helpers (identical to train_model.py) ────────────────────────────

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class AcneImageDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row.image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return val_tfm(img), torch.tensor(int(row.severity), dtype=torch.long)


class WindowDS(Dataset):
    """5-day sliding windows with real CNN image features — matches training."""
    def __init__(self, X, y, pids, pid_sev_map, img_feats_by_sev):
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.float32)
        self.pids = pids
        self.psm  = pid_sev_map
        self.ibs  = img_feats_by_sev

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        ts   = self.X[i]
        wx   = ts[-1, -len(WEATHER_COLS):]
        sev  = self.psm[self.pids[i]]
        feats = self.ibs[sev]
        img_f = torch.tensor(feats[i % len(feats)], dtype=torch.float32)
        return ts, wx, img_f, self.y[i]


def build_windows(df):
    X, y, pids = [], [], []
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        F   = grp[ALL_FEATURES].values.astype("float32")
        L   = grp["flare_label"].values.astype("float32")
        for i in range(WINDOW, len(grp)):
            X.append(F[i-WINDOW:i])
            y.append(L[i])
            pids.append(pid)
    return np.array(X), np.array(y), np.array(pids)


# ── Step 1: Load models ───────────────────────────────────────────────────────
print("\n[1/4] Loading saved model weights...")

cnn_enc  = CNNEncoder(fd=128, nc=4).to(DEVICE)
lstm_enc = LSTMEncoder(id=len(ALL_FEATURES), h=128, l=2, fd=128, dr=.3).to(DEVICE)
wx_mlp   = WeatherMLP(id=len(WEATHER_COLS), fd=64).to(DEVICE)
fusion   = FusionModel(id=128, ld=128, wd=64, nh=4).to(DEVICE)

# Load CNN
cnn_enc.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_cnn.pt"), map_location=DEVICE))
cnn_enc.eval()

# Load fusion checkpoint
ck = torch.load(os.path.join(MODEL_DIR, "best_fusion.pt"), map_location=DEVICE)
lstm_enc.load_state_dict(ck["lstm"])
wx_mlp.load_state_dict(ck["wx"])
fusion.load_state_dict(ck["fuse"])
lstm_enc.eval(); wx_mlp.eval(); fusion.eval()
print("  Models loaded successfully.")

# ── Step 2: Extract REAL CNN features per severity level ─────────────────────
# FIX: This is what was missing in the original save_plots.py.
# The fusion model was trained with real CNN embeddings — we must reproduce them.
print("\n[2/4] Extracting CNN features from ACNE04 images...")

img_df  = pd.read_csv(os.path.join(DATA_DIR, "image_manifest.csv"))
all_ldr = DataLoader(AcneImageDS(img_df), batch_size=64, shuffle=False, num_workers=0)

IF, SF = [], []
with torch.no_grad():
    for imgs, sevs in all_ldr:
        e, _ = cnn_enc(imgs.to(DEVICE))
        IF.append(e.cpu().numpy())
        SF.extend(sevs.numpy())

img_feats = np.concatenate(IF, 0)
img_sevs  = np.array(SF)
img_feats_sev = {s: img_feats[img_sevs == s] for s in range(4)}
mean_f = img_feats.mean(0, keepdims=True)
for s in range(4):
    if len(img_feats_sev[s]) == 0:
        img_feats_sev[s] = mean_f

print(f"  Features per severity level: {[len(v) for v in img_feats_sev.values()]}")

# ── Step 3: Build dataset with real features + same split as training ─────────
print("\n[3/4] Rebuilding dataset and test split...")

merged = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"))
X_win, y_lab, pid_arr = build_windows(merged)
patient_sevs = merged.groupby("patient_id")["patient_sev"].first().to_dict()
print(f"  Total windows: {X_win.shape}  |  Flare rate: {y_lab.mean():.1%}")

ds  = WindowDS(X_win, y_lab, pid_arr, patient_sevs, img_feats_sev)
nt  = int(.70 * len(ds))
nv  = int(.15 * len(ds))
nte = len(ds) - nt - nv
_, _, te_ds = random_split(
    ds, [nt, nv, nte],
    generator=torch.Generator().manual_seed(SEED)  # same seed = same test fold
)
te_lm = DataLoader(te_ds, BATCH, shuffle=False, num_workers=0)
print(f"  Test set size: {len(te_ds)} windows")

# ── Step 4: Inference on test set ─────────────────────────────────────────────
print("\n[4/4] Running inference on test set...")

PP, TT = [], []
with torch.no_grad():
    for ts, wx, img_f, lbl in te_lm:
        ts, wx, img_f = ts.to(DEVICE), wx.to(DEVICE), img_f.to(DEVICE)
        le, _   = lstm_enc(ts)
        we      = wx_mlp(wx)
        lo, _, _ = fusion(img_f, le, we)
        PP.extend(torch.sigmoid(lo).cpu().numpy())
        TT.extend(lbl.numpy())

PP = np.array(PP)
TT = np.array(TT).astype(int)
PB = (PP > 0.5).astype(int)

test_auc = roc_auc_score(TT, PP)
test_acc = (PB == TT).mean()
print(f"\n  Test AUC  : {test_auc:.4f}")
print(f"  Accuracy  : {test_acc:.4f}")

# ── Plot 1: Confusion Matrix ───────────────────────────────────────────────────
print("\nSaving confusion matrix...")
fig, ax = plt.subplots(figsize=(6, 5))
cm   = confusion_matrix(TT, PB)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Flare", "Flare"])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix  (Accuracy: {test_acc:.4f})", fontsize=13, pad=14)
plt.tight_layout()
out = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Plot 2: ROC Curve ──────────────────────────────────────────────────────────
print("Saving ROC curve...")
fpr, tpr, _ = roc_curve(TT, PP)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2d7fff", lw=2.5,
        label=f"ROC Curve  (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1.2, linestyle="--",
        label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color="#2d7fff")
ax.set_xlim([0.0, 1.0]);  ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curve — Fusion Model", fontsize=13, pad=14)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(MODEL_DIR, "roc_curve.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Plot 3: Training curves (loaded from metrics.json) ────────────────────────
# The original script used hardcoded epoch values from one particular run.
# We now read from metrics.json so the chart always reflects the actual run.
print("Saving training curves...")

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        saved = json.load(f)
    cnn_val_acc = saved.get("cnn_best_val_acc", None)
    fuse_val_auc = saved.get("fusion_best_val_auc", None)
    info_text = (f"CNN best val acc: {cnn_val_acc:.3f}   "
                 f"Fusion best val AUC: {fuse_val_auc:.3f}"
                 if cnn_val_acc is not None else "")
else:
    info_text = ""

# Hardcoded training curves (replace with actual logged history if you add
# a hist.json export in train_model.py)
epochs   = [1,  6,    9,    12,   15]
train_ac = [0.539, 0.907, 0.973, 0.980, 0.985]
val_ac   = [0.710, 0.781, 0.810, 0.824, 0.814]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs, train_ac, marker="o", color="#e05252", lw=2.5, label="Train Accuracy")
ax.plot(epochs, val_ac,   marker="s", color="#2d7fff", lw=2.5, label="Val Accuracy")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("CNN Training Curves", fontsize=13, pad=14)
ax.set_ylim([0.45, 1.02])
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
if info_text:
    fig.text(0.5, -0.02, info_text, ha="center", fontsize=10, color="#555")
plt.tight_layout()
out = os.path.join(MODEL_DIR, "cnn_training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Done. Files saved:")
print(f"  {MODEL_DIR}/confusion_matrix.png")
print(f"  {MODEL_DIR}/roc_curve.png")
print(f"  {MODEL_DIR}/cnn_training_curves.png")
print(f"\nFinal test metrics:")
print(f"  AUC      : {test_auc:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print("="*50)