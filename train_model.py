"""
================================================
 Acne Flare Predictor — Model Training
 Architecture:
   CNN (EfficientNet-B0)  → image embedding (128-d)
   BiLSTM + Attention     → lifestyle embedding (128-d)
   WeatherMLP             → weather embedding (64-d)
   Cross-Attention Fusion → flare probability

"""

import os, json, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
ACNE04_DIR  = "data/acne04_images"
IMG_SIZE    = 224
WINDOW      = 5          # days of history per sample
CNN_EP      = 15
MM_EP       = 40
BATCH       = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ Device: {DEVICE}")

LIFESTYLE_COLS = ["sleep_score","sugar_intake","dairy_intake","stress_level",
                  "water_intake","exercise","skincare_done"]
WEATHER_COLS   = ["temp_c","humidity_pct","precipitation","wind_speed"]
ALL_FEATURES   = LIFESTYLE_COLS + WEATHER_COLS   # 11 features


# ══════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════
class AcneImageDS(Dataset):
    """ACNE04 images for CNN severity classifier."""
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])

    def __init__(self, df, train=True):
        self.df  = df.reset_index(drop=True)
        self.tfm = self.train_tfm if train else self.val_tfm

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row.image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(128,128,128))
        img = self.tfm(img)
        sev = int(row.severity)
        return img, torch.tensor(sev, dtype=torch.long), torch.tensor(1. if sev>=2 else 0., dtype=torch.float)


class WindowDS(Dataset):
    """5-day sliding windows of lifestyle+weather → flare label."""
    def __init__(self, X, y, pids, pid_sev_map, img_feats_by_sev):
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.float32)
        self.pids= pids
        self.psm = pid_sev_map
        self.ibs = img_feats_by_sev

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        ts  = self.X[i]
        wx  = ts[-1, -len(WEATHER_COLS):]
        sev = self.psm[self.pids[i]]
        feats = self.ibs[sev]
        img_f = torch.tensor(feats[i % len(feats)], dtype=torch.float32)
        return ts, wx, img_f, self.y[i]


# ══════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════
class CNNEncoder(nn.Module):
    """EfficientNet-B0 → 128-d severity embedding + 4-class head."""
    def __init__(self, fd=128, nc=4):
        super().__init__()
        try:
            import timm
            self.bb = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
            in_feats = self.bb.num_features
        except Exception:
            # Fallback: simple CNN if timm unavailable
            self.bb = nn.Sequential(
                nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            in_feats = 32
        self.prj = nn.Sequential(nn.Linear(in_feats,256), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(256,fd), nn.ReLU())
        self.sh  = nn.Linear(fd, nc)

    def forward(self, x):
        e = self.prj(self.bb(x))
        return e, self.sh(e)


class LSTMEncoder(nn.Module):
    """BiLSTM + temporal attention → 128-d lifestyle embedding."""
    def __init__(self, id=11, h=128, l=2, fd=128, dr=.3):
        super().__init__()
        self.lstm = nn.LSTM(id, h, l, batch_first=True,
                             bidirectional=True, dropout=dr if l>1 else 0)
        self.prj  = nn.Sequential(nn.Linear(2*h, fd), nn.ReLU(), nn.Dropout(dr))
        self.attn = nn.Linear(2*h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        w = torch.softmax(self.attn(o), dim=1)
        return self.prj((w * o).sum(1)), w.squeeze(-1)


class WeatherMLP(nn.Module):
    """3-layer MLP → 64-d weather embedding."""
    def __init__(self, id=4, fd=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(id,64), nn.ReLU(), nn.Dropout(.2),
                                  nn.Linear(64,fd), nn.ReLU())
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
    """Cross-attention fusion of 3 modalities → flare probability logit."""
    def __init__(self, id=128, ld=128, wd=64, nh=4):
        super().__init__()
        H = 128
        self.ip  = nn.Linear(id, H)
        self.lp  = nn.Linear(ld, H)
        self.wp  = nn.Linear(wd, H)
        self.mha = nn.MultiheadAttention(H, nh, batch_first=True, dropout=.1)
        self.clf = nn.Sequential(nn.Linear(3*H,256), nn.ReLU(), nn.Dropout(.4),
                                  nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
        self.gate= nn.Sequential(nn.Linear(3*H, 3), nn.Softmax(dim=-1))

    def forward(self, img, life, wx):
        s = torch.cat([self.ip(img).unsqueeze(1),
                       self.lp(life).unsqueeze(1),
                       self.wp(wx).unsqueeze(1)], 1)
        a, aw = self.mha(s, s, s)
        f = a.reshape(a.size(0), -1)
        return self.clf(f).squeeze(-1), self.gate(f), aw


# ══════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════
def build_windows(df):
    X, y, pids = [], [], []
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        F   = grp[ALL_FEATURES].values.astype(np.float32)
        L   = grp["flare_label"].values.astype(np.float32)
        for i in range(WINDOW, len(grp)):
            X.append(F[i-WINDOW:i])
            y.append(L[i])
            pids.append(pid)
    return np.array(X), np.array(y), np.array(pids)


def cnn_epoch(cnn, ldr, opt, sch, ce, train=True):
    cnn.train() if train else cnn.eval()
    tl = ok = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, sevs, _ in ldr:
            imgs, sevs = imgs.to(DEVICE), sevs.to(DEVICE)
            e, pred = cnn(imgs)
            loss = ce(pred, sevs)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(imgs)
            ok += (pred.argmax(1) == sevs).sum().item()
            n  += len(imgs)
    if train and sch: sch.step()
    return tl/n, ok/n


def mm_epoch(lstm_enc, wx_mlp, fusion, ldr, opt, sch, bce, train=True):
    for m in [lstm_enc, wx_mlp, fusion]:
        m.train() if train else m.eval()
    tl = 0; PS = []; TS = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for ts, wx, img, lbl in ldr:
            ts, wx, img, lbl = [t.to(DEVICE) for t in [ts, wx, img, lbl]]
            le, _  = lstm_enc(ts)
            we     = wx_mlp(wx)
            lo, _, _= fusion(img, le, we)
            loss   = bce(lo, lbl)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(lstm_enc.parameters())+list(wx_mlp.parameters())+list(fusion.parameters()),
                    1.0)
                opt.step()
            tl += loss.item() * len(lbl)
            PS.extend(torch.sigmoid(lo).cpu().detach().numpy())
            TS.extend(lbl.cpu().numpy())
    if train and sch: sch.step()
    try:    auc = roc_auc_score(TS, PS)
    except: auc = 0.5
    acc = ((np.array(PS) > .5) == np.array(TS)).mean()
    return tl / len(ldr.dataset), auc, acc


# ══════════════════════════════════════════════
# MAIN TRAINING ROUTINE
# ══════════════════════════════════════════════
def main():
    # ── Load data ──────────────────────────────
    print("\n[1/5] Loading data …")
    merged = pd.read_csv(f"{DATA_DIR}/merged_dataset.csv")
    img_df = pd.read_csv(f"{DATA_DIR}/image_manifest.csv")
    print(f"  Merged dataset : {len(merged):,} rows")
    print(f"  Image manifest : {len(img_df)} images")

    # ── CNN training ───────────────────────────
    print("\n[2/5] Training CNN severity classifier …")
    from sklearn.model_selection import train_test_split
    tr_df, va_df = train_test_split(img_df, test_size=.2, stratify=img_df.severity, random_state=SEED)
    tr_img = DataLoader(AcneImageDS(tr_df, True),  batch_size=32, shuffle=True,  num_workers=0)
    va_img = DataLoader(AcneImageDS(va_df, False), batch_size=32, shuffle=False, num_workers=0)

    cnn_enc = CNNEncoder(fd=128, nc=4).to(DEVICE)
    opt_c   = optim.AdamW(cnn_enc.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_c   = CosineAnnealingLR(opt_c, T_max=CNN_EP)
    ce_loss = nn.CrossEntropyLoss()

    best_ca = 0
    for ep in range(1, CNN_EP+1):
        tl, ta = cnn_epoch(cnn_enc, tr_img, opt_c, sch_c, ce_loss, True)
        vl, va = cnn_epoch(cnn_enc, va_img, None,  None,  ce_loss, False)
        if va > best_ca:
            best_ca = va
            torch.save(cnn_enc.state_dict(), f"{MODEL_DIR}/best_cnn.pt")
        if ep % 3 == 0 or ep == 1:
            print(f"  Ep{ep:02d}  Train {ta:.3f}  Val {va:.3f}")
    cnn_enc.load_state_dict(torch.load(f"{MODEL_DIR}/best_cnn.pt", map_location=DEVICE))
    print(f"  ✅ Best CNN val acc: {best_ca:.3f}")

    # ── Extract CNN features per severity ──────
    print("\n[3/5] Extracting CNN features …")
    all_ldr = DataLoader(AcneImageDS(img_df, False), batch_size=64, shuffle=False, num_workers=0)
    IF, SF  = [], []
    cnn_enc.eval()
    with torch.no_grad():
        for imgs, sevs, _ in all_ldr:
            e, _ = cnn_enc(imgs.to(DEVICE))
            IF.append(e.cpu().numpy()); SF.extend(sevs.numpy())
    img_feats     = np.concatenate(IF, 0)
    img_sevs      = np.array(SF)
    img_feats_sev = {s: img_feats[img_sevs==s] for s in range(4)}
    # Fallback: use global mean if a level has no images
    mean_f        = img_feats.mean(0, keepdims=True)
    for s in range(4):
        if len(img_feats_sev[s]) == 0:
            img_feats_sev[s] = mean_f
    print(f"  Level features: {[len(v) for v in img_feats_sev.values()]}")

    # ── Build sliding windows ──────────────────
    print("\n[4/5] Building sliding windows …")
    X_win, y_lab, pid_arr = build_windows(merged)
    patient_sevs          = merged.groupby("patient_id")["patient_sev"].first().to_dict()
    print(f"  Windows: {X_win.shape}  Flare rate: {y_lab.mean():.1%}")

    ds   = WindowDS(X_win, y_lab, pid_arr, patient_sevs, img_feats_sev)
    nt   = int(.70 * len(ds)); nv = int(.15 * len(ds)); nte = len(ds)-nt-nv
    tr_ds, va_ds, te_ds = random_split(ds, [nt, nv, nte],
                                        generator=torch.Generator().manual_seed(SEED))
    tr_lm = DataLoader(tr_ds, BATCH, shuffle=True,  num_workers=0)
    va_lm = DataLoader(va_ds, BATCH, shuffle=False, num_workers=0)
    te_lm = DataLoader(te_ds, BATCH, shuffle=False, num_workers=0)

    # ── Train fusion model ─────────────────────
    print("\n[5/5] Training fusion model …")
    n0 = int((y_lab==0).sum()); n1 = int((y_lab==1).sum())
    pw = torch.tensor([n0/n1], dtype=torch.float).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    lstm_enc = LSTMEncoder(id=len(ALL_FEATURES), h=128, l=2, fd=128, dr=.3).to(DEVICE)
    wx_mlp   = WeatherMLP(id=len(WEATHER_COLS), fd=64).to(DEVICE)
    fusion   = FusionModel(id=128, ld=128, wd=64, nh=4).to(DEVICE)

    ap    = list(lstm_enc.parameters())+list(wx_mlp.parameters())+list(fusion.parameters())
    opt_m = optim.AdamW(ap, lr=8e-4, weight_decay=1e-4)
    sch_m = CosineAnnealingLR(opt_m, T_max=MM_EP, eta_min=1e-5)

    best_auc = 0; hist = []
    for ep in range(1, MM_EP+1):
        tl,ta,tc = mm_epoch(lstm_enc,wx_mlp,fusion,tr_lm,opt_m,sch_m,bce,True)
        vl,va,vc = mm_epoch(lstm_enc,wx_mlp,fusion,va_lm,None, None, bce,False)
        hist.append({"ep":ep,"tr":ta,"va":va})
        if va > best_auc:
            best_auc = va
            torch.save({"lstm":lstm_enc.state_dict(),"wx":wx_mlp.state_dict(),
                        "fuse":fusion.state_dict()}, f"{MODEL_DIR}/best_fusion.pt")
        if ep % 8 == 0 or ep == 1:
            print(f"  Ep{ep:02d}/{MM_EP}  Train AUC:{ta:.3f} acc:{tc:.3f} | Val AUC:{va:.3f} acc:{vc:.3f}")
    print(f"\n  ✅ Best val AUC: {best_auc:.4f}")

    # ── Test evaluation ────────────────────────
    ck = torch.load(f"{MODEL_DIR}/best_fusion.pt", map_location=DEVICE)
    lstm_enc.load_state_dict(ck["lstm"]); wx_mlp.load_state_dict(ck["wx"])
    fusion.load_state_dict(ck["fuse"])

    PP, TT = [], []
    for ts, wx, img, lbl in te_lm:
        ts, wx, img = ts.to(DEVICE), wx.to(DEVICE), img.to(DEVICE)
        le, _ = lstm_enc(ts); we = wx_mlp(wx)
        lo,_,_ = fusion(img, le, we)
        PP.extend(torch.sigmoid(lo).cpu().detach().numpy())
        TT.extend(lbl.numpy())
    PP = np.array(PP); TT = np.array(TT).astype(int)
    test_auc = roc_auc_score(TT, PP)
    test_acc = ((PP>.5)==TT).mean()
    print(f"\n  Test AUC: {test_auc:.4f}  |  Accuracy: {test_acc:.4f}")
    print(classification_report(TT,(PP>.5).astype(int),target_names=["No Flare","Flare"]))

    # ── Save metrics & feature list ────────────
    metrics = {
        "test_auc":           round(float(test_auc), 4),
        "test_acc":           round(float(test_acc), 4),
        "cnn_best_val_acc":   round(float(best_ca),  4),
        "fusion_best_val_auc":round(float(best_auc), 4),
        "window":             WINDOW,
        "lifestyle_cols":     LIFESTYLE_COLS,
        "weather_cols":       WEATHER_COLS,
        "all_features":       ALL_FEATURES,
    }
    with open(f"{MODEL_DIR}/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ All model files saved in ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()
