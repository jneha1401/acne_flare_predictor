"""
utils/inference.py
Loads trained models and provides predict_risk() for the Streamlit app.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ── mirror model definitions (keep in sync with train_model.py) ────────────
LIFESTYLE_COLS = ["sleep_score","sugar_intake","dairy_intake","stress_level",
                  "water_intake","exercise","skincare_done"]
WEATHER_COLS   = ["temp_c","humidity_pct","precipitation","wind_speed"]
ALL_FEATURES   = LIFESTYLE_COLS + WEATHER_COLS
WINDOW         = 5
MODEL_DIR      = "models"
IMG_SIZE       = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
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
    def __init__(self, id=4, fd=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(id,64), nn.ReLU(), nn.Dropout(.2),
                                  nn.Linear(64,fd), nn.ReLU())
    def forward(self, x): return self.net(x)


class FusionModel(nn.Module):
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


class CNNEncoder(nn.Module):
    def __init__(self, fd=128, nc=4):
        super().__init__()
        try:
            import timm
            self.bb = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            in_feats = self.bb.num_features
        except Exception:
            self.bb = nn.Sequential(
                nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            in_feats = 32
        self.prj = nn.Sequential(nn.Linear(in_feats,256), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(256,fd), nn.ReLU())
        self.sh  = nn.Linear(fd, nc)

    def forward(self, x):
        e = self.prj(self.bb(x))
        return e, self.sh(e)


# ── Singleton model loader ──────────────────────────────────────────────────
_models = None
_mean_img_feat = None

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])


def _load_models():
    global _models, _mean_img_feat
    if _models is not None:
        return _models

    lstm_enc = LSTMEncoder(id=len(ALL_FEATURES)).to(DEVICE)
    wx_mlp   = WeatherMLP(id=len(WEATHER_COLS)).to(DEVICE)
    fusion   = FusionModel().to(DEVICE)
    cnn_enc  = CNNEncoder().to(DEVICE)

    fusion_path = os.path.join(MODEL_DIR, "best_fusion.pt")
    cnn_path    = os.path.join(MODEL_DIR, "best_cnn.pt")

    if os.path.exists(fusion_path):
        ck = torch.load(fusion_path, map_location=DEVICE)
        lstm_enc.load_state_dict(ck["lstm"])
        wx_mlp.load_state_dict(ck["wx"])
        fusion.load_state_dict(ck["fuse"])

    if os.path.exists(cnn_path):
        cnn_enc.load_state_dict(torch.load(cnn_path, map_location=DEVICE))

    for m in [lstm_enc, wx_mlp, fusion, cnn_enc]:
        m.eval()

    # Fallback image feature (zero vector used when no image provided)
    _mean_img_feat = torch.zeros(1, 128).to(DEVICE)

    _models = {"lstm": lstm_enc, "wx": wx_mlp, "fusion": fusion, "cnn": cnn_enc}
    return _models


def _normalise_weather(wx_today: dict) -> np.ndarray:
    """
    wx_today: {"temp_c": float, "humidity_pct": float,
                "precipitation": float, "wind_speed": float}
    Returns normalised array using precomputed scaler params if available.
    """
    scaler_path = "data/scaler_params.json"
    vals = np.array([wx_today.get(c, 0.0) for c in WEATHER_COLS], dtype=np.float32)

    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            sp = json.load(f)
        mn  = np.array(sp["min_"],   dtype=np.float32)
        sc  = np.array(sp["scale_"], dtype=np.float32)
        vals = (vals - mn) / (sc + 1e-8)
    else:
        # Simple normalisation if scaler not found
        norms = np.array([40., 100., 10., 30.], dtype=np.float32)
        vals  = vals / (norms + 1e-8)

    return np.clip(vals, 0, 1)


def predict_risk(lifestyle_5days: dict, weather_5days: list,
                 face_img: Image.Image = None) -> dict:
    """
    Parameters
    ----------
    lifestyle_5days : dict
        Keys = LIFESTYLE_COLS, values = list of 5 floats (0-1), one per day.
    weather_5days : list of dict
        5 dicts with keys: temp_c, humidity_pct, precipitation, wind_speed.
    face_img : PIL.Image or None
        Optional current face photo.

    Returns
    -------
    dict with keys:
        probability, risk_level, top_triggers,
        modality_weights, attn_by_day, day_labels
    """
    models = _load_models()

    # ── Build 5×11 sequence ─────────────────────
    rows = []
    for d in range(WINDOW):
        ls_row = [lifestyle_5days[c][d] for c in LIFESTYLE_COLS]
        wx_arr = _normalise_weather(weather_5days[d] if d < len(weather_5days)
                                    else {})
        rows.append(ls_row + wx_arr.tolist())
    ts  = torch.tensor([rows], dtype=torch.float32).to(DEVICE)  # (1,5,11)
    wx  = ts[0, -1, -len(WEATHER_COLS):].unsqueeze(0)           # (1,4) last day

    # ── CNN image embedding ─────────────────────
    if face_img is not None:
        with torch.no_grad():
            img_t = val_tfm(face_img.convert("RGB")).unsqueeze(0).to(DEVICE)
            img_f, _ = models["cnn"](img_t)
    else:
        img_f = _mean_img_feat

    # ── Forward pass ────────────────────────────
    with torch.no_grad():
        le, attn_w = models["lstm"](ts)
        we         = models["wx"](wx)
        lo, gate, _= models["fusion"](img_f, le, we)
        prob       = torch.sigmoid(lo).item()

    # ── Attention by day ────────────────────────
    attn = attn_w[0].cpu().numpy()                    # (5,)
    day_labels = [f"Day-{WINDOW-1-i}" for i in range(WINDOW)]

    # ── Top triggers: deviation from "safe" baseline ─
    safe = {"sleep_score":0.8, "sugar_intake":0.2, "dairy_intake":0.2,
            "stress_level":0.2, "water_intake":0.8, "exercise":0.8,
            "skincare_done":0.9}
    trigger_scores = {}
    for c in LIFESTYLE_COLS:
        avg = np.mean(lifestyle_5days[c])
        diff = avg - safe.get(c, 0.5)
        # Positive diff = bad for sleep/water/exercise/skincare → negate
        if c in ("sleep_score","water_intake","exercise","skincare_done"):
            diff = -diff
        trigger_scores[c] = round(float(diff), 3)

    # Weather triggers
    last_wx = weather_5days[-1] if weather_5days else {}
    if last_wx.get("humidity_pct", 0) > 70:
        trigger_scores["high_humidity"] = round((last_wx["humidity_pct"]-70)/30, 3)
    if last_wx.get("temp_c", 0) > 30:
        trigger_scores["high_temp"] = round((last_wx["temp_c"]-30)/10, 3)

    top_triggers = sorted(trigger_scores.items(), key=lambda x: -abs(x[1]))[:5]

    modality_weights = dict(zip(
        ["Image", "Lifestyle", "Weather"],
        gate[0].cpu().numpy().tolist()
    ))

    return {
        "probability":       round(prob, 4),
        "risk_level":        "HIGH" if prob > .70 else ("MODERATE" if prob > .40 else "LOW"),
        "top_triggers":      top_triggers,
        "modality_weights":  modality_weights,
        "attn_by_day":       dict(zip(day_labels, attn.tolist())),
        "day_labels":        day_labels,
        "attn_values":       attn.tolist(),
    }


def fetch_weather_api(lat: float, lon: float, days: int = WINDOW) -> list:
    """Fetch last `days` days of weather from Open-Meteo archive API."""
    import datetime, requests_cache
    today    = datetime.date.today()
    start_dt = today - datetime.timedelta(days=days - 1)

    cache = requests_cache.CachedSession(".weather_cache", expire_after=3600)
    url   = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat, "longitude": lon,
        "timezone":   "auto",
        "start_date": start_dt.isoformat(),
        "end_date":   today.isoformat(),
        "daily": ["temperature_2m_max","temperature_2m_min",
                  "relative_humidity_2m_max","relative_humidity_2m_min",
                  "precipitation_sum","wind_speed_10m_max"],
    }
    try:
        r = cache.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = r.json()["daily"]
        result = []
        for i in range(len(d["time"])):
            result.append({
                "date":         d["time"][i],
                "temp_c":       round((d["temperature_2m_max"][i]+d["temperature_2m_min"][i])/2, 1),
                "humidity_pct": round((d["relative_humidity_2m_max"][i]+d["relative_humidity_2m_min"][i])/2, 1),
                "precipitation":round(d["precipitation_sum"][i] or 0, 2),
                "wind_speed":   round(d["wind_speed_10m_max"][i] or 0, 1),
            })
        return result[-days:]
    except Exception as e:
        print(f"Weather API error: {e}")
        import random
        return [{"date": (today-datetime.timedelta(days=days-1-i)).isoformat(),
                 "temp_c": round(random.uniform(20,35),1),
                 "humidity_pct": round(random.uniform(40,80),1),
                 "precipitation": 0.0, "wind_speed": 10.0}
                for i in range(days)]
