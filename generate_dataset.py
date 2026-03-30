"""
================================================
 Acne Flare Predictor — Dataset Generator
 Combines: ACNE04 images + Open-Meteo weather API
           + Evidence-based synthetic lifestyle logs
================================================
Run:  python generate_dataset.py
Output:
  data/lifestyle_logs.csv
  data/weather_data.csv
  data/merged_dataset.csv
  data/image_manifest.csv
================================================
"""

import os, json, random, warnings, datetime, math
import numpy as np
import pandas as pd
import requests
import requests_cache
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────
ACNE04_DIR   = "data/acne04_images"   # put your ACNE04 folders here
OUTPUT_DIR   = "data"
N_PATIENTS   = 200
N_DAYS       = 90
WINDOW_DAYS  = 5                       # sliding window used in training
RANDOM_SEED  = 42
# Jaipur coords — change if needed
LAT, LON, TZ = 26.92, 75.82, "Asia/Kolkata"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ACNE04 SEVERITY MAPPING (Hayashi / Wu et al. ICCV 2019) ────────────────
FOLDER_SEV = {
    "acne0_1024":          0,   # 0-5 lesions  → mild
    "acne1_1024":          1,   # 6-20 lesions → moderate
    "acne2_1024":          2,   # 21-50        → severe
    "acne3_1024":          3,   # 51+          → very severe
    "acne3_512_selection": 3,
}

# ── EVIDENCE-BASED FLARE PROBABILITY WEIGHTS ───────────────────────────────
# Sources documented in notebook literature table
EV = dict(
    sev_base    = [0.10, 0.30, 0.55, 0.80],   # baseline by severity level
    stress      = 0.15,    # Yosipovitch 2007, r=0.23
    sugar       = 0.10,    # Meixiong 2022, OR~3-25
    dairy       = 0.05,    # Juhl 2018 meta, OR=1.25
    sleep_bad   = 0.18,    # PMC12615111, OR=33
    humidity    = 0.08,    # Narang 2019, p<0.05
    temp_high   = 0.06,    # Yang 2020, p<0.05
    no_skincare = 0.12,    # clinical consensus
    no_exercise = 0.04,    # cortisol/anti-inflammatory
    low_water   = 0.03,
)


# ══════════════════════════════════════════════
# STEP 1 — Parse ACNE04 image manifest
# ══════════════════════════════════════════════
def build_image_manifest():
    import glob
    records = []
    for folder, sev in FOLDER_SEV.items():
        fp = os.path.join(ACNE04_DIR, folder)
        if not os.path.isdir(fp):
            print(f"  ⚠  {folder} not found — skipping")
            continue
        imgs = (glob.glob(os.path.join(fp, "*.jpg"))
              + glob.glob(os.path.join(fp, "*.jpeg"))
              + glob.glob(os.path.join(fp, "*.png")))
        for p in imgs:
            records.append({"image_path": p, "severity": sev})
        print(f"  {folder:30s} → severity {sev}  ({len(imgs)} images)")

    if not records:
        print("\n  ℹ  No ACNE04 images found — generating placeholder manifest")
        # Create placeholder so the rest of the pipeline still runs
        for sev in range(4):
            for i in range(10):
                records.append({"image_path": f"PLACEHOLDER_sev{sev}_{i}.jpg",
                                 "severity": sev})

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/image_manifest.csv", index=False)
    print(f"\n  ✅ Image manifest: {len(df)} rows saved → {OUTPUT_DIR}/image_manifest.csv")
    return df


# ══════════════════════════════════════════════
# STEP 2 — Fetch real weather (Open-Meteo archive API)
# ══════════════════════════════════════════════
def fetch_weather(lat=LAT, lon=LON, tz=TZ, days=N_DAYS):
    print(f"\n[Weather] Fetching {days} days from Open-Meteo archive …")
    today     = datetime.date.today()
    start_dt  = today - datetime.timedelta(days=days - 1)

    cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "timezone":   tz,
        "start_date": start_dt.isoformat(),
        "end_date":   today.isoformat(),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
    }

    try:
        resp = cache_session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        d = j["daily"]
        df = pd.DataFrame({
            "date":         pd.to_datetime(d["time"]),
            "temp_max":     d["temperature_2m_max"],
            "temp_min":     d["temperature_2m_min"],
            "humidity_max": d["relative_humidity_2m_max"],
            "humidity_min": d["relative_humidity_2m_min"],
            "precipitation":d["precipitation_sum"],
            "wind_speed":   d["wind_speed_10m_max"],
        })
        df["temp_avg"]     = (df["temp_max"]     + df["temp_min"])     / 2
        df["humidity_avg"] = (df["humidity_max"] + df["humidity_min"]) / 2
        df["date"]         = df["date"].dt.date
        print(f"  ✅ {len(df)} days fetched | "
              f"Temp {df.temp_avg.min():.1f}–{df.temp_avg.max():.1f}°C | "
              f"Humidity {df.humidity_avg.min():.0f}–{df.humidity_avg.max():.0f}%")
    except Exception as e:
        print(f"  ⚠  API error ({e}) — generating synthetic weather")
        dates = [start_dt + datetime.timedelta(days=i) for i in range(days)]
        df = pd.DataFrame({
            "date":         dates,
            "temp_avg":     np.random.uniform(18, 38, days),
            "humidity_avg": np.random.uniform(30, 85, days),
            "precipitation":np.random.exponential(2,  days),
            "wind_speed":   np.random.uniform(5,  25, days),
        })

    df.to_csv(f"{OUTPUT_DIR}/weather_data.csv", index=False)
    print(f"  Saved → {OUTPUT_DIR}/weather_data.csv")
    return df


# ══════════════════════════════════════════════
# STEP 3 — Evidence-based lifestyle log generator
# ══════════════════════════════════════════════
def _flare_prob(sev, stress, sugar, dairy, sleep, hum, temp, skincare, exercise, water):
    """Evidence-weighted flare probability (0-1)."""
    p = EV["sev_base"][sev]
    p += EV["stress"]      * stress
    p += EV["sugar"]       * sugar
    p += EV["dairy"]       * dairy
    p += EV["sleep_bad"]   * (1 - sleep)
    p += EV["humidity"]    * hum
    p += EV["temp_high"]   * temp
    p += EV["no_skincare"] * (1 - skincare)
    p += EV["no_exercise"] * (1 - exercise)
    p += EV["low_water"]   * (1 - water)
    return float(np.clip(p, 0.01, 0.99))


def generate_lifestyle_logs(weather_df, n_patients=N_PATIENTS, n_days=N_DAYS):
    print(f"\n[Lifestyle] Generating {n_patients} patients × {n_days} days …")
    # 50 patients per severity level
    patient_sevs = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    np.random.shuffle(patient_sevs)

    # Weather lookup: date → (temp_norm, hum_norm)
    from sklearn.preprocessing import MinMaxScaler
    wx = weather_df.copy()
    wx["date"] = pd.to_datetime(wx["date"]).dt.date
    wx = wx.set_index("date")
    for c in ["temp_avg","humidity_avg","precipitation","wind_speed"]:
        mn, mx = wx[c].min(), wx[c].max()
        wx[c+"_n"] = (wx[c]-mn)/(mx-mn+1e-8)
    wx_dates = list(wx.index)

    all_records = []
    for pid in range(n_patients):
        sev  = int(patient_sevs[pid])
        base = EV["sev_base"][sev]

        # Patient-level tendencies (personalized variation)
        tend_stress  = np.clip(np.random.normal(0.4, 0.2), 0, 1)
        tend_sugar   = np.clip(np.random.normal(0.4, 0.2), 0, 1)
        tend_dairy   = np.clip(np.random.normal(0.35,0.2), 0, 1)
        tend_sleep   = np.clip(np.random.normal(0.6, 0.2), 0, 1)
        tend_skin    = np.clip(np.random.normal(0.65,0.25),0, 1)
        tend_ex      = np.clip(np.random.normal(0.4, 0.2), 0, 1)
        tend_water   = np.clip(np.random.normal(0.6, 0.2), 0, 1)

        for di, date in enumerate(wx_dates[:n_days]):
            row_wx   = wx.loc[date] if date in wx.index else wx.iloc[-1]
            temp_n   = float(row_wx["temp_avg_n"])
            hum_n    = float(row_wx["humidity_avg_n"])
            prec_n   = float(row_wx.get("precipitation_n", 0))
            wind_n   = float(row_wx.get("wind_speed_n",    0))

            stress   = float(np.clip(tend_stress  + np.random.normal(0, 0.15), 0, 1))
            sugar    = float(np.clip(tend_sugar   + np.random.normal(0, 0.15), 0, 1))
            dairy    = float(np.clip(tend_dairy   + np.random.normal(0, 0.15), 0, 1))
            sleep    = float(np.clip(tend_sleep   + np.random.normal(0, 0.15), 0, 1))
            skincare = float(np.clip(tend_skin    + np.random.normal(0, 0.15), 0, 1))
            exercise = float(np.clip(tend_ex      + np.random.normal(0, 0.15), 0, 1))
            water    = float(np.clip(tend_water   + np.random.normal(0, 0.15), 0, 1))

            p = _flare_prob(sev, stress, sugar, dairy, sleep,
                            hum_n, temp_n, skincare, exercise, water)

            all_records.append({
                "patient_id":   pid,
                "patient_sev":  sev,
                "date":         date,
                "stress_level": round(stress,   4),
                "sugar_intake": round(sugar,    4),
                "dairy_intake": round(dairy,    4),
                "sleep_score":  round(sleep,    4),
                "water_intake": round(water,    4),
                "exercise":     round(exercise, 4),
                "skincare_done":round(skincare, 4),
                "temp_c":       round(float(row_wx["temp_avg"]),      2),
                "humidity_pct": round(float(row_wx["humidity_avg"]),  2),
                "precipitation":round(float(row_wx["precipitation"]), 2),
                "wind_speed":   round(float(row_wx["wind_speed"]),    2),
                "trigger_score":round(p, 4),
                "flare_label":  int(p > 0.50),
            })

    df = pd.DataFrame(all_records)
    rate_by_sev = {s: df[df.patient_sev==s].flare_label.mean() for s in range(4)}
    print(f"  ✅ {len(df):,} records generated")
    print(f"  Overall flare rate: {df.flare_label.mean():.1%}  (target: 40–55%)")
    for s,r in rate_by_sev.items():
        print(f"  Level {s}: {r:.1%}")

    df.to_csv(f"{OUTPUT_DIR}/lifestyle_logs.csv", index=False)
    print(f"  Saved → {OUTPUT_DIR}/lifestyle_logs.csv")
    return df


# ══════════════════════════════════════════════
# STEP 4 — Merge & normalise → merged_dataset.csv
# ══════════════════════════════════════════════
def merge_and_save(lifestyle_df):
    from sklearn.preprocessing import MinMaxScaler
    print(f"\n[Merge] Normalising weather columns …")
    df = lifestyle_df.copy()

    wx_cols = ["temp_c","humidity_pct","precipitation","wind_speed"]
    scaler  = MinMaxScaler()
    df[wx_cols] = scaler.fit_transform(df[wx_cols])

    df.to_csv(f"{OUTPUT_DIR}/merged_dataset.csv", index=False)
    print(f"  ✅ {len(df):,} rows | columns: {list(df.columns)}")
    print(f"  Saved → {OUTPUT_DIR}/merged_dataset.csv")

    # Also save scaler params for inference
    scaler_params = {
        "cols":   wx_cols,
        "min_":   scaler.data_min_.tolist(),
        "scale_": scaler.scale_.tolist(),
    }
    with open(f"{OUTPUT_DIR}/scaler_params.json","w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Scaler saved → {OUTPUT_DIR}/scaler_params.json")
    return df


# ══════════════════════════════════════════════
# STEP 5 — Quick dataset summary
# ══════════════════════════════════════════════
def print_summary(df):
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Rows      : {len(df):,}")
    print(f"  Patients  : {df.patient_id.nunique()}")
    print(f"  Days/pat  : {N_DAYS}")
    print(f"  Flare rate: {df.flare_label.mean():.1%}")
    print(f"  Severity distribution:")
    for s in range(4):
        n = (df.patient_sev==s).sum()
        print(f"    Level {s}: {n:,} records")
    print("="*55)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🧬  Acne Flare Predictor — Dataset Generator")
    print("="*55)

    print("\n[1/4] Parsing ACNE04 image manifest …")
    img_df = build_image_manifest()

    print("\n[2/4] Fetching weather data …")
    wx_df  = fetch_weather()

    print("\n[3/4] Generating lifestyle logs …")
    ls_df  = generate_lifestyle_logs(wx_df)

    print("\n[4/4] Merging & normalising …")
    final  = merge_and_save(ls_df)

    print_summary(final)
    print("\n✅  All done! Files saved in ./data/\n")
