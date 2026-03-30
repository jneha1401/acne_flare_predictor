"""
app.py — Acne Flare Predictor  (Streamlit frontend)
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json, datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AcneWatch — Personalised Flare Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f0f1a; color: #e8e8f0; }

    /* Cards */
    .risk-card {
        border-radius: 16px; padding: 28px 32px; margin: 8px 0;
        text-align: center; font-weight: 700;
    }
    .risk-HIGH    { background: linear-gradient(135deg,#ff4d4d22,#ff000033);
                    border: 2px solid #ff4d4d; }
    .risk-MODERATE{ background: linear-gradient(135deg,#ffa50022,#ff870033);
                    border: 2px solid #ffa500; }
    .risk-LOW     { background: linear-gradient(135deg,#00ff8822,#00cc6633);
                    border: 2px solid #00ff88; }

    .metric-box {
        background: #1a1a2e; border-radius: 12px; padding: 16px 20px;
        border: 1px solid #2a2a4a; margin: 4px;
    }
    .trigger-pill {
        display:inline-block; border-radius:20px; padding:5px 14px;
        font-size:0.82rem; margin:3px; font-weight:600;
    }
    .trigger-bad  { background:#ff4d4d33; border:1px solid #ff4d4d; color:#ff8888; }
    .trigger-good { background:#00ff8822; border:1px solid #00ff88; color:#00cc66; }

    h1,h2,h3 { color: #c9a0ff !important; }
    .stSlider > div > div > div > div { background: #c9a0ff !important; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white; border: none; border-radius: 10px;
        padding: 12px 32px; font-size: 1rem; font-weight: 600;
        width: 100%; transition: opacity .2s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .section-header {
        font-size:1.1rem; font-weight:700; color:#a78bfa;
        border-bottom:1px solid #2a2a4a; padding-bottom:6px; margin:16px 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR — location & image upload
# ══════════════════════════════════════════════
with st.sidebar:
    st.image("https://www.shutterstock.com/image-vector/cute-smiling-girl-cosmetic-patches-260nw-2006579096.jpg", width=64)
    st.title("AcneWatch ")
    st.caption("Personalised acne flare risk prediction using\nCNN + BiLSTM + Weather fusion")

    st.markdown("---")
    st.markdown("###  Your Location")
    lat = st.number_input("Latitude",  value=26.92, format="%.4f",
                           help="Used to fetch real weather data")
    lon = st.number_input("Longitude", value=75.82, format="%.4f")

    st.markdown("---")
    st.markdown("###  Face Photo (optional)")
    uploaded = st.file_uploader("Upload current face image",
                                 type=["jpg","jpeg","png"],
                                 help="Helps the CNN estimate acne severity")
    face_img = None
    if uploaded:
        face_img = Image.open(uploaded)
        st.image(face_img, caption="Uploaded", use_container_width=True)

    st.markdown("---")
    models_exist = (os.path.exists("models/best_fusion.pt") and
                    os.path.exists("models/best_cnn.pt"))
    if models_exist:
        st.success(" Models loaded")
        with open("models/metrics.json") as f:
            m = json.load(f)
        st.metric("Test AUC",  f"{m.get('test_auc', '—')}")
        st.metric("Test Acc",  f"{m.get('test_acc', '—')}")
    else:
        st.warning("  Models not found.\nRun `python train_model.py` first.\n\n"
                   "Running in **demo mode** with random predictions.")

# ══════════════════════════════════════════════
# HELPER — day labels
# ══════════════════════════════════════════════
today  = datetime.date.today()
DAYS   = [(today - datetime.timedelta(days=4-i)) for i in range(5)]
DAY_LABELS = [f"Day {d.strftime('%a %d')}" for d in DAYS]

# ══════════════════════════════════════════════
# QUESTIONS dict (module-level)
# ══════════════════════════════════════════════
QUESTIONS = {
    "sleep_score":  (" Sleep Quality",
                     "How well did you sleep? (0 = terrible, 1 = excellent)"),
    "sugar_intake": (" Sugar / High-GI Food",
                     "How much sugar/processed food did you eat? (0 = none, 1 = a lot)"),
    "dairy_intake": (" Dairy Intake",
                     "How much dairy did you consume? (0 = none, 1 = a lot)"),
    "stress_level": (" Stress Level",
                     "How stressed were you? (0 = calm, 1 = very stressed)"),
    "water_intake": (" Water Intake",
                     "How well hydrated were you? (0 = barely, 1 = well hydrated)"),
    "exercise":     (" Exercise",
                     "Did you exercise? (0 = none, 1 = full workout)"),
    "skincare_done":(" Skincare Routine",
                     "Did you complete your skincare routine? (0 = skipped, 1 = fully done)"),
}

# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; font-size:2.2rem;'>
   AcneWatch — Personalised Flare Predictor
</h1>
<p style='text-align:center; color:#888; margin-bottom:2rem;'>
  Combines your face image, 5-day lifestyle habits, and real weather data
  to predict your acne flare risk and highlight personal triggers.
</p>
""", unsafe_allow_html=True)

# ── Weather auto-fetch ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(lat, lon):
    try:
        from utils.inference import fetch_weather_api
        return fetch_weather_api(lat, lon, days=5)
    except Exception:
        return [{"date": d.isoformat(), "temp_c": 28.0, "humidity_pct": 65.0,
                 "precipitation": 0.0, "wind_speed": 12.0} for d in DAYS]

with st.spinner("🌤 Fetching weather for your location …"):
    weather_data = get_weather(lat, lon)

# ══════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([" Lifestyle Survey", " Weather Overview", " Results & Triggers"])

with tab1:
    st.markdown("##  Your 5-Day Lifestyle Log")
    st.info("Answer each question for the last 5 days. Sliders go 0 → 1 (Low → High).")

    lifestyle = {c: [] for c in
                 ["sleep_score","sugar_intake","dairy_intake","stress_level",
                  "water_intake","exercise","skincare_done"]}

    for feature, (label, hint) in QUESTIONS.items():
        st.markdown(f"<div class='section-header'>{label}</div>", unsafe_allow_html=True)
        st.caption(hint)
        cols = st.columns(5)
        for i, (col, dlabel) in enumerate(zip(cols, DAY_LABELS)):
            with col:
                val = col.slider(
                    dlabel, 0.0, 1.0, 0.5, 0.05,
                    key=f"{feature}_{i}",
                    label_visibility="visible"
                )
                lifestyle[feature].append(val)
        st.markdown("<br>", unsafe_allow_html=True)

with tab2:
    st.markdown("## 🌤 Weather Data (Auto-fetched)")
    if weather_data:
        wx_df = pd.DataFrame(weather_data)
        wx_cols = st.columns(len(weather_data))
        for col, row in zip(wx_cols, weather_data):
            with col:
                date_str = row.get("date", "—")
                t = row.get("temp_c", 0)
                h = row.get("humidity_pct", 0)
                p = row.get("precipitation", 0)
                icon = "🌧️" if p > 2 else ("🌥️" if h > 70 else "☀️")
                st.markdown(f"""
                <div class='metric-box' style='text-align:center'>
                  <div style='font-size:1.8rem'>{icon}</div>
                  <div style='font-size:0.75rem; color:#888'>{date_str}</div>
                  <div style='font-size:1.2rem; font-weight:700'>{t}°C</div>
                  <div style='font-size:0.85rem; color:#a0a0c0'>💧 {h}%</div>
                  <div style='font-size:0.8rem; color:#707090'>🌧 {p}mm</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Temperature & Humidity Trend")
        if len(wx_df) >= 2:
            fig = go.Figure()
            dates = [r.get("date","") for r in weather_data]
            fig.add_trace(go.Scatter(x=dates, y=[r["temp_c"] for r in weather_data],
                                     name="Temp (°C)", line=dict(color="#ff7c43", width=2.5)))
            fig.add_trace(go.Scatter(x=dates, y=[r["humidity_pct"] for r in weather_data],
                                     name="Humidity (%)", line=dict(color="#4cc9f0", width=2.5),
                                     yaxis="y2"))
            fig.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                font=dict(color="#e8e8f0"),
                yaxis=dict(title="Temperature (°C)", color="#ff7c43"),
                yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", color="#4cc9f0"),
                legend=dict(bgcolor="#1a1a2e"),
                height=280, margin=dict(t=20,b=20,l=20,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch weather — check your internet connection and coordinates.")

    st.info("  Weather is auto-fetched from **Open-Meteo archive API** . "
            "Change your coordinates in the sidebar to update.")

with tab3:
    st.markdown("##  Prediction Results")
    predict_btn = st.button("  Predict My Acne Flare Risk", use_container_width=True)

    if predict_btn:
        with st.spinner("Running multimodal inference …"):
            try:
                if models_exist:
                    from utils.inference import predict_risk
                    result = predict_risk(lifestyle, weather_data, face_img)
                else:
                    # Demo mode
                    prob = float(np.random.uniform(0.3, 0.85))
                    result = {
                        "probability":      round(prob, 4),
                        "risk_level":       "HIGH" if prob>.70 else ("MODERATE" if prob>.40 else "LOW"),
                        "top_triggers":     [("stress_level",0.3),("sleep_score",0.25),
                                             ("sugar_intake",0.15),("skincare_done",-0.2),
                                             ("humidity_avg",0.12)],
                        "modality_weights": {"Image":0.35,"Lifestyle":0.42,"Weather":0.23},
                        "attn_by_day":      {f"Day-{4-i}": round(0.12+i*0.02,3) for i in range(5)},
                        "day_labels":       [f"Day-{4-i}" for i in range(5)],
                        "attn_values":      [0.12,0.14,0.16,0.18,0.20],
                    }

                prob  = result["probability"]
                level = result["risk_level"]
                pct   = f"{prob*100:.1f}%"

                # ── Risk card ──────────────────────────────────
                color_map = {"HIGH":"#ff4d4d","MODERATE":"#ffa500","LOW":"#00ff88"}
                emoji_map = {"HIGH":"🔴","MODERATE":"🟠","LOW":"🟢"}
                color = color_map[level]
                st.markdown(f"""
                <div class='risk-card risk-{level}'>
                  <div style='font-size:3rem'>{emoji_map[level]}</div>
                  <div style='font-size:2.8rem; color:{color}'>{pct}</div>
                  <div style='font-size:1.4rem; color:{color}; margin-top:4px'>{level} RISK</div>
                  <div style='font-size:0.9rem; color:#aaa; margin-top:8px'>
                    Flare probability over next 24–48 hours
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Modality importance ────────────────────────
                st.markdown("###  What Drove This Prediction?")
                c1, c2, c3 = st.columns(3)
                mw = result["modality_weights"]
                for col, (mod, w) in zip([c1,c2,c3], mw.items()):
                    icon = {"Image":"📸","Lifestyle":"🏃","Weather":"🌤"}[mod]
                    with col:
                        st.markdown(f"""
                        <div class='metric-box' style='text-align:center'>
                          <div style='font-size:1.8rem'>{icon}</div>
                          <div style='font-size:0.85rem; color:#888'>{mod}</div>
                          <div style='font-size:1.5rem; font-weight:700; color:#c9a0ff'>
                            {w*100:.0f}%
                          </div>
                        </div>""", unsafe_allow_html=True)

                # ── Top triggers ───────────────────────────────
                st.markdown("###  Your Personal Triggers")
                trigger_labels = {
                    "sleep_score":   "Poor sleep",
                    "sugar_intake":  "High sugar intake",
                    "dairy_intake":  "Dairy consumption",
                    "stress_level":  "High stress",
                    "water_intake":  "Low hydration",
                    "exercise":      "No exercise",
                    "skincare_done": "Skipped skincare",
                    "high_humidity": "High humidity",
                    "high_temp":     "High temperature",
                }
                trigger_html = ""
                for feat, score in result["top_triggers"]:
                    nice = trigger_labels.get(feat, feat.replace("_"," ").title())
                    bad  = score > 0
                    cls  = "trigger-bad" if bad else "trigger-good"
                    arrow = "⬆ Worsening" if bad else "⬇ Protective"
                    trigger_html += (f"<span class='trigger-pill {cls}'>"
                                     f"{arrow}: {nice} ({score:+.2f})</span>")
                st.markdown(trigger_html, unsafe_allow_html=True)

                # ── Day-by-day attention ───────────────────────
                st.markdown("###  Which Days Influenced the Prediction Most?")
                attn = result["attn_values"]
                day_l= result["day_labels"]
                fig_attn = go.Figure(go.Bar(
                    x=day_l, y=attn,
                    marker=dict(color=attn, colorscale="Viridis", showscale=True,
                                colorbar=dict(title="Weight")),
                    text=[f"{v:.3f}" for v in attn], textposition="outside",
                ))
                fig_attn.update_layout(
                    paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                    font=dict(color="#e8e8f0"),
                    yaxis_title="Attention Weight", height=280,
                    margin=dict(t=20,b=20,l=20,r=20),
                )
                st.plotly_chart(fig_attn, use_container_width=True)

                # ── Lifestyle radar ────────────────────────────
                st.markdown("### 🕸 Lifestyle Profile (5-Day Average)")
                cats = list(QUESTIONS.keys())
                avgs = [np.mean(lifestyle[c]) for c in cats]
                nice_cats = [QUESTIONS[c][0].split(" ",1)[1] for c in cats]

                fig_radar = go.Figure(go.Scatterpolar(
                    r=avgs + [avgs[0]],
                    theta=nice_cats + [nice_cats[0]],
                    fill="toself",
                    fillcolor="rgba(168,85,247,0.2)",
                    line=dict(color="#a855f7", width=2),
                    marker=dict(size=6, color="#c9a0ff"),
                ))
                fig_radar.update_layout(
                    paper_bgcolor="#0f0f1a",
                    polar=dict(
                        bgcolor="#1a1a2e",
                        radialaxis=dict(visible=True, range=[0,1], color="#888"),
                        angularaxis=dict(color="#888"),
                    ),
                    font=dict(color="#e8e8f0"),
                    height=360, margin=dict(t=30,b=30,l=60,r=60),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # ── Personalised advice ────────────────────────
                st.markdown("###  Personalised Recommendations")

                # ✅ FIX: each value is a plain string, not a tuple
                advice_map = {
                    "sleep_score":   "😴 Aim for 7–9 hours of sleep tonight. Poor sleep raises cortisol, worsening acne.",
                    "sugar_intake":  "🍬 Cut high-GI foods. Sugar spikes insulin-like growth factor-1 (IGF-1), a key acne driver.",
                    "dairy_intake":  "🥛 Try reducing dairy for 2 weeks — multiple meta-analyses link it to flares.",
                    "stress_level":  "🧘 Try 10 min of mindfulness or deep breathing — stress raises sebum production.",
                    "water_intake":  "💧 Drink 2–3 L of water daily to flush toxins and maintain skin barrier.",
                    "exercise":      "🏃 Light exercise reduces cortisol and has anti-inflammatory effects on skin.",
                    "skincare_done": "🧴 Never skip your routine — barrier disruption allows bacteria to multiply overnight.",
                    "high_humidity": "💦 High humidity increases sweat and sebum — cleanse twice daily in this weather.",
                    "high_temp":     "☀️ Hot weather opens pores and increases oil. Use non-comedogenic SPF.",
                }

                # ✅ FIX: unpack as single string, not (icon, tip)
                for feat, score in result["top_triggers"]:
                    if feat in advice_map and score > 0.05:
                        tip = advice_map[feat]
                        st.markdown(f"- {tip}")

                st.success(" Prediction complete! Scroll up to explore all charts.")

            except Exception as e:
                st.error(f" Prediction error: {e}")
                import traceback; st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px; color:#555;'>
          <div style='font-size:3rem'></div>
          <div>Fill in the Lifestyle Survey and click <b>Predict</b> to see your results</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.8rem'>
  AcneWatch v1.0 · CNN (EfficientNet-B0) + BiLSTM + WeatherMLP + Cross-Attention Fusion ·
  Dataset: ACNE04 + Open-Meteo + Evidence-based synthetic lifestyle ·
  <b>Not medical advice</b>
</div>
""", unsafe_allow_html=True)