import os
import io
import datetime
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ReportLab imports for official PDF Notice & Report generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & STANDARDIZED PRODUCTION STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Electricity Theft Detection System | AMI Smart Grid Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Generous top padding for clean header visibility */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    z-index: 1;
}
.block-container {
    padding-top: 3.8rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px;
}
section[data-testid="stSidebar"] { display: none; }

/* ── STANDARDIZED CONTAINER BLOCKS & CARDS ── */
.header-box {
    background: #0f172a;
    border: 1.5px solid #0284c7;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.header-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    line-height: 1.25 !important;
    margin: 0 0 0.35rem 0 !important;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.header-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    font-weight: 400;
    margin: 0;
    line-height: 1.4;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1rem;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #34d399;
    font-weight: 700;
}
.pulsing-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    display: inline-block;
}

/* ── STANDARDIZED METRIC CARDS ── */
.metric-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.metric-box:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
}
.metric-lbl {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0;
}
.c-cyan   { color: #38bdf8; }
.c-red    { color: #f87171; }
.c-amber  { color: #fbbf24; }
.c-green  { color: #34d399; }
.c-slate  { color: #cbd5e1; }
.c-gold   { color: #facc15; }
.c-emerald{ color: #10b981; }

/* ── DISPATCH & NORMALIZATION PANELS ── */
.dispatch-panel {
    background: #0f172a;
    border: 1.5px solid #0284c7;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.normalized-card {
    background: rgba(16, 185, 129, 0.08);
    border: 1.5px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.dp-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADING & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "mean", "std", "min", "max", "median", "range",
    "load_factor", "peak_to_avg", "cv",
    "mean_abs_diff", "std_diff",
    "day_mean", "night_mean", "day_night_ratio",
    "zero_pct", "low_pct", "max_low_streak",
]

MODELS_DIR = "models"
DATA_DEFAULT_PATH = os.path.join("data", "processed", "window_features.csv")
TARIFF_RATE_INR = 7.20  # Standard average tariff in AP & Telangana (₹/kWh)

@st.cache_resource
def load_detection_models():
    try:
        rf  = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        iso = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        lof = joblib.load(os.path.join(MODELS_DIR, "lof_model.pkl"))
        sc  = joblib.load(os.path.join(MODELS_DIR, "scaler_anomaly.pkl"))
        thr = joblib.load(os.path.join(MODELS_DIR, "thresholds.pkl"))
        threshold = thr.get("threshold", 0.2) if isinstance(thr, dict) else 0.2
        return rf, xgb, iso, lof, sc, threshold, None
    except Exception as e:
        return None, None, None, None, None, 0.2, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPREHENSIVE 30 DISTRIBUTION FEEDER LOCATIONS ACROSS AP & TELANGANA
# ─────────────────────────────────────────────────────────────────────────────
FEEDER_CONFIG = {
    # ── TELANGANA DISCOMS (TSSPDCL & TSNPDCL) ──
    "TSSPDCL - Feeder HYD-01 (Gachibowli)": {
        "area": "Gachibowli & Financial District, Hyderabad",
        "city": "Hyderabad",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 17.4435,
        "base_lon": 78.3772,
        "police_stn": "APTS / Vigilance Police Station, Mint Compound, Hyderabad",
    },
    "TSSPDCL - Feeder HYD-02 (Banjara Hills)": {
        "area": "Banjara Hills & Jubilee Hills, Hyderabad",
        "city": "Hyderabad",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 17.4156,
        "base_lon": 78.4350,
        "police_stn": "APTS / Vigilance Police Station, Hyderabad Central",
    },
    "TSSPDCL - Feeder SEC-03 (Secunderabad)": {
        "area": "Secunderabad Station & Marredpally",
        "city": "Secunderabad",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 17.4420,
        "base_lon": 78.4990,
        "police_stn": "APTS / Vigilance Police Station, Secunderabad",
    },
    "TSSPDCL - Feeder HYD-04 (Charminar Zone)": {
        "area": "Charminar & Old City Substation, Hyderabad",
        "city": "Hyderabad",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 17.3616,
        "base_lon": 78.4747,
        "police_stn": "APTS / Vigilance Police Station, South Zone Hyderabad",
    },
    "TSSPDCL - Feeder PTN-05 (Patancheru Industrial)": {
        "area": "Patancheru Heavy Industrial Belt, Sangareddy",
        "city": "Sangareddy",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 17.5312,
        "base_lon": 78.2618,
        "police_stn": "TSSPDCL Vigilance Police Station, Sangareddy",
    },
    "TSSPDCL - Feeder MDC-06 (Medchal & Kompally)": {
        "area": "Medchal Outer Ring & Kompally Hub",
        "city": "Medchal",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Commercial / Residential",
        "base_lat": 17.6294,
        "base_lon": 78.4815,
        "police_stn": "TSSPDCL Vigilance Police Station, Medchal",
    },
    "TSSPDCL - Feeder JDC-07 (Jadcherla & Mahbubnagar)": {
        "area": "Jadcherla Green Industrial Park, Mahbubnagar",
        "city": "Mahbubnagar",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 16.7628,
        "base_lon": 78.1402,
        "police_stn": "TSSPDCL Vigilance Police Station, Mahbubnagar",
    },
    "TSSPDCL - Feeder NLG-08 (Miryalaguda Rice Hub)": {
        "area": "Miryalaguda Agro & Rice Mills Corridor, Nalgonda",
        "city": "Nalgonda",
        "state": "Telangana",
        "discom": "TSSPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 16.8711,
        "base_lon": 79.5644,
        "police_stn": "TSSPDCL Vigilance Police Station, Nalgonda",
    },
    "TSNPDCL - Feeder WGL-09 (Warangal Subedari)": {
        "area": "Hanamkonda & Subedari Circle, Warangal",
        "city": "Warangal",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 17.9784,
        "base_lon": 79.5941,
        "police_stn": "TSNPDCL Vigilance Police Station, Hanamkonda",
    },
    "TSNPDCL - Feeder KNR-10 (Karimnagar Center)": {
        "area": "Clock Tower Substation, Karimnagar",
        "city": "Karimnagar",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 18.4386,
        "base_lon": 79.1288,
        "police_stn": "TSNPDCL Vigilance Police Station, Karimnagar",
    },
    "TSNPDCL - Feeder NZB-11 (Nizamabad Industrial)": {
        "area": "Khaleelwadi Commercial Feeder, Nizamabad",
        "city": "Nizamabad",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 18.6725,
        "base_lon": 78.0941,
        "police_stn": "TSNPDCL Vigilance Police Station, Nizamabad",
    },
    "TSNPDCL - Feeder KMM-12 (Khammam Wyra)": {
        "area": "Wyra Road Substation, Khammam",
        "city": "Khammam",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 17.2473,
        "base_lon": 80.1514,
        "police_stn": "TSNPDCL Vigilance Police Station, Khammam",
    },
    "TSNPDCL - Feeder RMD-13 (Ramagundam Industrial)": {
        "area": "Godavarikhani & NTPC Power Belt, Ramagundam",
        "city": "Ramagundam",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 18.7618,
        "base_lon": 79.4750,
        "police_stn": "TSNPDCL Vigilance Police Station, Ramagundam",
    },
    "TSNPDCL - Feeder MCH-14 (Mancherial Coal Belt)": {
        "area": "Bellampalli Mining Corridor, Mancherial",
        "city": "Mancherial",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 18.8682,
        "base_lon": 79.4589,
        "police_stn": "TSNPDCL Vigilance Police Station, Mancherial",
    },
    "TSNPDCL - Feeder SDP-15 (Siddipet Urban)": {
        "area": "Integrated Substation & Town Feeder, Siddipet",
        "city": "Siddipet",
        "state": "Telangana",
        "discom": "TSNPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 18.1018,
        "base_lon": 78.8520,
        "police_stn": "TSNPDCL Vigilance Police Station, Siddipet",
    },

    # ── ANDHRA PRADESH DISCOMS (APEPDCL, APCPDCL, APSPDCL) ──
    "APEPDCL - Feeder VZ-16 (Visakhapatnam MVP)": {
        "area": "MVP Colony & Beach Road, Visakhapatnam",
        "city": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 17.7289,
        "base_lon": 83.3328,
        "police_stn": "APTS Anti-Power Theft Police Station, Visakhapatnam",
    },
    "APEPDCL - Feeder VZ-17 (Gajuwaka Industrial)": {
        "area": "Gajuwaka Steel & Industrial Belt, Visakhapatnam",
        "city": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 17.6905,
        "base_lon": 83.2104,
        "police_stn": "APTS Anti-Power Theft Police Station, Gajuwaka",
    },
    "APEPDCL - Feeder KKD-18 (Kakinada Port Hub)": {
        "area": "Port Road & Commercial Circle, Kakinada",
        "city": "Kakinada",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 16.9891,
        "base_lon": 82.2475,
        "police_stn": "APTS Anti-Power Theft Police Station, Kakinada",
    },
    "APEPDCL - Feeder RJY-19 (Rajahmundry Morampudi)": {
        "area": "Morampudi Commercial Substation, Rajahmundry",
        "city": "Rajahmundry",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 17.0005,
        "base_lon": 81.8040,
        "police_stn": "APTS Anti-Power Theft Police Station, Rajahmundry",
    },
    "APEPDCL - Feeder SKL-20 (Srikakulam Palasa Hub)": {
        "area": "Palasa Cashew Agro & Town Feeder, Srikakulam",
        "city": "Srikakulam",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 18.2969,
        "base_lon": 83.8968,
        "police_stn": "APTS Anti-Power Theft Police Station, Srikakulam",
    },
    "APEPDCL - Feeder VZM-21 (Vizianagaram Fort)": {
        "area": "Cantonment & Industrial Estate, Vizianagaram",
        "city": "Vizianagaram",
        "state": "Andhra Pradesh",
        "discom": "APEPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 18.1133,
        "base_lon": 83.3977,
        "police_stn": "APTS Anti-Power Theft Police Station, Vizianagaram",
    },
    "APCPDCL - Feeder VJA-22 (Vijayawada Benz Circle)": {
        "area": "Benz Circle & MG Road, Vijayawada",
        "city": "Vijayawada",
        "state": "Andhra Pradesh",
        "discom": "APCPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 16.5062,
        "base_lon": 80.6480,
        "police_stn": "APTS Anti-Power Theft Police Station, Vijayawada",
    },
    "APCPDCL - Feeder GNT-23 (Guntur Arundelpet)": {
        "area": "Arundelpet & Brodipet Commercial, Guntur",
        "city": "Guntur",
        "state": "Andhra Pradesh",
        "discom": "APCPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 16.3067,
        "base_lon": 80.4365,
        "police_stn": "APTS Anti-Power Theft Police Station, Guntur",
    },
    "APCPDCL - Feeder ELR-24 (Bhimavaram Aqua Delta)": {
        "area": "Bhimavaram & Eluru Delta Aquaculture Belt",
        "city": "Eluru",
        "state": "Andhra Pradesh",
        "discom": "APCPDCL",
        "tariff": "Agriculture / Aqua (LT-IV)",
        "base_lat": 16.5449,
        "base_lon": 81.5212,
        "police_stn": "APTS Anti-Power Theft Police Station, Eluru",
    },
    "APCPDCL - Feeder ONG-25 (Ongole Granite Belt)": {
        "area": "Chimakkurthy Granite Mining & City, Ongole",
        "city": "Ongole",
        "state": "Andhra Pradesh",
        "discom": "APCPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 15.5057,
        "base_lon": 80.0499,
        "police_stn": "APTS Anti-Power Theft Police Station, Ongole",
    },
    "APSPDCL - Feeder TPT-26 (Tirupati Alipiri)": {
        "area": "Alipiri & Temple Substation, Tirupati",
        "city": "Tirupati",
        "state": "Andhra Pradesh",
        "discom": "APSPDCL",
        "tariff": "Commercial / Hospitality",
        "base_lat": 13.6288,
        "base_lon": 79.4192,
        "police_stn": "APTS Anti-Power Theft Police Station, Tirupati",
    },
    "APSPDCL - Feeder KNL-27 (Kurnool Collectorate)": {
        "area": "Collectorate Circle & Town, Kurnool",
        "city": "Kurnool",
        "state": "Andhra Pradesh",
        "discom": "APSPDCL",
        "tariff": "Residential (LT-I)",
        "base_lat": 15.8281,
        "base_lon": 78.0373,
        "police_stn": "APTS Anti-Power Theft Police Station, Kurnool",
    },
    "APSPDCL - Feeder NLR-28 (Nellore Krishnapatnam)": {
        "area": "Krishnapatnam Port & City Industrial, Nellore",
        "city": "Nellore",
        "state": "Andhra Pradesh",
        "discom": "APSPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 14.4426,
        "base_lon": 79.9865,
        "police_stn": "APTS Anti-Power Theft Police Station, Nellore",
    },
    "APSPDCL - Feeder ATP-29 (Anantapur & Hindupur)": {
        "area": "Hindupur Industrial Border & Anantapur",
        "city": "Anantapur",
        "state": "Andhra Pradesh",
        "discom": "APSPDCL",
        "tariff": "Industrial (LT-III)",
        "base_lat": 14.6819,
        "base_lon": 77.6006,
        "police_stn": "APTS Anti-Power Theft Police Station, Anantapur",
    },
    "APSPDCL - Feeder KDP-30 (Kadapa Proddatur)": {
        "area": "Proddatur Commercial Hub, Kadapa",
        "city": "Kadapa",
        "state": "Andhra Pradesh",
        "discom": "APSPDCL",
        "tariff": "Commercial (LT-II)",
        "base_lat": 14.4673,
        "base_lon": 78.8242,
        "police_stn": "APTS Anti-Power Theft Police Station, Kadapa",
    },
}

def assign_meter_locations(meter_ids):
    topology = {}
    feeders = list(FEEDER_CONFIG.keys())
    
    for i, m_id in enumerate(sorted(meter_ids)):
        fdr = feeders[i % len(feeders)]
        cfg = FEEDER_CONFIG[fdr]
        h = hash(str(m_id))
        lat_offset = ((h % 1000) / 1000.0 - 0.5) * 0.015
        lon_offset = (((h // 1000) % 1000) / 1000.0 - 0.5) * 0.015
        
        topology[m_id] = {
            "feeder": fdr,
            "area": cfg["area"],
            "city": cfg["city"],
            "state": cfg["state"],
            "discom": cfg["discom"],
            "tariff": cfg["tariff"],
            "police_stn": cfg["police_stn"],
            "lat": cfg["base_lat"] + lat_offset,
            "lon": cfg["base_lon"] + lon_offset,
        }
    return topology


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_pipeline(df_raw, rf, xgb, iso, lof, sc):
    df = df_raw.copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    X = df[FEATURE_COLS]

    rf_probs  = rf.predict_proba(X)[:, 1]
    xgb_probs = xgb.predict_proba(X)[:, 1]
    ens_probs = (rf_probs + xgb_probs) / 2.0

    X_scaled = sc.transform(X)
    iso_flag = (iso.predict(X_scaled) == -1).astype(int)
    lof_flag = (lof.predict(X_scaled) == -1).astype(int)

    zero_flag   = (X["zero_pct"] >= 0.5).astype(int).values
    streak_flag = (X["max_low_streak"] >= 16).astype(int).values
    cv_flag     = (X["cv"] >= 4.0).astype(int).values
    extreme_flag = ((streak_flag + cv_flag) >= 1).astype(int)

    scores = []
    labels = []
    tamper_types = []

    for p, i, l, z, ext, mean_val, cv_val, strk in zip(
        ens_probs, iso_flag, lof_flag, zero_flag, extreme_flag, 
        X["mean"], X["cv"], X["max_low_streak"]
    ):
        score = 0
        if p >= 0.6:
            score += 2
        elif p >= 0.4:
            score += 1

        if i == 1 and l == 1:
            score += 2
        elif i == 1 or l == 1:
            score += 1

        score += int(z)
        score += int(ext)
        capped_score = min(score, 5)
        scores.append(capped_score)

        if capped_score >= 3:
            lvl = "Theft Detected"
        elif capped_score >= 1:
            lvl = "Suspicious"
        else:
            lvl = "Normal"
        labels.append(lvl)

        if lvl != "Normal":
            if z >= 0.5 or strk >= 18:
                tamper_types.append("Direct Bypass (Meter bypassed, recording ~0 usage)")
            elif cv_val >= 3.5:
                tamper_types.append("Alternate Hour Theft (Tampered during specific hours)")
            elif mean_val < 0.2 and p >= 0.5:
                tamper_types.append("Sudden Drop (Usage sharply dropped by ~80%)")
            elif p >= 0.5:
                tamper_types.append("Load Suppression (Reading only 30% to 50% of real power)")
            else:
                tamper_types.append("Flatline Tamper (Readings artificially kept constant)")
        else:
            tamper_types.append("Normal Power Consumption")

    df["fraud_prob"]   = ens_probs
    df["iso_flag"]     = iso_flag
    df["lof_flag"]     = lof_flag
    df["confidence"]   = scores
    df["risk_level"]   = labels
    df["tamper_type"]  = tamper_types

    return df


def assign_base_status(row):
    tot = row["total_days"]
    theft_ratio = row["theft_days"] / tot
    susp_ratio = row["suspicious_days"] / tot
    if theft_ratio >= 0.05:
        return "Theft Detected"
    elif theft_ratio >= 0.015 or susp_ratio >= 0.30:
        return "Suspicious"
    return "Normal"


# ─────────────────────────────────────────────────────────────────────────────
# 5. CACHED DATA PIPELINE (@st.cache_data)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_score_dataset(data_source, _rf, _xgb, _iso, _lof, _sc):
    if isinstance(data_source, (bytes, bytearray)):
        df_raw = pd.read_csv(io.BytesIO(data_source))
    else:
        df_raw = pd.read_csv(data_source)

    # Ensure required tracking columns exist
    if "meter_id" not in df_raw.columns:
        df_raw["meter_id"] = [f"METER_{i+1:04d}" for i in range(len(df_raw))]
    if "window_id" not in df_raw.columns:
        df_raw["window_id"] = df_raw.groupby("meter_id").cumcount()
    if "theft" not in df_raw.columns:
        df_raw["theft"] = 0

    df_scored = run_detection_pipeline(df_raw, _rf, _xgb, _iso, _lof, _sc)

    unique_meters = df_scored["meter_id"].unique()
    topology_map = assign_meter_locations(unique_meters)

    for col in ["feeder", "area", "city", "state", "discom", "tariff", "police_stn", "lat", "lon"]:
        df_scored[col] = df_scored["meter_id"].map(lambda m: topology_map[m][col])

    meter_summary = df_scored.groupby("meter_id").agg(
        total_days      = ("risk_level", "count"),
        theft_days      = ("risk_level", lambda x: (x == "Theft Detected").sum()),
        suspicious_days = ("risk_level", lambda x: (x == "Suspicious").sum()),
        normal_days     = ("risk_level", lambda x: (x == "Normal").sum()),
        avg_confidence  = ("confidence", "mean"),
        max_confidence  = ("confidence", "max"),
        avg_fraud_prob  = ("fraud_prob", "mean"),
        feeder          = ("feeder", "first"),
        area            = ("area", "first"),
        city            = ("city", "first"),
        state           = ("state", "first"),
        discom          = ("discom", "first"),
        tariff          = ("tariff", "first"),
        police_stn      = ("police_stn", "first"),
        lat             = ("lat", "first"),
        lon             = ("lon", "first"),
        avg_kwh         = ("mean", "mean"),
        dominant_theft  = ("tamper_type", lambda s: [v for v in s if v != "Normal Power Consumption"][0] if any(v != "Normal Power Consumption" for v in s) else "Normal Power Consumption"),
    ).reset_index()

    meter_summary["base_status"] = meter_summary.apply(assign_base_status, axis=1)
    return df_scored, meter_summary, topology_map


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORTLAB PDF ENGINES: NOTICE & RESTORATION CERTIFICATE
# ─────────────────────────────────────────────────────────────────────────────
def generate_inspection_pdf(meter_info, ticket_info=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=colors.HexColor('#0f172a'), alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor('#0284c7'), alignment=1)
    header_bar_style = ParagraphStyle('HeaderBar', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.white, alignment=1)
    label_style = ParagraphStyle('Label', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    val_style = ParagraphStyle('Value', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor('#0f172a'))
    alert_val_style = ParagraphStyle('AlertValue', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#dc2626'))

    # Header
    story.append(Paragraph("STATE ELECTRICITY REGULATORY COMMISSION", title_style))
    story.append(Spacer(1, 2))
    discom_name = meter_info.get('discom', 'DISCOM')
    story.append(Paragraph(f"ANTI-POWER THEFT SQUAD (APTS) &bull; {discom_name} VIGILANCE WING", subtitle_style))
    story.append(Spacer(1, 8))

    banner_text = Paragraph("OFFICIAL ELECTRICITY THEFT FIELD INSPECTION NOTICE & WARRANT", header_bar_style)
    banner_table = Table([[banner_text]], colWidths=[540], rowHeights=[22])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    now_str = datetime.datetime.now().strftime("%d-%b-%Y %H:%M HRS")
    notice_no = f"APTS/{discom_name}/{datetime.datetime.now().strftime('%Y%m%d')}/{str(meter_info.get('meter_id', '0000'))[-4:]}"
    meta_data = [
        [Paragraph(f"<b>Notice Reference No:</b> {notice_no}", label_style), Paragraph(f"<b>Issuance Timestamp:</b> {now_str}", val_style)],
        [Paragraph(f"<b>Statutory Provision:</b> Section 135 & 138 Electricity Act, 2003", label_style), Paragraph(f"<b>Jurisdiction Station:</b> {meter_info.get('police_stn', 'APTS Police')}", val_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
    story.append(meta_table)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=5, spaceAfter=8))

    # Identification
    story.append(Paragraph("<b>1. CONSUMER & SMART METER IDENTIFICATION</b>", label_style))
    story.append(Spacer(1, 4))
    det_data = [
        [Paragraph("Consumer Smart Meter ID", label_style), Paragraph(f"<b>{meter_info.get('meter_id')}</b>", val_style),
         Paragraph("Jurisdiction Discom", label_style), Paragraph(str(meter_info.get('discom')), val_style)],
        [Paragraph("Distribution Feeder", label_style), Paragraph(str(meter_info.get('feeder')), val_style),
         Paragraph("State / Territory", label_style), Paragraph(str(meter_info.get('state')), val_style)],
        [Paragraph("Substation / Area", label_style), Paragraph(str(meter_info.get('area')), val_style),
         Paragraph("Municipality / City", label_style), Paragraph(str(meter_info.get('city')), val_style)],
        [Paragraph("Tariff Category", label_style), Paragraph(str(meter_info.get('tariff')), val_style),
         Paragraph("GPS Geo-Coordinates", label_style), Paragraph(f"{meter_info.get('lat', 0):.5f}&deg; N, {meter_info.get('lon', 0):.5f}&deg; E", val_style)],
    ]
    det_table = Table(det_data, colWidths=[135, 135, 135, 135])
    det_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 8))

    # Forensic Evidence
    story.append(Paragraph("<b>2. AI FORENSIC DETECTION & REVENUE LOSS ASSESSMENT</b>", label_style))
    story.append(Spacer(1, 4))
    tot_days = meter_info.get('total_days', 365)
    theft_days = meter_info.get('theft_days', 0)
    avg_kwh = meter_info.get('avg_kwh', 0.0)
    est_stolen_kwh = theft_days * 24.0 * max(0.5, avg_kwh * 0.85)
    est_loss_inr = est_stolen_kwh * TARIFF_RATE_INR
    is_theft = meter_info.get('meter_status') == 'Theft Detected'

    ev_data = [
        [Paragraph("Classification Status", label_style), Paragraph(f"<b>{meter_info.get('meter_status')}</b>", alert_val_style if is_theft else val_style),
         Paragraph("Dominant Tampering Signature", label_style), Paragraph(str(meter_info.get('dominant_theft')), alert_val_style if is_theft else val_style)],
        [Paragraph("Tampered Days Count", label_style), Paragraph(f"<b>{theft_days}</b> of {tot_days} Days ({theft_days/tot_days*100:.1f}%)", val_style),
         Paragraph("ML Fraud Probability", label_style), Paragraph(f"<b>{meter_info.get('avg_fraud_prob', 0)*100:.1f}%</b> (Supervised Ensemble)", val_style)],
        [Paragraph("Est. Stolen Electricity", label_style), Paragraph(f"<b>{est_stolen_kwh:,.1f} kWh</b>", alert_val_style if is_theft else val_style),
         Paragraph("Est. Financial Revenue Loss", label_style), Paragraph(f"<b>₹ {est_loss_inr:,.2f}</b> (₹ {est_loss_inr/100000:.2f} Lakhs)", alert_val_style if is_theft else val_style)],
    ]
    ev_table = Table(ev_data, colWidths=[135, 135, 135, 135])
    ev_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#fef2f2') if is_theft else colors.HexColor('#f0fdf4')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#fef2f2') if is_theft else colors.HexColor('#f0fdf4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#fca5a5') if is_theft else colors.HexColor('#86efac')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ev_table)
    story.append(Spacer(1, 8))

    # Field Work Order Details
    if ticket_info:
        story.append(Paragraph("<b>3. ACTIVE FIELD DISPATCH & SQUAD ASSIGNMENT</b>", label_style))
        story.append(Spacer(1, 4))
        disp_data = [
            [Paragraph("Work Order ID", label_style), Paragraph(f"<b>{ticket_info.get('Work Order ID')}</b>", val_style),
             Paragraph("Assigned Squad", label_style), Paragraph(str(ticket_info.get('Assigned Squad')), val_style)],
            [Paragraph("Urgency Priority", label_style), Paragraph(f"<b>{ticket_info.get('Urgency')}</b>", alert_val_style),
             Paragraph("Dispatch Timestamp", label_style), Paragraph(str(ticket_info.get('Dispatched At')), val_style)],
            [Paragraph("Mandatory Directives", label_style), Paragraph(str(ticket_info.get('Instructions')), val_style), "", ""],
        ]
        disp_table = Table(disp_data, colWidths=[135, 135, 135, 135])
        disp_table.setStyle(TableStyle([
            ('SPAN', (1, 2), (3, 2)),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f0f9ff')),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f0f9ff')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bae6fd')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(disp_table)
        story.append(Spacer(1, 8))

    story.append(Paragraph("<b>STATUTORY MANDATE & LEGAL PROVISIONS:</b>", label_style))
    legal_text = (
        "Under Section 135 and 138 of the Indian Electricity Act 2003, any unauthorized tapping, meter tampering, "
        "CT/PT reversal, or artificial slowing of energy recording constitutes a cognizable offense punishable with "
        "imprisonment up to 3 years and/or compounding financial penalties."
    )
    story.append(Paragraph(legal_text, ParagraphStyle('Legal', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.2, leading=9.5, textColor=colors.HexColor('#475569'))))
    story.append(Spacer(1, 22))

    sig_data = [
        [Paragraph("____________________________<br/><b>Divisional Assistant Engineer (DAE)</b><br/>Discom Operations Division", val_style),
         Paragraph("____________________________<br/><b>APTS Vigilance Inspector</b><br/>Anti-Power Theft Police Station", val_style),
         Paragraph("____________________________<br/><b>Field Squad Leader</b><br/>Technical Verification Unit", val_style)]
    ]
    sig_table = Table(sig_data, colWidths=[180, 180, 180])
    sig_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(sig_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_restoration_certificate_pdf(meter_info, ticket_info, resolution_info):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=colors.HexColor('#0f172a'), alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor('#059669'), alignment=1)
    header_bar_style = ParagraphStyle('HeaderBar', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.white, alignment=1)
    label_style = ParagraphStyle('Label', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    val_style = ParagraphStyle('Value', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor('#0f172a'))
    success_val_style = ParagraphStyle('SuccessVal', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#059669'))

    story.append(Paragraph("STATE ELECTRICITY REGULATORY COMMISSION", title_style))
    story.append(Spacer(1, 2))
    discom_name = meter_info.get('discom', 'DISCOM')
    story.append(Paragraph(f"ANTI-POWER THEFT SQUAD (APTS) &bull; {discom_name} ENFORCEMENT & COMPLIANCE WING", subtitle_style))
    story.append(Spacer(1, 8))

    banner_text = Paragraph("OFFICIAL CERTIFICATE OF METER RESTORATION & PENALTY SETTLEMENT", header_bar_style)
    banner_table = Table([[banner_text]], colWidths=[540], rowHeights=[22])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#065f46')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    cert_no = f"APTS-CERT/{discom_name}/{datetime.datetime.now().strftime('%Y%m%d')}/{str(meter_info.get('meter_id', '0000'))[-4:]}"
    now_str = datetime.datetime.now().strftime("%d-%b-%Y %H:%M HRS")
    meta_data = [
        [Paragraph(f"<b>Certificate Reference No:</b> {cert_no}", label_style), Paragraph(f"<b>Certification Date:</b> {now_str}", val_style)],
        [Paragraph(f"<b>Statutory Compliance:</b> Section 152 Electricity Act, 2003", label_style), Paragraph(f"<b>Enforcement Authority:</b> {meter_info.get('police_stn', 'APTS Police')}", val_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
    story.append(meta_table)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=5, spaceAfter=8))

    # Installation Details
    story.append(Paragraph("<b>1. CONSUMER & INSTALLATION DETAILS</b>", label_style))
    story.append(Spacer(1, 4))
    det_data = [
        [Paragraph("Consumer Smart Meter ID", label_style), Paragraph(f"<b>{meter_info.get('meter_id')}</b>", val_style),
         Paragraph("Jurisdiction Discom", label_style), Paragraph(str(meter_info.get('discom')), val_style)],
        [Paragraph("Distribution Feeder", label_style), Paragraph(str(meter_info.get('feeder')), val_style),
         Paragraph("City & State", label_style), Paragraph(f"{meter_info.get('city')}, {meter_info.get('state')}", val_style)],
        [Paragraph("Tariff Category", label_style), Paragraph(str(meter_info.get('tariff')), val_style),
         Paragraph("GPS Geo-Coordinates", label_style), Paragraph(f"{meter_info.get('lat', 0):.5f}&deg; N, {meter_info.get('lon', 0):.5f}&deg; E", val_style)],
    ]
    det_table = Table(det_data, colWidths=[135, 135, 135, 135])
    det_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 8))

    # Field Rectification Findings
    story.append(Paragraph("<b>2. FIELD RECTIFICATION & RESTORATION FINDINGS</b>", label_style))
    story.append(Spacer(1, 4))
    rect_data = [
        [Paragraph("Original Anomaly Flagged", label_style), Paragraph(str(meter_info.get('dominant_theft', 'Tampering Detected')), val_style),
         Paragraph("Inspection Work Order ID", label_style), Paragraph(str(ticket_info.get('Work Order ID') if ticket_info else 'DIRECT-DISPATCH'), val_style)],
        [Paragraph("Inspecting Squad", label_style), Paragraph(str(resolution_info.get('squad_name', 'APTS Technical Team')), val_style),
         Paragraph("Restoration Timestamp", label_style), Paragraph(str(resolution_info.get('resolved_at', now_str)), val_style)],
        [Paragraph("Field Action Executed", label_style), Paragraph(f"<b>{resolution_info.get('action_taken', 'Tamper removed & recalibrated')}</b>", success_val_style),
         Paragraph("New Tamper-Proof Seal ID", label_style), Paragraph(f"<b>{resolution_info.get('new_seal_no', 'N/A')}</b>", success_val_style)],
        [Paragraph("Compounding Penalty Paid", label_style), Paragraph(f"<b>₹ {resolution_info.get('fine_collected', 0):,.2f}</b>", success_val_style),
         Paragraph("Post-Inspection Meter Status", label_style), Paragraph("<b>NORMAL & COMPLIANT</b>", success_val_style)],
    ]
    rect_table = Table(rect_data, colWidths=[135, 135, 135, 135])
    rect_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ecfdf5')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#ecfdf5')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#a7f3d0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(rect_table)
    story.append(Spacer(1, 8))

    # Clearance Declaration
    story.append(Paragraph("<b>3. REGULATORY CLEARANCE & LEGAL SETTLEMENT DECLARATION</b>", label_style))
    story.append(Spacer(1, 4))
    decl_text = (
        f"This is to certify that smart meter <b>{meter_info.get('meter_id')}</b> installed at the above premises "
        "has been fully inspected by the authorized APTS Vigilance & Technical Team. All illegal bypass lines, shunts, "
        "and unauthorized meter modifications have been removed and seized as evidence. A new official tamper-evident "
        f"meter seal (ID: <b>{resolution_info.get('new_seal_no', 'N/A')}</b>) has been affixed. "
        f"The compounding assessment fee / civil penalty of <b>₹ {resolution_info.get('fine_collected', 0):,.2f}</b> has been duly "
        "settled under Section 152 of the Indian Electricity Act 2003. Power service is hereby officially restored to "
        "regular statutory billing status. The consumer profile is restored to <b>NORMAL</b> in the central utility SCADA and AMI grid databases."
    )
    story.append(Paragraph(decl_text, ParagraphStyle('Decl', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#334155'))))
    story.append(Spacer(1, 24))

    sig_data = [
        [Paragraph("____________________________<br/><b>Divisional Assistant Engineer (DAE)</b><br/>Discom Operations Division", val_style),
         Paragraph("____________________________<br/><b>APTS Vigilance Inspector</b><br/>Anti-Power Theft Police Station", val_style),
         Paragraph("____________________________<br/><b>Consumer / Representative</b><br/>Acknowledgment of Rectification", val_style)]
    ]
    sig_table = Table(sig_data, colWidths=[180, 180, 180])
    sig_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(sig_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 7. INITIALIZE SESSION STATE WITH RESOLUTION LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
if "dispatch_log" not in st.session_state:
    st.session_state.dispatch_log = []
if "normalized_meters" not in st.session_state:
    st.session_state.normalized_meters = {}
if "map_view_mode" not in st.session_state:
    st.session_state.map_view_mode = "🌐 Show All Meters (Full Grid Network - 60 Meters)"
if "state_filter" not in st.session_state:
    st.session_state.state_filter = "All States (AP & TS)"
if "discom_filter" not in st.session_state:
    st.session_state.discom_filter = "All Discoms"
if "status_filter" not in st.session_state:
    st.session_state.status_filter = ["Theft Detected", "Suspicious", "Normal"]
if "selected_meter_alert" not in st.session_state:
    st.session_state.selected_meter_alert = None
if "meter_cat_filter" not in st.session_state:
    st.session_state.meter_cat_filter = "All Meters (60)"
if "selected_detail_meter" not in st.session_state:
    st.session_state.selected_detail_meter = None


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOAD MODELS & CACHED INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
rf, xgb, iso, lof, sc, threshold, err = load_detection_models()

# ── TOP HEADER ──
st.markdown("""
<div class="header-box">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
            <div class="header-title">
                ⚡ AI-POWERED ELECTRICITY THEFT DETECTION SYSTEM
            </div>
            <div class="header-subtitle">
                A Hybrid Machine Learning, Geospatial AMI Telemetry & Closed-Loop Field Normalization Platform
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.35rem; font-weight: 500;">
                Telangana & Andhra Pradesh Power Distribution Companies (TSSPDCL &middot; TSNPDCL &middot; APEPDCL &middot; APCPDCL &middot; APSPDCL)
            </div>
        </div>
        <div>
            <span class="live-pill">
                <span class="pulsing-dot"></span> SMART METER MONITORING ACTIVE
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if err:
    st.error(f"❌ Error loading ML models from `models/`: {err}. Please run `python run_pipeline.py` first.")
    st.stop()

# ── DATA SOURCE CONTROLS & INTERACTIVE CSV UPLOADER ──
SAMPLE_CSV_DATA = """meter_id,window_id,mean,std,min,max,median,range,load_factor,peak_to_avg,cv,mean_abs_diff,std_diff,day_mean,night_mean,day_night_ratio,zero_pct,low_pct,max_low_streak
M_DEMO_NORMAL_01,0,0.0165,0.0110,0.0001,0.0326,0.0155,0.0325,0.506,1.973,0.670,0.0044,0.0053,0.0208,0.0122,1.702,0.0,0.166,4
M_DEMO_NORMAL_01,1,0.0172,0.0115,0.0001,0.0341,0.0160,0.0340,0.510,1.960,0.668,0.0046,0.0055,0.0215,0.0128,1.680,0.0,0.150,3
M_DEMO_NORMAL_01,2,0.0158,0.0108,0.0001,0.0310,0.0149,0.0309,0.509,1.962,0.683,0.0042,0.0051,0.0199,0.0117,1.701,0.0,0.160,4
M_DEMO_THEFT_02,0,0.0007,0.0004,0.000003,0.0014,0.0006,0.0014,0.509,1.961,0.671,0.00019,0.00023,0.0009,0.0005,1.703,0.45,0.850,24
M_DEMO_THEFT_02,1,0.0005,0.0003,0.000002,0.0010,0.0005,0.0010,0.512,1.950,0.669,0.00014,0.00017,0.0006,0.0004,1.708,0.52,0.920,28
M_DEMO_THEFT_02,2,0.0004,0.0002,0.000001,0.0009,0.0004,0.0009,0.485,2.060,0.710,0.00012,0.00015,0.0005,0.0003,1.666,0.60,0.950,30
M_DEMO_SUSP_03,0,0.0125,0.0195,0.00005,0.0450,0.0080,0.0449,0.277,3.600,1.560,0.0062,0.0088,0.0180,0.0070,2.571,0.15,0.350,12
M_DEMO_SUSP_03,1,0.0085,0.0160,0.00002,0.0390,0.0050,0.0389,0.217,4.588,1.882,0.0055,0.0079,0.0130,0.0040,3.250,0.22,0.450,15
M_DEMO_SUSP_03,2,0.0110,0.0180,0.00004,0.0420,0.0070,0.0419,0.261,3.818,1.636,0.0058,0.0082,0.0160,0.0060,2.666,0.18,0.400,14
"""

with st.expander("📂 Telemetry Data Source & Custom CSV Uploader (Click to Upload or Download Template)", expanded=False):
    col_up1, col_up2 = st.columns([3, 2], gap="large")
    with col_up1:
        st.markdown("##### 📤 Upload Custom Smart Meter Telemetry CSV")
        st.caption("Upload any smart meter dataset (with meter_id and feature columns) to run instant AI inference across the grid.")
        uploaded_file = st.file_uploader(
            "Select Telemetry CSV File:",
            type=["csv"],
            key="custom_meter_uploader",
            help="Upload custom smart meter CSV to test AI fraud detection."
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            if st.session_state.get("custom_csv_name") != uploaded_file.name or st.session_state.get("custom_csv_bytes") != raw_bytes:
                st.session_state["custom_csv_bytes"] = raw_bytes
                st.session_state["custom_csv_name"] = uploaded_file.name
                st.toast(f"Uploaded and analyzing `{uploaded_file.name}`!", icon="📁")

    with col_up2:
        st.markdown("##### 📥 Benchmark Test Template & Reset")
        st.caption("Need a test file? Download our 3-meter test benchmark (Normal, Neutral Bypass & Suspicious):")
        st.download_button(
            label="📥 Download Sample Telemetry CSV Template",
            data=SAMPLE_CSV_DATA,
            file_name="sample_smart_meter_telemetry.csv",
            mime="text/csv",
            use_container_width=True
        )
        if "custom_csv_bytes" in st.session_state and st.session_state["custom_csv_bytes"]:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset to Default 60-Meter Grid Telemetry", use_container_width=True):
                del st.session_state["custom_csv_bytes"]
                del st.session_state["custom_csv_name"]
                st.toast("Restored default 30-feeder grid telemetry!", icon="⚡")
                st.rerun()

# Determine active telemetry source
if "custom_csv_bytes" in st.session_state and st.session_state["custom_csv_bytes"]:
    data_source = st.session_state["custom_csv_bytes"]
    active_source_name = st.session_state.get("custom_csv_name", "Uploaded Custom File")
    is_custom_dataset = True
    st.info(f"📁 **Active Data Stream:** Custom Uploaded File `{active_source_name}` &mdash; Live ML Classification Active! (To revert, open the expander above and click 'Reset').")
else:
    if not os.path.exists(DATA_DEFAULT_PATH):
        st.warning("No preprocessed dataset found. Please run the training pipeline first.")
        st.stop()
    data_source = DATA_DEFAULT_PATH
    active_source_name = "AP & Telangana Smart Grid (30 Feeders · 60 Monitored Meters)"
    is_custom_dataset = False

# Cached inference & aggregation
df_scored, meter_summary_base, topology_map = load_and_score_dataset(data_source, rf, xgb, iso, lof, sc)

# Apply dynamic resolution lifecycle: normalized meters become Normal
meter_summary = meter_summary_base.copy()

def get_effective_status(row):
    m_id = str(row["meter_id"])
    if m_id in st.session_state.normalized_meters:
        return "Normal"
    return row["base_status"]

meter_summary["meter_status"] = meter_summary.apply(get_effective_status, axis=1)
meter_summary["is_normalized"] = meter_summary["meter_id"].map(lambda m: str(m) in st.session_state.normalized_meters)

# Sync dynamic dispatch status from session_state
dispatched_meters = {item["Meter ID"]: item["Status"] for item in st.session_state.dispatch_log}
meter_summary["dispatch_status"] = meter_summary["meter_id"].map(lambda m: dispatched_meters.get(m, "No Team Sent"))

# Fast dictionary lookup for format_func
meter_meta_lookup = {
    str(r["meter_id"]): f"[{r['meter_status']}{' - Normalized' if r['is_normalized'] else ''}] {r['meter_id']} ({r['city']}, {r['discom']})"
    for _, r in meter_summary.iterrows()
}

# Grid statistics & Estimated Monetary Revenue Loss in INR (₹)
total_meters_count = len(meter_summary)
theft_meters_count = (meter_summary["meter_status"] == "Theft Detected").sum()
suspicious_count   = (meter_summary["meter_status"] == "Suspicious").sum()
normal_count       = (meter_summary["meter_status"] == "Normal").sum()

# Penalties recovered
total_penalties_recovered = sum(info.get("fine_collected", 0) for info in st.session_state.normalized_meters.values())

total_kwh_billed = df_scored["mean"].sum() * 24.0
theft_windows = df_scored[df_scored["risk_level"] != "Normal"]
estimated_stolen_kwh = theft_windows["mean"].sum() * 24.0 * 0.85
estimated_revenue_loss_inr = (estimated_stolen_kwh * TARIFF_RATE_INR) / 100000.0  # ₹ Lakhs


# ─────────────────────────────────────────────────────────────────────────────
# 9. STANDARDIZED KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-box">
        <p class="metric-val c-slate">{total_meters_count}</p>
        <p class="metric-lbl">Total Meters Monitored</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-box">
        <p class="metric-val c-green">{normal_count}</p>
        <p class="metric-lbl">Normal Meters ({len(st.session_state.normalized_meters)} Rectified)</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-box">
        <p class="metric-val c-red">{theft_meters_count}</p>
        <p class="metric-lbl">Theft Detected (Action Needed)</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-box">
        <p class="metric-val c-amber">{suspicious_count}</p>
        <p class="metric-lbl">Under Watch (Suspicious)</p>
    </div>
    """, unsafe_allow_html=True)

with c5:
    if total_penalties_recovered > 0:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val c-emerald">₹ {total_penalties_recovered/100000.0:.2f} L</p>
            <p class="metric-lbl">Penalties Recovered</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val c-gold">₹ {estimated_revenue_loss_inr:.2f} L</p>
            <p class="metric-lbl">Estimated Stolen Revenue</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN OPERATION TABS WITH STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
tab_map, tab_dispatch, tab_detail = st.tabs([
    "🗺️ Andhra & Telangana Map & Discom Losses",
    "🚨 Theft Alerts & Squad Normalization",
    "🔍 Detailed Meter Check & Proof",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: ANDHRA & TELANGANA MAP & POWER LOSS
# ═════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("### 🗺️ Andhra Pradesh & Telangana Smart Meter Network")
    st.write(f"Real-time geographic distribution across **30 distribution feeders** spanning **Telangana** (TSSPDCL / TSNPDCL) and **Andhra Pradesh** (APEPDCL / APCPDCL / APSPDCL). Normalized meters appear as verified green nodes.")

    map_view_mode = st.radio(
        "Map Display View:",
        ["🌐 Show All Meters (Full Grid Network - 60 Meters)", "🚨 Show Active Theft & Suspicious Alerts Only"],
        key="map_view_mode",
        horizontal=True
    )

    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])

    with col_f1:
        state_opts = ["All States (AP & TS)", "Telangana", "Andhra Pradesh"]
        state_filter = st.selectbox("Filter by State:", state_opts, key="state_filter")

    with col_f2:
        if state_filter == "Telangana":
            discom_opts = ["All Telangana Discoms", "TSSPDCL", "TSNPDCL"]
        elif state_filter == "Andhra Pradesh":
            discom_opts = ["All AP Discoms", "APEPDCL", "APCPDCL", "APSPDCL"]
        else:
            discom_opts = ["All Discoms", "TSSPDCL", "TSNPDCL", "APEPDCL", "APCPDCL", "APSPDCL"]
        selected_discom = st.selectbox("Filter by Discom:", discom_opts, key="discom_filter")

    with col_f3:
        status_filter = st.multiselect(
            "Filter by Meter Status:",
            ["Theft Detected", "Suspicious", "Normal"],
            default=["Theft Detected", "Suspicious", "Normal"],
            key="status_filter"
        )

    # Filter map data instantaneously from in-memory cache
    map_data = meter_summary.copy()
    if "Alerts Only" in map_view_mode:
        map_data = map_data[map_data["meter_status"].isin(["Theft Detected", "Suspicious"])]

    if state_filter == "Telangana":
        map_data = map_data[map_data["state"] == "Telangana"]
    elif state_filter == "Andhra Pradesh":
        map_data = map_data[map_data["state"] == "Andhra Pradesh"]

    if selected_discom not in ["All Discoms", "All Telangana Discoms", "All AP Discoms"]:
        map_data = map_data[map_data["discom"] == selected_discom]

    if status_filter:
        map_data = map_data[map_data["meter_status"].isin(status_filter)]

    # Dynamic camera positioning
    if state_filter == "Telangana":
        c_lat, c_lon, c_zoom = 17.65, 78.95, 7.1
    elif state_filter == "Andhra Pradesh":
        c_lat, c_lon, c_zoom = 16.30, 81.10, 6.7
    else:
        c_lat, c_lon, c_zoom = 16.85, 80.15, 6.4

    if len(map_data) > 0:
        color_map = {
            "Theft Detected": "#ef4444",
            "Suspicious": "#f59e0b",
            "Normal": "#10b981",
        }
        
        fig_map = px.scatter_mapbox(
            map_data,
            lat="lat",
            lon="lon",
            color="meter_status",
            color_discrete_map=color_map,
            size=[22 if s == "Theft Detected" else 16 if s == "Suspicious" else 12 for s in map_data["meter_status"]],
            hover_name="meter_id",
            hover_data={
                "meter_status": True,
                "is_normalized": True,
                "city": True,
                "state": True,
                "discom": True,
                "tariff": True,
                "dominant_theft": True,
                "dispatch_status": True,
                "lat": False,
                "lon": False,
            },
            zoom=c_zoom,
            center={"lat": c_lat, "lon": c_lon},
            mapbox_style="open-street-map",
            height=500,
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(15, 23, 42, 0.88)",
                font=dict(color="#f8fafc")
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No smart meters match the selected filters.")

    # Feeder Power Loss Table
    st.markdown("#### ⚡ Power & Revenue Loss by Discom and Area")
    st.write("Calculates total supplied vs billed electricity and estimated financial revenue loss at ₹7.20 / kWh across all 30 distribution zones.")

    feeder_stats = []
    for fdr, cfg in FEEDER_CONFIG.items():
        fdr_meters = meter_summary[meter_summary["feeder"] == fdr]
        billed_mwh = (fdr_meters["avg_kwh"].sum() * 24.0 * 365) / 1000.0 if not fdr_meters.empty else 0.0
        n_theft = (fdr_meters["meter_status"] == "Theft Detected").sum()
        
        supplied_mwh = (billed_mwh * (1.28 if n_theft > 0 else 1.05)) if billed_mwh > 0 else 0.0
        loss_mwh = max(0.0, supplied_mwh - billed_mwh)
        loss_pct = (loss_mwh / supplied_mwh * 100.0) if supplied_mwh > 0 else 0.0
        loss_inr = (loss_mwh * 1000.0 * TARIFF_RATE_INR) / 100000.0 if supplied_mwh > 0 else 0.0
        
        feeder_stats.append({
            "Feeder / Substation": fdr,
            "Discom": cfg["discom"],
            "State": cfg["state"],
            "Connected Meters": len(fdr_meters),
            "Theft Meters": n_theft,
            "Supplied (MWh)": f"{supplied_mwh:.1f}",
            "Billed (MWh)": f"{billed_mwh:.1f}",
            "Lost Power (%)": f"{loss_pct:.1f}%",
            "Lost Revenue (₹)": f"₹ {loss_inr:.2f} Lakhs",
            "Loss Severity": "🔴 HIGH LOSS" if loss_pct > 20 else "🟡 MEDIUM" if loss_pct > 10 else "🟢 NORMAL"
        })

    st.dataframe(pd.DataFrame(feeder_stats), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: THEFT ALERTS, TEAM DISPATCH & RESOLUTION NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_dispatch:
    st.markdown("### 🚨 Anti-Power Theft Dispatch & Field Normalization")
    st.write("Assign APTS vigilance squads to investigate theft, and once verified and rectified on-site, mark meters as **Normalized** to restore them to compliance.")

    theft_meters = meter_summary[meter_summary["meter_status"].isin(["Theft Detected", "Suspicious"])].copy()
    theft_meters = theft_meters.sort_values(by=["max_confidence", "theft_days"], ascending=False)

    col_q1, col_q2 = st.columns([5, 4], gap="large")

    with col_q1:
        meter_list = theft_meters["meter_id"].tolist()
        if not meter_list:
            st.success("🎉 All smart meters are currently in Normal status! No active theft detected.")
            selected_meter_alert = None
            m_info = None
        else:
            if st.session_state.get("selected_meter_alert") not in meter_list:
                st.session_state.selected_meter_alert = meter_list[0]

            curr_sel_idx = meter_list.index(st.session_state.selected_meter_alert)

            selected_meter_alert = st.selectbox(
                "Select Flagged Meter to Review / Dispatch:",
                meter_list,
                index=curr_sel_idx,
                format_func=lambda m: meter_meta_lookup.get(str(m), str(m))
            )
            st.session_state.selected_meter_alert = selected_meter_alert

            # Alerts Table
            preview_df = theft_meters[[
                "meter_id", "meter_status", "dominant_theft", 
                "city", "state", "discom", "tariff", "dispatch_status"
            ]].copy()
            preview_df.columns = ["Meter ID", "Status", "Theft Pattern", "City", "State", "Discom", "Tariff Category", "Inspection Team"]
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with col_q2:
        if selected_meter_alert:
            matched_alert = meter_summary[meter_summary["meter_id"] == selected_meter_alert]
            if not matched_alert.empty:
                m_info = matched_alert.iloc[0]
            else:
                m_info = theft_meters.iloc[0] if not theft_meters.empty else meter_summary.iloc[0]
                selected_meter_alert = m_info["meter_id"]
                st.session_state.selected_meter_alert = selected_meter_alert

            existing_ticket = next((t for t in st.session_state.dispatch_log if t.get("Meter ID") == m_info["meter_id"]), None)

            st.markdown(f"""
            <div class="dispatch-panel">
                <div class="dp-header">⚡ Field Work Order: Meter {m_info['meter_id']}</div>
                <p><b>Jurisdiction Discom:</b> {m_info['discom']} &middot; {m_info['state']}</p>
                <p><b>Area & City:</b> {m_info['area']} ({m_info['city']})</p>
                <p><b>Tariff Category:</b> {m_info['tariff']}</p>
                <p><b>Theft Method Detected:</b> {m_info['dominant_theft']}</p>
                <p><b>Designated Police Station:</b> <code style="color:#38bdf8;">{m_info['police_stn']}</code></p>
                <p><b>Current Status:</b> <code>{m_info['dispatch_status']}</code></p>
            </div>
            """, unsafe_allow_html=True)

            pdf_bytes = generate_inspection_pdf(m_info.to_dict(), existing_ticket)
            
            col_pdf1, col_pdf2 = st.columns([1, 1])
            with col_pdf1:
                st.download_button(
                    label="📄 Download Official APTS Notice (PDF)",
                    data=pdf_bytes,
                    file_name=f"APTS_Notice_{m_info['meter_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Official inspection notice under Section 135 & 138 of the Electricity Act."
                )
            with col_pdf2:
                if st.button("🔍 Inspect in Tab 3", use_container_width=True):
                    st.session_state.selected_detail_meter = m_info["meter_id"]
                    st.toast(f"Switched inspection target to Meter {m_info['meter_id']}", icon="🔍")
                    st.rerun()

            with st.form("send_crew_form"):
                st.write("**Dispatch Authorized Vigilance / Inspection Squad:**")
                crew_assigned = st.selectbox(
                    "Assign Inspection Squad",
                    [
                        f"{m_info['discom']} Anti-Power Theft Squad (APTS) - Team 1",
                        f"{m_info['discom']} Rapid Vigilance Squad - Central",
                        "Divisional Assistant Engineer (DAE) Technical Squad",
                        "Special Task Force Night Patrol Unit"
                    ]
                )
                urgency = st.selectbox(
                    "Action Priority",
                    ["Emergency - Disconnect / Inspect Immediately", "High - Inspect Within 24 Hours", "Routine - Scheduled Review"]
                )
                instructions = st.text_area(
                    "Special Instructions for the Inspection Squad",
                    value=f"Verify CT/PT seals, neutral bypass tap, and phase reversing. Suspected method: {m_info['dominant_theft']}. Contact local APTS police station if tampering confirmed."
                )
                
                submit_team = st.form_submit_button("🚀 Dispatch Inspection Squad Now", use_container_width=True)

                if submit_team:
                    ticket_id = f"WO-{datetime.datetime.now().strftime('%y%m%d%H%M')}-{str(m_info['meter_id'])[-4:]}"
                    new_ticket = {
                        "Work Order ID": ticket_id,
                        "Meter ID": m_info["meter_id"],
                        "Discom": m_info["discom"],
                        "City": m_info["city"],
                        "State": m_info["state"],
                        "Assigned Squad": crew_assigned,
                        "Vigilance Police Station": m_info["police_stn"],
                        "Urgency": urgency,
                        "Instructions": instructions,
                        "Status": "Team Dispatched",
                        "Dispatched At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    st.session_state.dispatch_log = [t for t in st.session_state.dispatch_log if t.get("Meter ID") != m_info["meter_id"]]
                    st.session_state.dispatch_log.append(new_ticket)
                    
                    st.toast(f"🚀 Squad {crew_assigned} dispatched to Meter {m_info['meter_id']}!", icon="⚡")
                    st.toast(f"📋 Notice issued to {m_info['police_stn']}", icon="👮")
                    st.success(f"✅ Squad {crew_assigned} dispatched to Meter {m_info['meter_id']} (Order: {ticket_id})!")
                    st.rerun()

    # ── ON-SITE FIELD VERIFICATION & RESOLUTION NORMALIZATION WORKFLOW ──
    st.markdown("---")
    st.markdown("#### 🛠️ Field Squad Normalization & Restoration Hub")
    st.write("When an APTS squad arrives on-site and removes the bypass tap or replaces the tampered seal, submit the resolution below. The meter will immediately transition to **Normal** across all maps, KPI cards, and reports.")

    # Find tickets that are dispatched but not yet normalized
    dispatched_tickets = [t for t in st.session_state.dispatch_log if t.get("Status") == "Team Dispatched" and str(t.get("Meter ID")) not in st.session_state.normalized_meters]

    if dispatched_tickets:
        norm_col1, norm_col2 = st.columns([1, 1], gap="large")
        with norm_col1:
            ticket_options = [f"{t['Meter ID']} - {t['Assigned Squad']} (WO: {t['Work Order ID']})" for t in dispatched_tickets]
            selected_ticket_str = st.selectbox("Select Dispatched Team to Log Restoration Findings:", ticket_options)
            sel_ticket_m_id = selected_ticket_str.split(" - ")[0]
            active_ticket = next(t for t in dispatched_tickets if t["Meter ID"] == sel_ticket_m_id)

            st.info(f"📍 **Target Meter:** {active_ticket['Meter ID']} | **Discom:** {active_ticket['Discom']} | **City:** {active_ticket['City']}")

        with norm_col2:
            with st.form("normalize_meter_form"):
                st.write(f"**Log Tamper Removal & Normalization for Meter {active_ticket['Meter ID']}:**")
                action_executed = st.selectbox(
                    "Field Action Taken on Site:",
                    [
                        "Direct neutral bypass tap removed & digital tamper-evident seal applied",
                        "Shunted CT/PT unit removed and recalibrated to 100% metering accuracy",
                        "Illegal phase reversal tap disconnected & secure seal box installed",
                        "Tampered meter seized as evidence; replaced with new tamper-proof smart meter",
                        "False resistor load clamp removed & impounded"
                    ]
                )
                seal_number = st.text_input("New Security Seal ID Affixed:", value=f"APTS-SEAL-{active_ticket['Meter ID'][-4:]}-VERIFIED")
                fine_recovered = st.number_input("Compounding Penalty Recovered (₹) under Section 152:", min_value=0, max_value=500000, value=40000, step=5000)
                officer_name = st.text_input("Inspecting Vigilance Officer / DAE:", value="Er. K. V. Rao (DAE Vigilance)")
                
                submit_resolution = st.form_submit_button("✅ Submit Resolution & Normalize Meter Now", use_container_width=True)

                if submit_resolution:
                    # Update normalized meters state
                    st.session_state.normalized_meters[active_ticket["Meter ID"]] = {
                        "resolved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "action_taken": action_executed,
                        "new_seal_no": seal_number,
                        "fine_collected": fine_recovered,
                        "squad_name": active_ticket["Assigned Squad"],
                        "officer_name": officer_name
                    }
                    # Update work order in dispatch log
                    for t in st.session_state.dispatch_log:
                        if t.get("Meter ID") == active_ticket["Meter ID"]:
                            t["Status"] = "Resolved & Normalized"
                            t["Resolved At"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            t["Seal No"] = seal_number
                            t["Penalty Collected"] = f"₹ {fine_recovered:,.2f}"

                    st.toast(f"🎉 Meter {active_ticket['Meter ID']} successfully normalized and restored to Normal!", icon="✅")
                    st.success(f"🎉 Meter {active_ticket['Meter ID']} normalized! Status is now NORMAL across the dashboard.")
                    st.rerun()
    else:
        st.info("ℹ️ No active teams currently awaiting on-site resolution. Dispatch a squad above to enable the normalization workflow.")

    # Normalized Meters History & Restoration Certificates
    if st.session_state.normalized_meters:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📜 Normalized & Restored Meters History (Compliance Registry)")
        
        for m_id, res_info in st.session_state.normalized_meters.items():
            m_row = meter_summary_base[meter_summary_base["meter_id"] == m_id]
            m_dict = m_row.iloc[0].to_dict() if not m_row.empty else {"meter_id": m_id, "discom": "DISCOM", "feeder": "FEEDER", "city": "CITY", "state": "STATE", "tariff": "Commercial"}
            ticket_dict = next((t for t in st.session_state.dispatch_log if t.get("Meter ID") == m_id), None)
            
            cert_pdf_bytes = generate_restoration_certificate_pdf(m_dict, ticket_dict, res_info)

            col_rc1, col_rc2, col_rc3 = st.columns([3, 2, 1])
            with col_rc1:
                st.markdown(f"""
                <div class="normalized-card">
                    <b style="color:#10b981; font-size:1rem;">✅ Meter {m_id} &mdash; Normalized & Restored</b><br/>
                    <span style="font-size:0.85rem; color:#cbd5e1;"><b>Action:</b> {res_info['action_taken']}</span><br/>
                    <span style="font-size:0.85rem; color:#cbd5e1;"><b>Seal No:</b> <code style="color:#38bdf8;">{res_info['new_seal_no']}</code> &middot; <b>Officer:</b> {res_info['officer_name']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_rc2:
                st.markdown(f"""
                <div style="padding-top:0.4rem;">
                    <p style="margin:0; font-size:0.85rem;"><b>Penalty Recovered:</b> <span style="color:#34d399; font-weight:700;">₹ {res_info['fine_collected']:,.2f}</span></p>
                    <p style="margin:0; font-size:0.85rem; color:#94a3b8;">Restored at: {res_info['resolved_at']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_rc3:
                st.download_button(
                    label="📄 Certificate (PDF)",
                    data=cert_pdf_bytes,
                    file_name=f"APTS_Restoration_Certificate_{m_id}.pdf",
                    mime="application/pdf",
                    key=f"cert_dl_{m_id}",
                    use_container_width=True
                )
                if st.button("🔄 Reopen", key=f"reopen_{m_id}", help="Re-open investigation if tamper persists"):
                    del st.session_state.normalized_meters[m_id]
                    for t in st.session_state.dispatch_log:
                        if t.get("Meter ID") == m_id:
                            t["Status"] = "Team Dispatched"
                    st.toast(f"Reopened investigation for Meter {m_id}", icon="⚠️")
                    st.rerun()

    # Dispatched Teams Log Table
    if st.session_state.dispatch_log:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Dispatched Squads & Work Orders Log")
        dispatch_df = pd.DataFrame(st.session_state.dispatch_log)
        st.dataframe(dispatch_df, use_container_width=True, hide_index=True)

        csv_data = dispatch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Work Orders Log (CSV)",
            csv_data,
            "discom_inspection_orders.csv",
            "text/csv",
            on_click=lambda: st.toast("📥 Work orders audit log CSV exported successfully!", icon="📋")
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DETAILED METER CHECK & PROOF
# ═════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("### 🔍 Detailed Meter Examination & AI Proof")
    st.write("Inspect any smart meter to see its complete 365-day energy consumption telemetry, machine learning diagnostic breakdown, and behavioral anomalies.")

    # Status Category Filter
    meter_cat_filter = st.radio(
        "Filter Meter Selector by Status:",
        ["All Meters (60)", "🟢 Normal & Healthy Meters", "🔴 Theft Detected Meters", "🟡 Suspicious Meters"],
        key="meter_cat_filter",
        horizontal=True
    )

    if "Normal" in meter_cat_filter:
        selectable_meters = sorted(meter_summary[meter_summary["meter_status"] == "Normal"]["meter_id"].tolist())
    elif "Theft Detected" in meter_cat_filter:
        selectable_meters = sorted(meter_summary[meter_summary["meter_status"] == "Theft Detected"]["meter_id"].tolist())
    elif "Suspicious" in meter_cat_filter:
        selectable_meters = sorted(meter_summary[meter_summary["meter_status"] == "Suspicious"]["meter_id"].tolist())
    else:
        selectable_meters = sorted(meter_summary["meter_id"].tolist())

    if not selectable_meters:
        selectable_meters = sorted(meter_summary["meter_id"].tolist())

    if st.session_state.get("selected_detail_meter") not in selectable_meters:
        st.session_state.selected_detail_meter = selectable_meters[0]

    detail_idx = selectable_meters.index(st.session_state.selected_detail_meter)

    meter_choice = st.selectbox(
        "Select a Meter to Inspect:",
        selectable_meters,
        index=detail_idx,
        format_func=lambda m: meter_meta_lookup.get(str(m), str(m))
    )
    st.session_state.selected_detail_meter = meter_choice

    # Safe row lookup
    matched_detail = meter_summary[meter_summary["meter_id"] == meter_choice]
    if not matched_detail.empty:
        m_detail = matched_detail.iloc[0]
    else:
        m_detail = meter_summary.iloc[0]
        meter_choice = m_detail["meter_id"]
        st.session_state.selected_detail_meter = meter_choice

    meter_windows = df_scored[df_scored["meter_id"] == meter_choice].copy()
    if "window_id" in meter_windows.columns:
        meter_windows = meter_windows.sort_values("window_id")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val c-cyan">{meter_choice}</p>
            <p class="metric-lbl">Meter ID &middot; {m_detail['tariff']}</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        status_color = 'c-emerald' if m_detail['is_normalized'] else 'c-red' if m_detail['meter_status']=='Theft Detected' else 'c-amber' if m_detail['meter_status']=='Suspicious' else 'c-green'
        disp_status = f"{m_detail['meter_status']}{' (Restored)' if m_detail['is_normalized'] else ''}"
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val {status_color}">{disp_status}</p>
            <p class="metric-lbl">Classification Status</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val c-slate">{m_detail['city']}, {m_detail['state']}</p>
            <p class="metric-lbl">Discom: {m_detail['discom']}</p>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown(f"""
        <div class="metric-box">
            <p class="metric-val c-cyan">{m_detail['avg_kwh']:.2f} kWh/day</p>
            <p class="metric-lbl">Avg Consumption</p>
        </div>
        """, unsafe_allow_html=True)

    # Contextual Explanation Banner
    if m_detail["is_normalized"]:
        norm_data = st.session_state.normalized_meters.get(str(meter_choice), {})
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 12px; padding: 1.25rem 1.4rem; margin-top: 1.2rem; margin-bottom: 1.2rem;">
            <div style="display:flex; align-items:center; gap: 0.6rem;">
                <span style="font-size: 1.3rem;">🎉</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #34d399;">Meter Successfully Normalized & Restored by APTS Squad</span>
            </div>
            <p style="color: #cbd5e1; margin: 0.4rem 0 0 0; font-size: 0.92rem; line-height: 1.5;">
                Tampering has been officially rectified on-site by <b>{norm_data.get('squad_name', 'APTS Squad')}</b>. 
                Action executed: <i>{norm_data.get('action_taken', 'Tamper removed')}</i>. 
                New security seal ID: <code style="color:#38bdf8;">{norm_data.get('new_seal_no', 'N/A')}</code> &middot; 
                Compounding penalty recovered: <b>₹ {norm_data.get('fine_collected', 0):,.2f}</b>. 
                Status is restored to <b>NORMAL</b> in utility SCADA and billing.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif m_detail["meter_status"] == "Normal":
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1.5px solid #10b981; border-radius: 12px; padding: 1.25rem 1.4rem; margin-top: 1.2rem; margin-bottom: 1.2rem;">
            <div style="display:flex; align-items:center; gap: 0.6rem;">
                <span style="font-size: 1.25rem;">✅</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #34d399;">Authentic Normal Consumer — No Power Theft Detected</span>
            </div>
            <p style="color: #cbd5e1; margin: 0.4rem 0 0 0; font-size: 0.92rem; line-height: 1.5;">
                This smart meter exhibits clean, standard electricity consumption ({m_detail['normal_days']} of 365 days fully normal). There are no indicators of meter tampering, CT/PT disconnection, neutral phase bypass, or illegal load suppression. <b>Emerald green</b> markers in the telemetry chart confirm healthy continuous readings.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif m_detail["meter_status"] == "Theft Detected":
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1.5px solid #ef4444; border-radius: 12px; padding: 1.25rem 1.4rem; margin-top: 1.2rem; margin-bottom: 1.2rem;">
            <div style="display:flex; align-items:center; gap: 0.6rem;">
                <span style="font-size: 1.25rem;">🚨</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #f87171;">Confirmed Electricity Theft — {m_detail['dominant_theft']}</span>
            </div>
            <p style="color: #cbd5e1; margin: 0.4rem 0 0 0; font-size: 0.92rem; line-height: 1.5;">
                Severe electricity theft detected across <b>{m_detail['theft_days']} days</b>. The machine learning pipeline identified unmistakable tampering signatures. Discom inspection squad dispatch is recommended via Tab 2.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.12); border: 1.5px solid #f59e0b; border-radius: 12px; padding: 1.25rem 1.4rem; margin-top: 1.2rem; margin-bottom: 1.2rem;">
            <div style="display:flex; align-items:center; gap: 0.6rem;">
                <span style="font-size: 1.25rem;">⚠️</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #fbbf24;">Under Active Watch — Irregular Consumption Pattern</span>
            </div>
            <p style="color: #cbd5e1; margin: 0.4rem 0 0 0; font-size: 0.92rem; line-height: 1.5;">
                Meter shows intermittent fluctuations ({m_detail['suspicious_days']} suspicious days). Automated daily AMI monitoring enabled.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Download Forensic Audit or Clearance PDF Report
    existing_ticket_for_detail = next((t for t in st.session_state.dispatch_log if t.get("Meter ID") == m_detail["meter_id"]), None)
    
    if m_detail["is_normalized"]:
        norm_detail_data = st.session_state.normalized_meters.get(str(meter_choice), {})
        clearance_pdf = generate_restoration_certificate_pdf(m_detail.to_dict(), existing_ticket_for_detail, norm_detail_data)
        st.download_button(
            label=f"📄 Download Certificate of Meter Restoration & Clearance (PDF)",
            data=clearance_pdf,
            file_name=f"APTS_Restoration_Certificate_{meter_choice}.pdf",
            mime="application/pdf",
            on_click=lambda: st.toast(f"📄 Restoration Certificate downloaded for Meter {meter_choice}!", icon="🎉")
        )
    else:
        detail_pdf = generate_inspection_pdf(m_detail.to_dict(), existing_ticket_for_detail)
        st.download_button(
            label=f"📄 Download Forensic Audit Report for Meter {meter_choice} (PDF)",
            data=detail_pdf,
            file_name=f"Forensic_Report_{meter_choice}.pdf",
            mime="application/pdf",
            on_click=lambda: st.toast(f"📄 Forensic audit report downloaded for Meter {meter_choice}!", icon="✅")
        )

    # Daily Usage Graph
    st.markdown("#### 📈 365-Day Energy Consumption Telemetry")
    st.write("Continuous daily reading history. **Emerald Green** dots indicate normal authentic usage; **Red** dots represent detected theft days.")
    
    color_map_series = {"Normal": "#10b981", "Suspicious": "#f59e0b", "Theft Detected": "#ef4444"}
    x_axis = meter_windows["window_id"].tolist() if "window_id" in meter_windows.columns else list(range(len(meter_windows)))

    fig_series = go.Figure()

    fig_series.add_trace(go.Scatter(
        x=x_axis,
        y=meter_windows["mean"],
        mode="lines",
        line=dict(color="#334155", width=1.6),
        name="Telemetry Baseline Trend",
        hoverinfo="skip"
    ))

    for r_lvl in ["Normal", "Suspicious", "Theft Detected"]:
        subset = meter_windows[meter_windows["risk_level"] == r_lvl]
        if not subset.empty:
            sub_x = subset["window_id"].tolist() if "window_id" in subset.columns else [i for i, m in enumerate(meter_windows["risk_level"] == r_lvl) if m]
            fig_series.add_trace(go.Scatter(
                x=sub_x,
                y=subset["mean"],
                mode="markers",
                name=f"{r_lvl} Days ({len(subset)})",
                marker=dict(
                    color=color_map_series[r_lvl],
                    size=11 if r_lvl == "Theft Detected" else 8 if r_lvl == "Suspicious" else 5,
                    opacity=0.88
                ),
                hovertemplate="<b>%{text}</b><br>Day #%{x}<br>Recorded: %{y:.3f} kWh<extra></extra>",
                text=subset["tamper_type"]
            ))

    fig_series.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0f172a",
        height=370,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Day of the Year (1 to 365)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Daily Consumption (kWh)", gridcolor="#1e293b", color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#f8fafc"))
    )
    st.plotly_chart(fig_series, use_container_width=True)

    # Diagnostic Tables
    col_e1, col_e2 = st.columns([1, 1], gap="large")

    with col_e1:
        st.markdown("#### 🤖 Machine Learning Diagnostic Proof")
        rf_mean = meter_windows["fraud_prob"].mean()
        iso_rate = (meter_windows["iso_flag"] == 1).mean()
        lof_rate = (meter_windows["lof_flag"] == 1).mean()
        avg_conf = meter_windows["confidence"].mean()

        signal_df = pd.DataFrame({
            "Detection Model / Component": [
                "Supervised Ensemble (XGBoost + Random Forest)",
                "Outlier Boundary Check (Isolation Forest)",
                "Local Density Anomaly (Local Outlier Factor)",
                "Hybrid Confidence Score"
            ],
            "Score / Probability": [
                f"{rf_mean * 100:.1f}% Theft Probability",
                f"{iso_rate * 100:.1f}% Days Outlier",
                f"{lof_rate * 100:.1f}% Days Density Shift",
                f"{avg_conf:.1f} out of 5 Points"
            ],
            "Evaluation": [
                "Matches verified electricity theft profile" if rf_mean >= 0.4 else "Authentic consumer pattern (< 10% fraud probability)",
                "Severe deviation from baseline load" if iso_rate >= 0.1 else "Within expected grid variation (No abnormal spikes/drops)",
                "Unnatural sudden drop compared to nearby meters" if lof_rate >= 0.1 else "Consistent local density (Matches neighborhood pattern)",
                "Immediate Action Required" if avg_conf >= 3 else "Keep on Surveillance" if avg_conf >= 1 else "Normal Operation (Compliant)"
            ]
        })
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

    with col_e2:
        st.markdown("#### 🎯 Behavioral Red Flags Radar")
        st.write("Identifies the specific tampering traits exhibited by this consumer:")
        
        radar_categories = ["Zero Hours", "Low-Usage Streak", "Erratic Swings (CV)", "Peak Spike", "Day/Night Ratio"]
        
        z_norm = min(1.0, float(meter_windows["zero_pct"].mean() / 0.5))
        s_norm = min(1.0, float(meter_windows["max_low_streak"].mean() / 20.0))
        cv_norm = min(1.0, float(meter_windows["cv"].mean() / 4.0))
        pta_norm = min(1.0, float(meter_windows["peak_to_avg"].mean() / 8.0))
        dn_norm = min(1.0, float(meter_windows["day_night_ratio"].mean() / 3.0))

        radar_vals = [z_norm, s_norm, cv_norm, pta_norm, dn_norm]
        radar_vals.append(radar_vals[0])
        categories_closed = radar_categories + [radar_categories[0]]

        radar_color = '#10b981' if (m_detail['meter_status'] == 'Normal' or m_detail['is_normalized']) else '#f59e0b' if m_detail['meter_status'] == 'Suspicious' else '#ef4444'
        radar_fill = 'rgba(16, 185, 129, 0.35)' if (m_detail['meter_status'] == 'Normal' or m_detail['is_normalized']) else 'rgba(245, 158, 11, 0.35)' if m_detail['meter_status'] == 'Suspicious' else 'rgba(239, 68, 68, 0.35)'

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals,
            theta=categories_closed,
            fill='toself',
            fillcolor=radar_fill,
            line=dict(color=radar_color, width=2),
            name=meter_choice
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="#1e293b", color="#94a3b8"),
                angularaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                bgcolor="#0f172a"
            ),
            paper_bgcolor="#0b1120",
            showlegend=False,
            height=300,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
