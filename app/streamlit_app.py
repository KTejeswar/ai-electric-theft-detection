import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — clean, utilitarian, no AI-slop gradients
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.block-container {
    padding: 2.5rem 3.5rem;
    max-width: 1280px;
}
section[data-testid="stSidebar"] { display: none; }

/* ── PAGE HEADER ── */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #111827;
    margin: 0 0 .2rem;
    letter-spacing: -.3px;
}
.page-sub {
    color: #6b7280;
    font-size: .88rem;
    font-weight: 300;
    margin: 0;
}
.page-rule {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.4rem 0 1.6rem;
}

/* ── SUMMARY CARDS ── */
.cards-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    flex: 1;
}
.card-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    line-height: 1;
}
.card-label {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .9px;
    color: #9ca3af;
    margin-top: .35rem;
}
.c-red   { color: #dc2626; }
.c-amber { color: #d97706; }
.c-green { color: #16a34a; }
.c-slate { color: #1d4ed8; }

/* ── ALERT TABLE ── */
.tbl-wrap { overflow-x: auto; }
.atbl {
    width: 100%;
    border-collapse: collapse;
    font-size: .88rem;
}
.atbl thead tr {
    background: #f9fafb;
    border-bottom: 2px solid #e5e7eb;
}
.atbl th {
    padding: .65rem 1rem;
    text-align: left;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #6b7280;
    font-weight: 500;
}
.atbl td {
    padding: .8rem 1rem;
    border-bottom: 1px solid #f3f4f6;
    vertical-align: middle;
}
.atbl tbody tr:hover { background: #fafafa; }
.atbl .meter-id {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: .85rem;
    color: #111827;
}
.atbl .finding-cell { color: #4b5563; line-height: 1.4; }
.atbl .windows-cell {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .8rem;
    color: #9ca3af;
}

/* ── STATUS BADGES ── */
.bh {
    display: inline-block;
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fca5a5;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
    white-space: nowrap;
}
.bs {
    display: inline-block;
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fcd34d;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
    white-space: nowrap;
}
.bn {
    display: inline-block;
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #86efac;
    padding: 2px 9px;
    border-radius: 4px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
    white-space: nowrap;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9ca3af;
    margin-bottom: .6rem;
}

/* ── FINDING BULLETS ── */
.bul-r {
    background: #fef2f2;
    border-left: 3px solid #dc2626;
    padding: .55rem 1rem;
    margin: .4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: .87rem;
    color: #374151;
    line-height: 1.5;
}
.bul-a {
    background: #fffbeb;
    border-left: 3px solid #d97706;
    padding: .55rem 1rem;
    margin: .4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: .87rem;
    color: #374151;
    line-height: 1.5;
}
.bul-g {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    padding: .55rem 1rem;
    margin: .4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: .87rem;
    color: #374151;
    line-height: 1.5;
}

/* ── METER DETAIL HEADER ── */
.meter-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    margin: 0 0 .25rem;
}
.meter-stats {
    font-size: .82rem;
    color: #6b7280;
    margin-bottom: 1.2rem;
}

/* ── RECOMMENDATION BOX ── */
.rec-box {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: .8rem 1.1rem;
    margin-top: 1rem;
    font-size: .85rem;
    color: #374151;
}
.rec-label {
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #9ca3af;
    margin-bottom: .3rem;
}

/* ── UPLOAD ZONE ── */
div[data-testid="stFileUploader"] {
    background: #fff;
    border: 1.5px dashed #d1d5db;
    border-radius: 8px;
    padding: .5rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6b7280;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "mean", "std", "min", "max", "median", "range",
    "load_factor", "peak_to_avg", "cv",
    "mean_abs_diff", "std_diff",
    "day_mean", "night_mean", "day_night_ratio",
    "zero_pct", "low_pct", "max_low_streak",
]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "notebooks", "models")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf  = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        iso = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        lof = joblib.load(os.path.join(MODELS_DIR, "lof_model.pkl"))
        sc  = joblib.load(os.path.join(MODELS_DIR, "scaler_anomaly.pkl"))
        thr = joblib.load(os.path.join(MODELS_DIR, "thresholds.pkl"))
        return rf, xgb, iso, lof, sc, thr.get("threshold", 0.2), None
    except Exception as e:
        return None, None, None, None, None, 0.2, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(df_raw, rf, xgb, iso, lof, sc, threshold):
    df = df_raw.copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    X = df[FEATURE_COLS]

    rf_p  = rf.predict_proba(X)[:, 1]
    xgb_p = xgb.predict_proba(X)[:, 1]
    ens_p = (rf_p + xgb_p) / 2
    sup   = (ens_p >= threshold).astype(int)

    Xs    = sc.transform(X)
    iso_f = (iso.predict(Xs) == -1).astype(int)
    lof_f = (lof.predict(Xs) == -1).astype(int)
    anom  = ((iso_f + lof_f) >= 1).astype(int)

    labels = []
    for s, a in zip(sup, anom):
        if   s == 1 and a == 1: labels.append("HIGH RISK")
        elif s == 1 or  a == 1: labels.append("SUSPICIOUS")
        else:                   labels.append("NORMAL")

    df["fraud_prob"] = ens_p
    df["iso_flag"]   = iso_f
    df["lof_flag"]   = lof_f
    df["anom_flag"]  = anom
    df["risk_level"] = labels
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — status badge HTML
# ─────────────────────────────────────────────────────────────────────────────
def badge(status):
    if status == "HIGH RISK":
        return '<span class="bh">HIGH RISK</span>'
    if status == "SUSPICIOUS":
        return '<span class="bs">SUSPICIOUS</span>'
    return '<span class="bn">NORMAL</span>'


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — one-line table finding
# ─────────────────────────────────────────────────────────────────────────────
def short_finding(meter_data, status):
    if status == "NORMAL":
        return "No suspicious patterns found."

    z   = meter_data["zero_pct"].mean()
    pta = meter_data["peak_to_avg"].mean()
    st_ = meter_data["max_low_streak"].mean()
    n_h = (meter_data["risk_level"] == "HIGH RISK").sum()

    if z > 0.6:
        return f"Zero consumption in {z:.0%} of monitored hours — possible meter bypass."
    if z > 0.35:
        return f"Frequent zero-reading periods ({z:.0%} of hours) — unusual for active premises."
    if n_h > 0:
        return f"{n_h} window(s) flagged by both detection models simultaneously."
    if st_ > 12:
        return f"Consumption flat at near-zero for up to {st_:.0f} consecutive hours."
    if pta > 6:
        return f"Usage spikes to {pta:.1f}× the average — erratic, inconsistent pattern."
    return "Irregular consumption pattern — review recommended."


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — detailed findings list for drill-down panel
# ─────────────────────────────────────────────────────────────────────────────
def detail_findings(meter_data, status):
    if status == "NORMAL":
        return [("g", "Consumption looks normal across all monitored periods. No action needed.")]

    out = []
    z   = meter_data["zero_pct"].mean()
    lp  = meter_data["low_pct"].mean()
    st_ = meter_data["max_low_streak"].mean()
    cv  = meter_data["cv"].mean()
    pta = meter_data["peak_to_avg"].mean()
    lf  = meter_data["load_factor"].mean()
    n_h = (meter_data["risk_level"] == "HIGH RISK").sum()
    n_s = (meter_data["risk_level"] == "SUSPICIOUS").sum()
    n_t = len(meter_data)

    tone = "r" if status == "HIGH RISK" else "a"

    if z > 0.6:
        out.append((tone,
            f"Meter records zero consumption in {z:.0%} of hours. "
            "This is consistent with a meter bypass or physical tampering."))
    elif z > 0.3:
        out.append((tone,
            f"Zero-reading hours account for {z:.0%} of the monitoring period — "
            "significantly above normal."))

    if lp > 0.5 and z < 0.4:
        out.append((tone,
            f"Very low (but non-zero) consumption in {lp:.0%} of hours. "
            "This pattern can indicate load suppression — usage being hidden, not eliminated."))

    if st_ > 12:
        out.append((tone,
            f"The longest uninterrupted stretch of near-zero consumption lasts "
            f"{st_:.0f} hours on average. Normal meters rarely go this long without measurable use."))
    elif st_ > 6:
        out.append((tone,
            f"Streaks of {st_:.0f} consecutive low-usage hours detected — "
            "unusual for an occupied premises."))

    if pta > 8:
        out.append((tone,
            f"Peak usage reaches {pta:.1f}× the window average. "
            "Extreme spikes often accompany deliberate usage hiding between readings."))
    elif pta > 5:
        out.append((tone,
            f"Periodic spikes reaching {pta:.1f}× the average suggest "
            "usage is being concentrated at specific times."))

    if cv > 3:
        out.append((tone,
            f"Hour-to-hour consumption is highly erratic (variability score: {cv:.1f}). "
            "Most residential meters show much more predictable patterns."))

    if lf < 0.12:
        out.append((tone,
            f"Load factor of {lf:.2f} — energy use is highly intermittent "
            "with very low consistency across the monitoring period."))

    if n_h > 0:
        out.append((tone,
            f"{n_h} of {n_t} monitoring windows were flagged simultaneously by both "
            "the supervised classifier and the anomaly detector — the strongest possible signal."))
    elif n_s > 0:
        out.append((tone,
            f"{n_s} of {n_t} windows flagged as suspicious — "
            "one detection model raised an alert but the other did not."))

    if not out:
        out.append((tone, "Anomalous patterns detected that deviate from expected consumption behaviour."))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="page-title">⚡ Electricity Theft Detection</p>
<p class="page-sub">Upload meter window data to scan for suspicious consumption patterns.</p>
<hr class="page-rule">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
rf, xgb, iso, lof, sc, threshold, err = load_models()
if err:
    st.error(f"Could not load detection models: {err}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload window_features.csv",
    type=["csv"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0;color:#9ca3af;">
        <div style="font-size:2rem;margin-bottom:.6rem;">📂</div>
        <div style="font-size:.9rem;font-weight:500;color:#6b7280;">
            Upload window_features.csv to begin</div>
        <div style="font-size:.8rem;margin-top:.3rem;">
            Output from the feature engineering notebook</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded)
missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

has_meter  = "meter_id"  in df_raw.columns
has_window = "window_id" in df_raw.columns
has_truth  = "theft"     in df_raw.columns


# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Scanning for suspicious patterns…"):
    result = run_pipeline(df_raw, rf, xgb, iso, lof, sc, threshold)


# ─────────────────────────────────────────────────────────────────────────────
# PER-METER SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
if has_meter:
    risk_map = {"HIGH RISK": 2, "SUSPICIOUS": 1, "NORMAL": 0}
    result["_rs"] = result["risk_level"].map(risk_map)

    ms = result.groupby("meter_id").agg(
        avg_fraud_prob   = ("fraud_prob",      "mean"),
        high_risk_count  = ("risk_level",      lambda x: (x == "HIGH RISK").sum()),
        suspicious_count = ("risk_level",      lambda x: (x == "SUSPICIOUS").sum()),
        normal_count     = ("risk_level",      lambda x: (x == "NORMAL").sum()),
        total_windows    = ("risk_level",      "count"),
        avg_rs           = ("_rs",             "mean"),
    ).reset_index()

    def meter_status(row):
        if row["high_risk_count"] > 0:  return "HIGH RISK"
        if row["suspicious_count"] > 0: return "SUSPICIOUS"
        return "NORMAL"

    ms["status"] = ms.apply(meter_status, axis=1)
    ms = ms.sort_values("avg_rs", ascending=False)

    ms["finding"] = ms.apply(
        lambda r: short_finding(result[result["meter_id"] == r["meter_id"]], r["status"]),
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STRIP
# ─────────────────────────────────────────────────────────────────────────────
n_total   = len(result)
n_high    = (result["risk_level"] == "HIGH RISK").sum()
n_susp    = (result["risk_level"] == "SUSPICIOUS").sum()
n_norm    = (result["risk_level"] == "NORMAL").sum()
n_meters  = ms["meter_id"].nunique() if has_meter else "—"
n_flagged = (ms["status"] != "NORMAL").sum() if has_meter else "—"

c0, c1, c2, c3, c4 = st.columns(5)

with c0:
    st.markdown(f"""<div class="card">
        <p class="card-num c-slate">{n_meters}</p>
        <p class="card-label">Meters scanned</p>
    </div>""", unsafe_allow_html=True)

with c1:
    st.markdown(f"""<div class="card">
        <p class="card-num c-slate">{n_total:,}</p>
        <p class="card-label">Windows checked</p>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="card">
        <p class="card-num c-red">{n_high:,}</p>
        <p class="card-label">High-risk windows</p>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="card">
        <p class="card-num c-amber">{n_susp:,}</p>
        <p class="card-label">Suspicious windows</p>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="card">
        <p class="card-num c-slate">{n_flagged}</p>
        <p class="card-label">Meters flagged</p>
    </div>""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALERT TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Meter Alert Board</p>', unsafe_allow_html=True)

if has_meter:
    rows = ""
    for _, row in ms.iterrows():
        flagged = row["high_risk_count"] + row["suspicious_count"]
        win_txt = f'{flagged} / {row["total_windows"]}'
        rows += f"""
        <tr>
            <td class="meter-id">{row["meter_id"]}</td>
            <td>{badge(row["status"])}</td>
            <td class="finding-cell">{row["finding"]}</td>
            <td class="windows-cell">{win_txt}</td>
        </tr>"""

    st.markdown(f"""
    <div class="tbl-wrap">
    <table class="atbl">
        <thead>
            <tr>
                <th>Meter</th>
                <th>Status</th>
                <th>Finding</th>
                <th>Flagged / Total Windows</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No meter_id column found — showing window-level results only.")


# ─────────────────────────────────────────────────────────────────────────────
# METER DRILL-DOWN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<p class="section-label">Investigate a meter</p>', unsafe_allow_html=True)

if not has_meter:
    st.info("No meter_id column — cannot drill down by meter.")
    st.stop()

# selectbox — high-risk meters float to top
sorted_meters = ms["meter_id"].tolist()
sel = st.selectbox(
    "Select a meter to investigate",
    sorted_meters,
    label_visibility="collapsed",
)

meter_row  = ms[ms["meter_id"] == sel].iloc[0]
meter_data = result[result["meter_id"] == sel].copy()
if has_window:
    meter_data = meter_data.sort_values("window_id")

status      = meter_row["status"]
n_hr        = meter_row["high_risk_count"]
n_su        = meter_row["suspicious_count"]
n_no        = meter_row["normal_count"]
n_win       = meter_row["total_windows"]
status_color = {"HIGH RISK": "#dc2626", "SUSPICIOUS": "#d97706", "NORMAL": "#16a34a"}[status]

# Layout — findings left, chart right
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown(f"""
    <p class="meter-name">Meter {sel} &nbsp; {badge(status)}</p>
    <p class="meter-stats">
        {n_hr} high-risk &nbsp;·&nbsp; {n_su} suspicious &nbsp;·&nbsp;
        {n_no} normal &nbsp;·&nbsp; {n_win} windows total
    </p>
    """, unsafe_allow_html=True)

    findings = detail_findings(meter_data, status)
    st.markdown("**What we found**")
    for tone, text in findings:
        cls = {"r": "bul-r", "a": "bul-a", "g": "bul-g"}[tone]
        st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

    # Recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    if status == "HIGH RISK":
        st.markdown("""
        <div class="rec-box">
            <p class="rec-label">Recommended action</p>
            Schedule a physical meter inspection at this location. 
            Both detection models agree on anomalous behaviour — this warrants priority review.
        </div>
        """, unsafe_allow_html=True)
    elif status == "SUSPICIOUS":
        st.markdown("""
        <div class="rec-box">
            <p class="rec-label">Recommended action</p>
            Add to the watchlist. Monitor consumption over the next billing cycle 
            before escalating. One model raised an alert — worth tracking.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rec-box">
            <p class="rec-label">Recommended action</p>
            No action required. Continue routine monitoring.
        </div>
        """, unsafe_allow_html=True)

with right:
    # ── Consumption chart ──
    color_map  = {"HIGH RISK": "#dc2626", "SUSPICIOUS": "#d97706", "NORMAL": "#6b7280"}
    x_vals     = meter_data["window_id"].tolist() if has_window else list(range(len(meter_data)))

    fig = go.Figure()

    # Grey baseline line
    fig.add_trace(go.Scatter(
        x=x_vals, y=meter_data["mean"],
        mode="lines",
        line=dict(color="#d1d5db", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Coloured markers per risk level
    for risk in ["NORMAL", "SUSPICIOUS", "HIGH RISK"]:
        mask = meter_data["risk_level"] == risk
        if not mask.any():
            continue
        m_data = meter_data[mask]
        m_x    = m_data["window_id"].tolist() if has_window else [
            i for i, m in enumerate(mask.values) if m]
        fig.add_trace(go.Scatter(
            x=m_x,
            y=m_data["mean"].values,
            mode="markers",
            name=risk,
            marker=dict(
                color=color_map[risk],
                size=12 if risk == "HIGH RISK" else 8 if risk == "SUSPICIOUS" else 6,
                opacity=0.9,
                line=dict(color="white", width=1) if risk == "HIGH RISK" else dict(width=0),
            ),
            hovertemplate=(
                f"<b>{risk}</b><br>"
                "Window: %{x}<br>"
                "Avg kWh: %{y:.5f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#374151", size=12),
        margin=dict(l=50, r=20, t=45, b=45),
        title=dict(
            text=f"Consumption — Meter {sel}",
            font=dict(size=13, family="IBM Plex Mono, monospace"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Window (24-hour periods)",
            gridcolor="#f3f4f6",
            linecolor="#e5e7eb",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Mean kWh",
            gridcolor="#f3f4f6",
            linecolor="#e5e7eb",
            tickfont=dict(size=10),
        ),
        height=370,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FLAGGED WINDOWS TABLE
# ─────────────────────────────────────────────────────────────────────────────
flagged_rows = meter_data[meter_data["risk_level"] != "NORMAL"]

if len(flagged_rows) > 0:
    st.markdown(
        f'<p class="section-label" style="margin-top:1.2rem;">'
        f'Flagged windows — {len(flagged_rows)} of {n_win}</p>',
        unsafe_allow_html=True,
    )

    show_cols = []
    if has_window:
        show_cols.append("window_id")
    show_cols += ["risk_level", "zero_pct", "low_pct", "max_low_streak",
                  "peak_to_avg", "cv", "fraud_prob"]
    show_cols  = [c for c in show_cols if c in flagged_rows.columns]

    disp = flagged_rows[show_cols].copy().reset_index(drop=True)

    # Human-readable column names
    rename_map = {
        "window_id":      "Window",
        "risk_level":     "Status",
        "zero_pct":       "Zero-reading hours",
        "low_pct":        "Low-usage hours",
        "max_low_streak": "Longest low streak",
        "peak_to_avg":    "Peak / Avg ratio",
        "cv":             "Usage variability",
        "fraud_prob":     "Model confidence",
    }
    disp.rename(columns={k: v for k, v in rename_map.items() if k in disp.columns}, inplace=True)

    # Format percentages
    for col in ["Zero-reading hours", "Low-usage hours", "Model confidence"]:
        if col in disp.columns:
            disp[col] = disp[col].map("{:.0%}".format)
    for col in ["Peak / Avg ratio", "Usage variability"]:
        if col in disp.columns:
            disp[col] = disp[col].map("{:.2f}".format)
    if "Longest low streak" in disp.columns:
        disp["Longest low streak"] = disp["Longest low streak"].astype(int).astype(str) + " hrs"

    st.dataframe(
        disp.sort_values("Status", ascending=True) if "Status" in disp.columns else disp,
        use_container_width=True,
        hide_index=True,
        height=min(300, 60 + len(disp) * 40),
    )
else:
    st.markdown(
        '<div class="bul-g" style="margin-top:1rem;">✓ No flagged windows for this meter.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION PERFORMANCE (only if ground truth present — recall only)
# ─────────────────────────────────────────────────────────────────────────────
if has_truth:
    from sklearn.metrics import recall_score, confusion_matrix

    result["_pred"] = result["risk_level"].map({"HIGH RISK": 1, "SUSPICIOUS": 1, "NORMAL": 0})
    y_true = result["theft"]
    y_pred = result["_pred"]

    rec    = recall_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    st.markdown("---")
    st.markdown('<p class="section-label">Detection performance</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                padding:1.2rem 1.5rem;font-size:.9rem;color:#374151;
                max-width:540px;line-height:1.7;">
        The system caught <b>{tp}</b> out of <b>{int(tp+fn)}</b> confirmed theft cases
        in this dataset — a detection rate of <b>{rec:.0%}</b>.<br>
        <span style="color:#6b7280;font-size:.82rem;">
            {fn} cases were missed. {fp} normal meters were flagged for review.
        </span>
    </div>
    """, unsafe_allow_html=True)