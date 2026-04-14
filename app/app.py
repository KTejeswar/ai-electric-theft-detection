import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Energy Theft Detection",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("⚡ Hybrid Energy Theft Detection System")

st.markdown("""
Upload feature-engineered smart meter data and detect potential electricity theft using a hybrid ML system.
""")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "notebooks", "models")

    rf_model = joblib.load(os.path.join(base_path, "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join(base_path, "xgb_model.pkl"))
    thresholds = joblib.load(os.path.join(base_path, "thresholds.pkl"))

    iso_forest = joblib.load(os.path.join(base_path, "isolation_forest.pkl"))
    lof_model = joblib.load(os.path.join(base_path, "lof_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler_anomaly.pkl"))

    return rf_model, xgb_model, thresholds, iso_forest, lof_model, scaler

# -------------------------------
# Initialize Models
# -------------------------------
try:
    rf_model, xgb_model, thresholds, iso_forest, lof_model, scaler = load_models()
    st.success("Models loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload window_features.csv", type=["csv"])

# -------------------------------
# MAIN PIPELINE
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully ✅")

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape}")

    # -------------------------------
    # Prepare Features
    # -------------------------------
    df_model = df.copy()

    drop_cols = ["meter_id"]

    if "theft" in df_model.columns:
        drop_cols.append("theft")

    if "window_start" in df_model.columns:
        drop_cols.append("window_start")

    X = df_model.drop(columns=drop_cols, errors="ignore")

    # Remove invalid rows
    valid_mask = (X["mean"] > 0)
    X = X[valid_mask]
    df = df[valid_mask].reset_index(drop=True)

    # Align features
    model_features = rf_model.feature_names_in_

    for col in model_features:
        if col not in X.columns:
            X[col] = 0

    X = X[model_features]

    # -------------------------------
    # Supervised
    # -------------------------------
    rf_probs = rf_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    ensemble_probs = (rf_probs + xgb_probs) / 2

    if isinstance(thresholds, dict):
        threshold = thresholds.get("threshold", 0.3)
    else:
        threshold = thresholds

    supervised_pred = (ensemble_probs >= threshold).astype(int)

    # -------------------------------
    # Anomaly
    # -------------------------------
    X_scaled = scaler.transform(X)

    iso_flag = (iso_forest.predict(X_scaled) == -1).astype(int)
    lof_flag = (lof_model.predict(X_scaled) == -1).astype(int)

    anomaly_flag = ((iso_flag + lof_flag) >= 1).astype(int)

    # -------------------------------
    # Hybrid Logic
    # -------------------------------
    final_labels = []

    for sup, anom in zip(supervised_pred, anomaly_flag):
        if sup == 1 and anom == 1:
            final_labels.append("HIGH RISK")
        elif sup == 0 and anom == 1:
            final_labels.append("SUSPICIOUS")
        elif sup == 1 and anom == 0:
            final_labels.append("SUSPICIOUS")
        else:
            final_labels.append("NORMAL")

    df["risk_level"] = final_labels
    df["anomaly_flag"] = anomaly_flag
    df["fraud_prob"] = ensemble_probs

    # -------------------------------
    # Explainability
    # -------------------------------
    reasons = []
    for _, row in df.iterrows():
        r = []
        if row["fraud_prob"] > 0.3:
            r.append("High fraud probability")
        if row["anomaly_flag"] == 1:
            r.append("Anomalous behavior")
        if row["zero_pct"] > 0.8:
            r.append("High zero consumption")
        if row["peak_to_avg"] > 5:
            r.append("Spiky usage")
        if row["load_factor"] < 0.1:
            r.append("Low consistency")

        if not r:
            r.append("Normal behavior")

        reasons.append(", ".join(r))

    df["reason"] = reasons

    # -------------------------------
    # Results
    # -------------------------------
    st.subheader("🔍 Detection Results")
    st.dataframe(df[["meter_id", "risk_level", "fraud_prob", "anomaly_flag", "reason"]].head())

    # -------------------------------
    # Risk Summary
    # -------------------------------
    st.subheader("🚨 Risk Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 HIGH RISK", (df["risk_level"] == "HIGH RISK").sum())
    col2.metric("🟡 SUSPICIOUS", (df["risk_level"] == "SUSPICIOUS").sum())
    col3.metric("🟢 NORMAL", (df["risk_level"] == "NORMAL").sum())

    # -------------------------------
    # Meter Analysis
    # -------------------------------
    st.subheader("🔎 Meter Analysis")

    selected_meter = st.selectbox("Select Meter ID", df["meter_id"].unique())
    filtered_df = df[df["meter_id"] == selected_meter]

    st.dataframe(filtered_df.head())

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📈 Meter Behavior Trends")

    chart_data = filtered_df[["window_id", "mean", "peak_to_avg", "zero_pct"]]
    st.line_chart(chart_data.set_index("window_id"))

    # -------------------------------
# Dynamic Interpretation
# -------------------------------
    st.subheader("🧠 Interpretation")

    avg_zero = filtered_df["zero_pct"].mean()
    avg_peak = filtered_df["peak_to_avg"].mean()
    avg_load = filtered_df["load_factor"].mean()

    insight = []

    if avg_zero > 0.5:
        insight.append("high periods of zero consumption")
    if avg_peak > 4:
        insight.append("frequent abnormal spikes")
    if avg_load < 0.2:
        insight.append("inconsistent usage patterns")

    if len(insight) == 0:
        text = "This meter shows stable and consistent energy consumption patterns."
    else:
        text = f"This meter shows {', '.join(insight)}, which may indicate irregular or suspicious behavior."

    st.write(text)

    # -------------------------------
    # Suspicious Windows
    # -------------------------------
    st.subheader("⚠️ Suspicious Windows")

    suspicious = filtered_df[
        filtered_df["risk_level"].isin(["HIGH RISK", "SUSPICIOUS"])
    ]

    st.dataframe(suspicious[["window_id", "risk_level", "fraud_prob", "reason"]])

    # -------------------------------
    # Peer Comparison
    # -------------------------------
    st.subheader("📊 Peer Comparison")

    features = ["mean", "load_factor", "peak_to_avg", "zero_pct"]

    comp = pd.DataFrame({
        "Feature": features,
        "Meter": filtered_df[features].mean().values,
        "Population": df[features].mean().values
    })

    st.dataframe(comp)

    # -------------------------------
    # Ranking
    # -------------------------------
    st.subheader("📋 Suspicious Meter Ranking")

    df["risk_score"] = df["fraud_prob"] + df["anomaly_flag"]

    ranking_df = df.groupby("meter_id").agg({
        "risk_score": "mean",
        "risk_level": lambda x: (x == "HIGH RISK").sum(),
        "anomaly_flag": "sum"
    }).rename(columns={
        "risk_score": "avg_risk_score",
        "risk_level": "high_risk_count",    
        "anomaly_flag": "total_anomalies"
    }).reset_index()

    ranking_df = ranking_df.sort_values(by="avg_risk_score", ascending=False)

    st.dataframe(ranking_df.head(10))

    # -------------------------------
    # Risk Distribution
    # -------------------------------
    st.subheader("📈 Risk Distribution")
    st.bar_chart(df["risk_level"].value_counts())

else:
    st.info("Please upload a CSV file to proceed.")