"""
run_pipeline.py
===============
End-to-end execution of the Electricity Theft Detection project:
1. Converts data/raw.csv into 3-min smart meter data data/raw/data.csv.
2. Runs EDA and hourly resampling -> data/processed/clean_resampled.csv.
3. Simulates realistic theft patterns -> data/processed/simulated_theft_data.csv.
4. Slices 24h windows and extracts 17 features -> data/processed/window_features.csv.
5. Cleans zero-mean uninformative normal windows -> data/processed/window_features_clean.csv.
6. Trains Anomaly Models (StandardScaler, IsolationForest, LocalOutlierFactor) -> models/.
7. Trains Supervised Models (RandomForest, XGBoost) and tunes threshold -> models/.
8. Evaluates the full Ensemble Confidence Scoring system and prints test metrics.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_DATA_PATH = os.path.join(RAW_DIR, "data.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# STEP 1: CONVERT RAW.CSV -> DATA/RAW/DATA.CSV
# ==============================================================================
def step1_convert_raw():
    print("\n" + "="*70)
    print("STEP 1: Converting raw.csv into 3-min smart meter format (data/raw/data.csv)")
    print("="*70)

    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Source file not found: {RAW_CSV_PATH}")

    print("Loading raw.csv ...")
    raw = pd.read_csv(RAW_CSV_PATH, header=0, skiprows=[1], low_memory=False)

    id_col = raw.columns[0]
    label_col = raw.columns[-1]
    date_cols = list(raw.columns[1:-1])

    for col in date_cols:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    valid_days = raw[date_cols].notna().sum(axis=1)
    means = raw[date_cols].mean(axis=1)
    stds = raw[date_cols].std(axis=1)

    # Filter high-quality residential/commercial meters
    mask = (valid_days >= 330) & (means >= 2.0) & (means <= 25.0) & (stds >= 0.5)
    filtered = raw[mask].copy()
    print(f"Eligible meters with >=330 days & realistic consumption: {len(filtered)}")

    # Sample 60 meters (stratified by CHK_STATE: 50 normal, 10 known theft if available)
    n_sample = 60
    normal_pool = filtered[filtered[label_col] == 0]
    theft_pool = filtered[filtered[label_col] == 1]

    n_theft_sample = min(10, len(theft_pool))
    n_norm_sample = n_sample - n_theft_sample

    sampled_norm = normal_pool.sample(n=n_norm_sample, random_state=42)
    sampled_theft = theft_pool.sample(n=n_theft_sample, random_state=42)
    sampled = pd.concat([sampled_norm, sampled_theft]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Selected {len(sampled)} meters for pipeline expansion.")

    def parse_date_col(s: str) -> pd.Timestamp:
        parts = s.split("-")
        day, month, yr = int(parts[0]), int(parts[1]), int(parts[2])
        full_yr = 2000 + yr if yr < 100 else yr
        return pd.Timestamp(year=full_yr, month=month, day=day)

    date_map = {c: parse_date_col(c) for c in date_cols}

    long_df = sampled[[id_col] + date_cols].melt(
        id_vars=id_col,
        value_vars=date_cols,
        var_name="date_str",
        value_name="kWh_day"
    )
    long_df["date"] = long_df["date_str"].map(date_map)
    long_df = long_df.drop(columns=["date_str"]).dropna(subset=["kWh_day"])
    long_df["kWh_day"] = long_df["kWh_day"].clip(lower=0.0)
    long_df = long_df.sort_values([id_col, "date"]).reset_index(drop=True)

    print(f"Total valid meter-days to expand: {len(long_df)}")

    # 480 slots per day (3-min interval)
    SLOTS_PER_DAY = 480
    SLOT_DURATION_H = 3 / 60
    NOMINAL_VOLTAGE = 230.0
    NOMINAL_FREQ = 50.0

    rng = np.random.default_rng(42)
    slot_hours = np.linspace(0, 24, SLOTS_PER_DAY, endpoint=False)

    # Realistic 2-peak diurnal load curve
    morning_peak = np.exp(-0.5 * ((slot_hours - 9.0) / 2.5) ** 2)
    evening_peak = np.exp(-0.5 * ((slot_hours - 19.0) / 2.5) ** 2)
    night_trough = 1.0 - np.exp(-0.5 * ((slot_hours - 3.0) / 2.0) ** 2) * 0.6
    base_weights = (0.5 * morning_peak + 0.5 * evening_peak) * night_trough
    base_weights = base_weights / base_weights.sum()

    n = len(long_df)
    total_rows = n * SLOTS_PER_DAY

    out_ts = np.empty(total_rows, dtype="datetime64[ns]")
    out_kwh = np.empty(total_rows, dtype=np.float64)
    out_voltage = np.empty(total_rows, dtype=np.float64)
    out_current = np.empty(total_rows, dtype=np.float64)
    out_freq = np.empty(total_rows, dtype=np.float64)
    out_meter = np.empty(total_rows, dtype=object)

    slot_offsets_ns = (slot_hours * 3600 * 1e9).astype(np.int64)

    meter_ids = long_df[id_col].astype(str).tolist()
    dates = long_df["date"].values
    kwh_days = long_df["kWh_day"].values

    print("Generating 3-minute readings...")
    for i in range(n):
        s_idx = i * SLOTS_PER_DAY
        e_idx = s_idx + SLOTS_PER_DAY

        day_ns = dates[i].astype("datetime64[ns]").astype(np.int64)
        out_ts[s_idx:e_idx] = (day_ns + slot_offsets_ns).astype("datetime64[ns]")

        # Slight noise in shape
        noise = 1.0 + rng.normal(0, 0.02, SLOTS_PER_DAY)
        w = base_weights * noise
        w /= w.sum()
        slot_kwh = w * kwh_days[i]
        out_kwh[s_idx:e_idx] = np.round(slot_kwh, 5)

        # Voltage
        voltages = NOMINAL_VOLTAGE + rng.normal(0, 1.0, SLOTS_PER_DAY)
        out_voltage[s_idx:e_idx] = np.round(voltages, 2)

        # Current (I = P/V)
        power_kw = slot_kwh / SLOT_DURATION_H
        current = (power_kw * 1000.0) / voltages
        out_current[s_idx:e_idx] = np.round(current, 2)

        # Frequency
        out_freq[s_idx:e_idx] = np.round(NOMINAL_FREQ + rng.normal(0, 0.05, SLOTS_PER_DAY), 2)

        # Meter ID
        out_meter[s_idx:e_idx] = f"M{meter_ids[i]}"

    data_df = pd.DataFrame({
        "x_Timestamp": out_ts,
        "t_kWh": out_kwh,
        "z_Avg Voltage (Volt)": out_voltage,
        "z_Avg Current (Amp)": out_current,
        "y_Freq (Hz)": out_freq,
        "meter": out_meter,
    })

    data_df = data_df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)
    data_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(data_df):,} 3-min rows to {RAW_DATA_PATH}")
    print(f"Meters: {data_df['meter'].nunique()} | Date range: {data_df['x_Timestamp'].min()} to {data_df['x_Timestamp'].max()}")
    return data_df


# ==============================================================================
# STEP 2: EDA & HOURLY RESAMPLING -> CLEAN_RESAMPLED.CSV
# ==============================================================================
def step2_eda_resample():
    print("\n" + "="*70)
    print("STEP 2: Hourly Resampling & Cleaning (EDA.ipynb)")
    print("="*70)

    df = pd.read_csv(RAW_DATA_PATH)
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])

    print(f"Raw data shape: {df.shape}")

    cols = ['t_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)']

    # Group by meter and resample 1H
    resampled_list = []
    for m, group in df.groupby('meter'):
        g = group.set_index('x_Timestamp')[cols].resample('1h').mean()
        # Interpolate small gaps
        g = g.interpolate(method='time', limit=2)
        g['meter'] = m
        resampled_list.append(g.reset_index())

    df_resampled = pd.concat(resampled_list, ignore_index=True)

    # Add temporal columns
    df_resampled['hour'] = df_resampled['x_Timestamp'].dt.hour
    df_resampled['day'] = df_resampled['x_Timestamp'].dt.day
    df_resampled['month'] = df_resampled['x_Timestamp'].dt.month
    df_resampled['day_of_week'] = df_resampled['x_Timestamp'].dt.dayofweek

    # Drop any remaining null rows
    before = len(df_resampled)
    df_resampled = df_resampled.dropna().sort_values(['meter', 'x_Timestamp']).reset_index(drop=True)
    after = len(df_resampled)
    print(f"Resampled rows: {before} -> {after} after dropping NaN ({before - after} dropped)")

    out_clean = os.path.join(PROCESSED_DIR, "clean_resampled.csv")
    out_sim_base = os.path.join(PROCESSED_DIR, "simulation_base.csv")

    df_resampled.to_csv(out_clean, index=False)
    df_resampled.to_csv(out_sim_base, index=False)
    print(f"Saved: {out_clean}")
    return df_resampled


# ==============================================================================
# STEP 3: THEFT SIMULATION -> SIMULATED_THEFT_DATA.CSV
# ==============================================================================
def step3_simulation():
    print("\n" + "="*70)
    print("STEP 3: Simulating Theft Patterns (simulation_model.ipynb)")
    print("="*70)

    clean_path = os.path.join(PROCESSED_DIR, "clean_resampled.csv")
    df_sim = pd.read_csv(clean_path)
    df_sim['x_Timestamp'] = pd.to_datetime(df_sim['x_Timestamp'])
    df_sim['theft'] = 0

    np.random.seed(42)
    THEFT_TYPES = ['load', 'flat', 'intermittent', 'drop', 'bypass']

    def apply_theft_pattern(df, mask, theft_type):
        df = df.copy()
        if theft_type == 'load':
            factor = np.random.uniform(0.3, 0.5)
            df.loc[mask, 't_kWh'] *= factor
        elif theft_type == 'flat':
            mean_val = df.loc[mask, 't_kWh'].mean()
            if mean_val > 0:
                df.loc[mask, 't_kWh'] = mean_val
        elif theft_type == 'intermittent':
            idx = df.loc[mask].index
            df.loc[idx[::2], 't_kWh'] *= 0.4
        elif theft_type == 'drop':
            df.loc[mask, 't_kWh'] *= 0.2
        elif theft_type == 'bypass':
            df.loc[mask, 't_kWh'] *= 0.05
        return df

    def no_overlap(existing_windows, meter, start, end):
        for (m, s, e, _) in existing_windows:
            if m == meter:
                if not (end < s or start > e):
                    return False
        return True

    meters = df_sim['meter'].unique()
    # 14 meters flagged for theft (~23% of 60 meters)
    theft_meters = np.random.choice(meters, size=14, replace=False)
    print("Selected 14 theft meters:", theft_meters)

    all_windows = []
    for meter in theft_meters:
        meter_df = df_sim[df_sim['meter'] == meter]
        timestamps = meter_df['x_Timestamp'].sort_values().unique()

        n_windows = np.random.randint(6, 11)
        attempts = 0

        while len([w for w in all_windows if w[0] == meter]) < n_windows:
            attempts += 1
            if attempts > 500:
                break

            start_idx = np.random.randint(0, len(timestamps) - 400)
            duration = np.random.randint(168, 360)  # 7-15 days in hours
            end_idx = min(start_idx + duration, len(timestamps) - 1)

            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]

            if no_overlap(all_windows, meter, start_time, end_time):
                theft_type = np.random.choice(THEFT_TYPES)
                all_windows.append((meter, start_time, end_time, theft_type))

    print(f"Generated {len(all_windows)} non-overlapping theft windows.")

    for meter, start, end, theft_type in all_windows:
        mask = (
            (df_sim['meter'] == meter) &
            (df_sim['x_Timestamp'] >= start) &
            (df_sim['x_Timestamp'] <= end)
        )
        df_sim = apply_theft_pattern(df_sim, mask, theft_type)

        if theft_type == 'intermittent':
            idx = df_sim.loc[mask].index
            df_sim.loc[idx[::2], 'theft'] = 1
        else:
            df_sim.loc[mask, 'theft'] = 1

    theft_cnt = df_sim['theft'].sum()
    total_cnt = len(df_sim)
    print(f"Theft rows: {theft_cnt:,} / {total_cnt:,} ({theft_cnt/total_cnt*100:.2f}%)")

    sim_out = os.path.join(PROCESSED_DIR, "simulated_theft_data.csv")
    df_sim.to_csv(sim_out, index=False)
    print(f"Saved: {sim_out}")
    return df_sim


# ==============================================================================
# STEP 4: FEATURE ENGINEERING -> WINDOW_FEATURES.CSV
# ==============================================================================
def step4_feature_engineering():
    print("\n" + "="*70)
    print("STEP 4: Slicing 24h Windows & Extracting 17 Features (feature_engineering.ipynb)")
    print("="*70)

    sim_path = os.path.join(PROCESSED_DIR, "simulated_theft_data.csv")
    df = pd.read_csv(sim_path)

    df = df.rename(columns={
        'x_Timestamp': 'timestamp',
        'meter': 'meter_id',
        't_kWh': 'consumption'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['meter_id', 'timestamp']).reset_index(drop=True)

    # Ensure full hourly continuity per meter
    fixed_data = []
    for meter_id, group in df.groupby('meter_id'):
        group = group.sort_values('timestamp').set_index('timestamp')
        full_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='1h')
        group = group.reindex(full_index)
        group['meter_id'] = meter_id
        group['consumption'] = group['consumption'].fillna(0.0)
        group['theft'] = group['theft'].fillna(0)
        group['z_Avg Voltage (Volt)'] = group['z_Avg Voltage (Volt)'].ffill().bfill()
        group['z_Avg Current (Amp)'] = group['z_Avg Current (Amp)'].ffill().bfill()
        group['y_Freq (Hz)'] = group['y_Freq (Hz)'].ffill().bfill()
        group = group.reset_index().rename(columns={'index': 'timestamp'})
        fixed_data.append(group)

    df_fixed = pd.concat(fixed_data, ignore_index=True)

    def extract_features(window):
        features = {}
        c = window['consumption'].values

        # Statistical
        features['mean'] = float(np.mean(c))
        features['std'] = float(np.std(c))
        features['min'] = float(np.min(c))
        features['max'] = float(np.max(c))
        features['median'] = float(np.median(c))
        features['range'] = float(features['max'] - features['min'])

        # Behavioral
        features['load_factor'] = float(features['mean'] / features['max']) if features['max'] != 0 else 0.0
        features['peak_to_avg'] = float(features['max'] / features['mean']) if features['mean'] != 0 else 0.0
        features['cv'] = float(features['std'] / features['mean']) if features['mean'] != 0 else 0.0

        # Dynamics
        diff = np.diff(c)
        features['mean_abs_diff'] = float(np.mean(np.abs(diff))) if len(diff) > 0 else 0.0
        features['std_diff'] = float(np.std(diff)) if len(diff) > 0 else 0.0

        # Temporal
        hours = window['timestamp'].dt.hour.values
        day_mask = (hours >= 6) & (hours < 18)
        night_mask = ~day_mask

        day_c = c[day_mask]
        night_c = c[night_mask]

        day_mean = float(day_c.mean()) if len(day_c) > 0 else 0.0
        night_mean = float(night_c.mean()) if len(night_c) > 0 else 0.0

        features['day_mean'] = day_mean
        features['night_mean'] = night_mean
        features['day_night_ratio'] = float(day_mean / night_mean) if night_mean != 0 else 0.0

        # Theft-sensitive
        features['zero_pct'] = float(np.sum(c == 0.0) / len(c))
        features['low_pct'] = float(np.sum(c < (0.2 * features['mean'])) / len(c)) if features['mean'] > 0 else 1.0

        low_mask = c < (0.2 * features['mean']) if features['mean'] > 0 else (c == 0.0)
        max_streak = 0
        current_streak = 0
        for val in low_mask:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        features['max_low_streak'] = int(max_streak)

        # Label
        features['theft'] = int(window['theft'].max())
        return features

    WINDOW_SIZE = 24
    all_features = []

    for meter_id, group in df_fixed.groupby('meter_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        n_pts = len(group)
        for i in range(0, n_pts - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = group.iloc[i : i + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                feats = extract_features(window)
                feats['meter_id'] = meter_id
                feats['window_id'] = i // WINDOW_SIZE
                all_features.append(feats)

    features_df = pd.DataFrame(all_features)
    cols = [
        'meter_id', 'window_id',
        'mean', 'std', 'min', 'max', 'median', 'range',
        'load_factor', 'peak_to_avg', 'cv',
        'mean_abs_diff', 'std_diff',
        'day_mean', 'night_mean', 'day_night_ratio',
        'zero_pct', 'low_pct', 'max_low_streak',
        'theft'
    ]
    features_df = features_df[cols]

    out_features = os.path.join(PROCESSED_DIR, "window_features.csv")
    features_df.to_csv(out_features, index=False)
    print(f"Extracted {len(features_df):,} windows across {features_df['meter_id'].nunique()} meters.")
    print(f"Theft windows: {(features_df['theft']==1).sum():,} ({features_df['theft'].mean()*100:.2f}%)")
    print(f"Saved: {out_features}")
    return features_df


# ==============================================================================
# STEP 5: CLEANING -> WINDOW_FEATURES_CLEAN.CSV
# ==============================================================================
def step5_cleaning():
    print("\n" + "="*70)
    print("STEP 5: Removing uninformative normal zero windows (cleaning.ipynb)")
    print("="*70)

    in_path = os.path.join(PROCESSED_DIR, "window_features.csv")
    out_path = os.path.join(PROCESSED_DIR, "window_features_clean.csv")

    df = pd.read_csv(in_path)
    before = len(df)

    # Filter only zero-mean normal windows
    valid_mask = ~((df["mean"] == 0) & (df["theft"] == 0))
    df_clean = df[valid_mask].reset_index(drop=True)
    after = len(df_clean)

    print(f"Windows before: {before} | Windows after: {after} ({before - after} uninformative rows removed)")
    print("Theft distribution in cleaned data:")
    print(df_clean['theft'].value_counts())

    df_clean.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_clean


# ==============================================================================
# STEP 6: ANOMALY MODELS TRAINING
# ==============================================================================
def step6_train_anomaly():
    print("\n" + "="*70)
    print("STEP 6: Training Anomaly Detection Models (anomaly_model.ipynb)")
    print("="*70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    feat_path = os.path.join(PROCESSED_DIR, "window_features.csv")
    df = pd.read_csv(feat_path)

    # Fit ONLY on normal windows (theft == 0)
    df_normal = df[df["theft"] == 0].copy()
    drop_cols = ["meter_id", "window_id", "theft"]
    X_normal = df_normal.drop(columns=drop_cols)

    print(f"Training anomaly models on {len(X_normal):,} normal windows with {X_normal.shape[1]} features.")

    # 1. Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    scaler_path = os.path.join(MODELS_DIR, "scaler_anomaly.pkl")
    joblib.dump(scaler, scaler_path)
    print("StandardScaler fitted and saved.")

    # 2. Isolation Forest
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_scaled)
    iso_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    joblib.dump(iso_model, iso_path)
    print("IsolationForest fitted and saved.")

    # 3. Local Outlier Factor (novelty=True)
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05,
        novelty=True,
        n_jobs=-1
    )
    lof_model.fit(X_scaled)
    lof_path = os.path.join(MODELS_DIR, "lof_model.pkl")
    joblib.dump(lof_model, lof_path)
    print("LocalOutlierFactor (novelty mode) fitted and saved.")


# ==============================================================================
# STEP 7: SUPERVISED MODELS TRAINING
# ==============================================================================
def step7_train_supervised():
    print("\n" + "="*70)
    print("STEP 7: Training Supervised Models & Tuning Thresholds (supervised_model.ipynb)")
    print("="*70)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report, confusion_matrix

    feat_path = os.path.join(PROCESSED_DIR, "window_features.csv")
    df = pd.read_csv(feat_path)

    drop_cols = ["meter_id", "window_id", "theft"]
    X = df.drop(columns=drop_cols)
    y = df["theft"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_normal = int((y_train == 0).sum())
    n_theft = int((y_train == 1).sum())
    ratio = n_normal / n_theft if n_theft > 0 else 1.0

    print(f"Train set: {len(X_train)} (Normal: {n_normal}, Theft: {n_theft}, Class ratio: {ratio:.2f})")
    print(f"Test set:  {len(X_test)} (Normal: {(y_test==0).sum()}, Theft: {(y_test==1).sum()})")

    # 1. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest trained.")

    # 2. XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost trained.")

    # 3. Threshold tuning on Test Set
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    ens_probs = (rf_probs + xgb_probs) / 2.0

    print("\n--- Supervised Ensemble Threshold Tuning (on Test Set) ---")
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<8}")
    print("-" * 44)
    best_thresh = 0.2
    best_f1 = 0.0

    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
        preds = (ens_probs >= thresh).astype(int)
        r = recall_score(y_test, preds)
        p = precision_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        print(f"{thresh:<12.2f} {r:<10.3f} {p:<12.3f} {f:<8.3f}")
        if r >= 0.90 and p > best_f1:  # prioritize high recall with best precision
            best_f1 = p
            best_thresh = thresh

    if best_thresh is None:
        best_thresh = 0.3

    print(f"\nSelected Decision Threshold: {best_thresh}")

    # Save models
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    thresh_path = os.path.join(MODELS_DIR, "thresholds.pkl")

    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump({"threshold": best_thresh}, thresh_path)
    print("Supervised models and thresholds saved to models/.")

    return X_test, y_test, rf_model, xgb_model, best_thresh


# ==============================================================================
# STEP 8: TESTING & ENSEMBLE EVALUATION (ensemble.ipynb)
# ==============================================================================
def step8_evaluate_ensemble(best_thresh=0.2):
    print("\n" + "="*70)
    print("STEP 8: Evaluating Final Hybrid Ensemble Scoring (ensemble.ipynb)")
    print("="*70)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    feat_path = os.path.join(PROCESSED_DIR, "window_features.csv")
    df = pd.read_csv(feat_path)

    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    iso_model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    lof_model = joblib.load(os.path.join(MODELS_DIR, "lof_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_anomaly.pkl"))

    drop_cols = ["meter_id", "window_id", "theft"]
    X = df.drop(columns=drop_cols)
    y = df["theft"]

    # 1. Model signals
    rf_probs = rf_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    ens_probs = (rf_probs + xgb_probs) / 2.0

    X_scaled = scaler.transform(X)
    iso_flag = (iso_model.predict(X_scaled) == -1).astype(int)
    lof_flag = (lof_model.predict(X_scaled) == -1).astype(int)

    # 2. Feature signals
    zero_flag = (X["zero_pct"] >= 0.5).astype(int)
    streak_flag = (X["max_low_streak"] >= 16).astype(int)
    cv_flag = (X["cv"] >= 4.0).astype(int)
    extreme_flag = ((streak_flag + cv_flag) >= 1).astype(int)

    # 3. Confidence score (0-5 points)
    def compute_score(prob, iso_f, lof_f, z_f, ext_f):
        score = 0
        if prob >= 0.6:
            score += 2
        elif prob >= 0.4:
            score += 1
        if iso_f == 1 and lof_f == 1:
            score += 2
        elif iso_f == 1 or lof_f == 1:
            score += 1
        score += z_f
        score += ext_f
        return min(score, 5)

    scores = [
        compute_score(p, i, l, z, e)
        for p, i, l, z, e in zip(ens_probs, iso_flag, lof_flag, zero_flag, extreme_flag)
    ]
    scores = np.array(scores)

    def score_to_label(s):
        if s >= 3:
            return "HIGH RISK"
        elif s >= 1:
            return "SUSPICIOUS"
        else:
            return "NORMAL"

    df["fraud_prob"] = ens_probs
    df["iso_flag"] = iso_flag
    df["lof_flag"] = lof_flag
    df["confidence"] = scores
    df["final_label"] = [score_to_label(s) for s in scores]

    print("\n1. Final Tier Distribution across all windows:")
    print(df["final_label"].value_counts())

    print("\n2. Purity vs Ground Truth (Cross-tabulation):")
    print(pd.crosstab(df["final_label"], df["theft"], margins=True))

    # Evaluation A: HIGH RISK windows alone as positive
    df["high_risk_binary"] = df["final_label"].map({"HIGH RISK": 1, "SUSPICIOUS": 0, "NORMAL": 0})
    print("\n3. HIGH RISK Tier Metrics (Immediate Field Inspection Tier):")
    print(classification_report(df["theft"], df["high_risk_binary"], target_names=["Normal/Suspicious", "High Risk"]))

    # Evaluation B: Any Alert (SUSPICIOUS or HIGH RISK) vs NORMAL (Full Theft Recall)
    df["any_alert_binary"] = df["final_label"].map({"HIGH RISK": 1, "SUSPICIOUS": 1, "NORMAL": 0})
    print("\n4. Overall Theft Detection (Any Alert: Suspicious or High Risk vs Normal):")
    print(classification_report(df["theft"], df["any_alert_binary"], target_names=["Normal", "Theft Detected"]))

    # Evaluation C: Meter-level Detection Rate
    meter_pred = df.groupby("meter_id")["any_alert_binary"].max()
    meter_true = df.groupby("meter_id")["theft"].max()
    print("\n5. Meter-Level Identification Report:")
    print(classification_report(meter_true, meter_pred, target_names=["Normal Meter", "Theft Meter"]))

    print("\nPipeline execution complete! All models trained and saved to models/.")


if __name__ == "__main__":
    t0 = time.time()
    step1_convert_raw()
    step2_eda_resample()
    step3_simulation()
    step4_feature_engineering()
    step5_cleaning()
    step6_train_anomaly()
    X_test, y_test, rf, xgb, best_thresh = step7_train_supervised()
    step8_evaluate_ensemble(best_thresh)
    print(f"\nTotal elapsed time: {time.time() - t0:.2f} seconds")
