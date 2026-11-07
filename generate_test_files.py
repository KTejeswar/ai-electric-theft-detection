import os
import numpy as np
import pandas as pd

np.random.seed(42)

dirs = ['for_testing', 'for testing']
for d in dirs:
    os.makedirs(d, exist_ok=True)

# ── 1. High Theft Commercial & Industrial (60 readings: 5 meters x 12 windows) ──
rows_1 = []
meters_1 = [f"HT_COMM_MTR_{i+1:02d}" for i in range(5)]
for m in meters_1:
    for w in range(12):
        if w < 2:  # Baseline normal
            mean_val = float(np.random.uniform(0.035, 0.055))
            std_val = float(mean_val * np.random.uniform(0.3, 0.5))
            zero_p = 0.0
            low_p = float(np.random.uniform(0.05, 0.15))
            low_streak = int(np.random.randint(1, 4))
            p2a = float(np.random.uniform(1.8, 2.4))
            d_n_ratio = float(np.random.uniform(1.6, 2.1))
            th_flag = 0
        else:  # Severe bypass & CT shunting
            mean_val = float(np.random.uniform(0.0005, 0.0025))
            std_val = float(mean_val * np.random.uniform(0.8, 1.8))
            zero_p = float(np.random.uniform(0.40, 0.75))
            low_p = float(np.random.uniform(0.80, 0.98))
            low_streak = int(np.random.randint(18, 30))
            p2a = float(np.random.uniform(4.5, 9.0))
            d_n_ratio = float(np.random.uniform(0.6, 1.3))
            th_flag = 1
        
        max_val = mean_val * p2a
        min_val = max(1e-6, mean_val * 0.05)
        median_val = mean_val * 0.95
        cv_val = std_val / max(mean_val, 1e-6)
        
        rows_1.append({
            "meter_id": m, "window_id": w, "mean": round(mean_val, 6),
            "std": round(std_val, 6), "min": round(min_val, 6), "max": round(max_val, 6),
            "median": round(median_val, 6), "range": round(max_val - min_val, 6),
            "load_factor": round(mean_val / max(max_val, 1e-6), 4),
            "peak_to_avg": round(p2a, 4), "cv": round(cv_val, 4),
            "mean_abs_diff": round(std_val * 0.45, 6), "std_diff": round(std_val * 0.55, 6),
            "day_mean": round(mean_val * 1.25, 6), "night_mean": round(mean_val * 0.75, 6),
            "day_night_ratio": round(d_n_ratio, 4), "zero_pct": round(zero_p, 4),
            "low_pct": round(low_p, 4), "max_low_streak": low_streak,
            "theft": th_flag
        })

df_1 = pd.DataFrame(rows_1)

# ── 2. Healthy Residential Smart Grid (60 readings: 6 meters x 10 windows) ──
rows_2 = []
meters_2 = [f"RES_HEALTHY_MTR_{i+1:02d}" for i in range(6)]
for m in meters_2:
    base_mean = float(np.random.uniform(0.015, 0.028))
    for w in range(10):
        mean_val = float(base_mean * np.random.uniform(0.9, 1.1))
        std_val = float(mean_val * np.random.uniform(0.4, 0.6))
        p2a = float(np.random.uniform(1.8, 2.3))
        max_val = mean_val * p2a
        min_val = max(5e-5, mean_val * 0.08)
        median_val = mean_val * 0.96
        cv_val = std_val / mean_val
        zero_p = 0.0
        low_p = float(np.random.uniform(0.08, 0.18))
        low_streak = int(np.random.randint(1, 5))
        d_n_ratio = float(np.random.uniform(1.6, 2.0))

        rows_2.append({
            "meter_id": m, "window_id": w, "mean": round(mean_val, 6),
            "std": round(std_val, 6), "min": round(min_val, 6), "max": round(max_val, 6),
            "median": round(median_val, 6), "range": round(max_val - min_val, 6),
            "load_factor": round(mean_val / max(max_val, 1e-6), 4),
            "peak_to_avg": round(p2a, 4), "cv": round(cv_val, 4),
            "mean_abs_diff": round(std_val * 0.42, 6), "std_diff": round(std_val * 0.50, 6),
            "day_mean": round(mean_val * 1.3, 6), "night_mean": round(mean_val * 0.7, 6),
            "day_night_ratio": round(d_n_ratio, 4), "zero_pct": round(zero_p, 4),
            "low_pct": round(low_p, 4), "max_low_streak": low_streak,
            "theft": 0
        })

df_2 = pd.DataFrame(rows_2)

# ── 3. Mixed Feeder & Intermittent Tampering (60 readings: 5 meters x 12 windows) ──
rows_3 = []
meters_3 = [f"MIXED_FEEDER_MTR_{i+1:02d}" for i in range(5)]
meter_types = ["Normal", "Suspicious", "Theft", "Suspicious", "Theft"]
for m, m_type in zip(meters_3, meter_types):
    base_mean = float(np.random.uniform(0.018, 0.032))
    for w in range(12):
        if m_type == "Normal":
            mean_val = float(base_mean * np.random.uniform(0.92, 1.08))
            std_val = float(mean_val * 0.48)
            p2a = 2.0
            zero_p = 0.0
            low_p = 0.12
            low_streak = 3
            d_n_ratio = 1.75
            th_flag = 0
        elif m_type == "Suspicious":
            mean_val = float(base_mean * (0.6 if w % 2 == 1 else 1.05))
            std_val = float(mean_val * np.random.uniform(1.2, 1.8))
            p2a = float(np.random.uniform(3.2, 4.5))
            zero_p = float(np.random.uniform(0.12, 0.25))
            low_p = float(np.random.uniform(0.30, 0.48))
            low_streak = int(np.random.randint(7, 13))
            d_n_ratio = float(np.random.uniform(2.2, 3.2))
            th_flag = 0
        else:  # Theft
            if w < 3:
                mean_val = float(base_mean * 1.0)
                std_val = float(mean_val * 0.5)
                p2a = 2.1
                zero_p = 0.0
                low_p = 0.15
                low_streak = 4
                d_n_ratio = 1.8
                th_flag = 0
            else:
                mean_val = float(base_mean * 0.08)
                std_val = float(mean_val * 1.4)
                p2a = float(np.random.uniform(5.0, 8.5))
                zero_p = float(np.random.uniform(0.50, 0.80))
                low_p = float(np.random.uniform(0.85, 0.98))
                low_streak = int(np.random.randint(18, 29))
                d_n_ratio = float(np.random.uniform(0.8, 1.2))
                th_flag = 1
        
        max_val = mean_val * p2a
        min_val = max(1e-6, mean_val * 0.05)
        median_val = mean_val * 0.95
        cv_val = std_val / max(mean_val, 1e-6)

        rows_3.append({
            "meter_id": m, "window_id": w, "mean": round(mean_val, 6),
            "std": round(std_val, 6), "min": round(min_val, 6), "max": round(max_val, 6),
            "median": round(median_val, 6), "range": round(max_val - min_val, 6),
            "load_factor": round(mean_val / max(max_val, 1e-6), 4),
            "peak_to_avg": round(p2a, 4), "cv": round(cv_val, 4),
            "mean_abs_diff": round(std_val * 0.45, 6), "std_diff": round(std_val * 0.55, 6),
            "day_mean": round(mean_val * 1.25, 6), "night_mean": round(mean_val * 0.75, 6),
            "day_night_ratio": round(d_n_ratio, 4), "zero_pct": round(zero_p, 4),
            "low_pct": round(low_p, 4), "max_low_streak": low_streak,
            "theft": th_flag
        })

df_3 = pd.DataFrame(rows_3)

# Save into both 'for_testing' and 'for testing' directories
for d in dirs:
    df_1.to_csv(os.path.join(d, "scenario_1_high_theft_commercial.csv"), index=False)
    df_2.to_csv(os.path.join(d, "scenario_2_residential_healthy_grid.csv"), index=False)
    df_3.to_csv(os.path.join(d, "scenario_3_mixed_intermittent_tampering.csv"), index=False)

print(f"Generated successfully in {dirs}:")
print(f"  - scenario_1_high_theft_commercial.csv: {len(df_1)} readings")
print(f"  - scenario_2_residential_healthy_grid.csv: {len(df_2)} readings")
print(f"  - scenario_3_mixed_intermittent_tampering.csv: {len(df_3)} readings")
