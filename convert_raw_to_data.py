"""
convert_raw_to_data.py
======================
Converts the wide-format raw.csv (SGCC-style: CONS_NO rows, daily date columns,
CHK_STATE label column, 2015 data) into the long-format data.csv that the
project's EDA.ipynb notebook expects.

Target schema (data/raw/data.csv):
  x_Timestamp            datetime64[ns]  – timestamp at start of each 3-min slot
  t_kWh                  float64         – energy reading (kWh per 3-min slot)
  z_Avg Voltage (Volt)   float64         – average voltage (synthesised)
  z_Avg Current (Amp)    float64         – average current (synthesised)
  y_Freq (Hz)            float64         – grid frequency (synthesised)
  meter                  object          – meter ID (string, e.g. "M00001")

Methodology
-----------
raw.csv contains one row per customer (CONS_NO) with 365 daily kWh columns
(dd-mm-yy format) plus a CHK_STATE flag column.  The pipeline expects 3-minute
interval data with voltage, current, and frequency.

Transformation steps:
  1. Skip the index row (row 2 in the file, which is just sequential numbers).
  2. Keep only rows that have at least some non-null energy readings.
  3. Melt the wide table from (CONS_NO × daily) → long (CONS_NO, date, kWh_day).
  4. Convert each daily kWh value to per-3-minute slots:
       – The day is split into 480 slots of 3 minutes each (24 h × 20 slots/h).
       – A synthetic intra-day load curve is generated using a cosine pattern
         that approximates residential usage (low at night, peak morning/evening).
       – Readings are scaled so they sum to the daily kWh value.
  5. Synthesise voltage, current, and frequency columns:
       – Voltage: 230 V ± small white noise (σ=1 V).
       – Current: derived from kWh and voltage via I = P / V  (P in Watts).
       – Frequency: 50 Hz ± small white noise (σ=0.05 Hz).
  6. Name each meter "M{CONS_NO:0>8}".
  7. Write output to data/raw/data.csv.

Run:
    python convert_raw_to_data.py
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_CSV      = os.path.join(SCRIPT_DIR, "data", "raw.csv")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "data", "raw")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SLOTS_PER_DAY   = 480          # 24 h × 20 slots/h  (3-min each)
SLOT_DURATION_H = 3 / 60       # hours per slot = 0.05
NOMINAL_VOLTAGE = 230.0        # Volts
NOMINAL_FREQ    = 50.0         # Hz
VOLTAGE_NOISE   = 1.0          # σ in Volts
FREQ_NOISE      = 0.05         # σ in Hz
RNG_SEED        = 42

# ── 1. Load raw.csv ───────────────────────────────────────────────────────────
print("Loading raw.csv …")
raw = pd.read_csv(RAW_CSV, header=0, low_memory=False)

# Row 0 of the parsed DataFrame is the numeric index row (1,2,3,…,365,)
# We detect and drop it: its CONS_NO cell is NaN or empty.
if pd.isna(raw.iloc[0, 0]) or str(raw.iloc[0, 0]).strip() == "":
    raw = raw.iloc[1:].reset_index(drop=True)
    print("  Dropped numeric index row.")

# ── 2. Identify columns ───────────────────────────────────────────────────────
# First column is CONS_NO, last column is CHK_STATE, the rest are date columns.
id_col     = raw.columns[0]           # 'CONS_NO'
label_col  = raw.columns[-1]          # 'CHK_STATE'
date_cols  = list(raw.columns[1:-1])  # 365 date strings like '01-01-15'

print(f"  {len(raw)} meters found.")
print(f"  {len(date_cols)} date columns found (first: {date_cols[0]}, last: {date_cols[-1]}).")

# ── 3. Parse date columns → proper dates ─────────────────────────────────────
def parse_date_col(s: str) -> pd.Timestamp:
    """Parse 'dd-mm-yy' → Timestamp for year 2015."""
    day, month, yr = s.split("-")
    return pd.Timestamp(f"20{yr}-{month}-{day}")

date_map = {col: parse_date_col(col) for col in date_cols}

# ── 4. Convert numeric energy columns ─────────────────────────────────────────
for col in date_cols:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# ── 5. Drop meters with no data at all ────────────────────────────────────────
raw = raw[raw[date_cols].notna().any(axis=1)].copy()
print(f"  {len(raw)} meters retained after dropping all-null rows.")

# ── 6. Melt to long format ────────────────────────────────────────────────────
print("Melting to long format …")
long = raw[[id_col] + date_cols].melt(
    id_vars=id_col,
    value_vars=date_cols,
    var_name="date_str",
    value_name="kWh_day",
)

# Map date string → Timestamp
long["date"] = long["date_str"].map(date_map)
long = long.drop(columns=["date_str"])

# Drop rows where kWh_day is NaN (meter not yet installed / data gap)
long = long.dropna(subset=["kWh_day"]).copy()
long["kWh_day"] = long["kWh_day"].clip(lower=0.0)   # clip negative readings

print(f"  {len(long)} (meter, day) pairs retained.")

# ── 7. Build intra-day load curve weights ─────────────────────────────────────
# A cosine pattern: low at 03:00, peaks at ~09:00 and ~19:00.
# Weights sum to 1.0 so that kWh_slot = weight × kWh_day.
rng = np.random.default_rng(RNG_SEED)

slot_hours = np.array([i * SLOT_DURATION_H for i in range(SLOTS_PER_DAY)])  # [0, 23.95)

# Two-peak cosine blend
morning_peak = np.exp(-0.5 * ((slot_hours - 9.0) / 2.5) ** 2)
evening_peak = np.exp(-0.5 * ((slot_hours - 19.0) / 2.5) ** 2)
night_trough = 1 - np.exp(-0.5 * ((slot_hours - 3.0) / 2.0) ** 2) * 0.6

base_weights = (0.5 * morning_peak + 0.5 * evening_peak) * night_trough
base_weights = base_weights / base_weights.sum()          # normalise → sum = 1.0

# ── 8. Expand each (meter, day, kWh_day) → 480 3-min rows ────────────────────
print("Expanding to 3-minute slots … (this may take a moment)")

n = len(long)
total_rows = n * SLOTS_PER_DAY

# Pre-allocate output arrays
out_timestamp  = np.empty(total_rows, dtype="datetime64[ns]")
out_kwh        = np.empty(total_rows, dtype=np.float64)
out_voltage    = np.empty(total_rows, dtype=np.float64)
out_current    = np.empty(total_rows, dtype=np.float64)
out_freq       = np.empty(total_rows, dtype=np.float64)
out_meter      = np.empty(total_rows, dtype=object)

slot_offsets_ns = (slot_hours * 3600 * 1e9).astype(np.int64)  # nanoseconds

for i, (_, row) in enumerate(long.iterrows()):
    start = i * SLOTS_PER_DAY
    end   = start + SLOTS_PER_DAY

    day_ns = np.datetime64(row["date"], "ns").astype(np.int64)

    # Timestamps
    out_timestamp[start:end] = (day_ns + slot_offsets_ns).astype("datetime64[ns]")

    # Energy: distribute daily kWh across 480 slots with slight randomness
    noise = 1.0 + rng.normal(0, 0.02, SLOTS_PER_DAY)
    noisy_weights = base_weights * noise
    noisy_weights = noisy_weights / noisy_weights.sum()
    kwh_slots = noisy_weights * row["kWh_day"]
    out_kwh[start:end] = kwh_slots

    # Voltage: Gaussian noise around nominal
    voltages = NOMINAL_VOLTAGE + rng.normal(0, VOLTAGE_NOISE, SLOTS_PER_DAY)
    out_voltage[start:end] = np.round(voltages, 2)

    # Current: I = P/V ; P (kW) = kWh / slot_duration_h
    power_kw = kwh_slots / SLOT_DURATION_H          # kW per slot
    current  = (power_kw * 1000) / voltages         # Amps  (P in Watts)
    out_current[start:end] = np.round(current, 2)

    # Frequency: Gaussian noise around 50 Hz
    out_freq[start:end] = np.round(
        NOMINAL_FREQ + rng.normal(0, FREQ_NOISE, SLOTS_PER_DAY), 2
    )

    # Meter ID
    meter_id = f"M{str(row[id_col]).strip()}"
    out_meter[start:end] = meter_id

    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{n} meter-days …")

# ── 9. Assemble DataFrame ─────────────────────────────────────────────────────
print("Assembling final DataFrame …")
df = pd.DataFrame({
    "x_Timestamp"          : out_timestamp,
    "t_kWh"                : np.round(out_kwh, 5),
    "z_Avg Voltage (Volt)" : out_voltage,
    "z_Avg Current (Amp)"  : out_current,
    "y_Freq (Hz)"          : out_freq,
    "meter"                : out_meter,
})

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

# Sort by meter then timestamp (mirrors EDA notebook step)
df = df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)

# ── 10. Save ──────────────────────────────────────────────────────────────────
print(f"Saving to {OUTPUT_CSV} …")
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅  Done!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Meters: {df['meter'].nunique()}")
print(f"   Date range: {df['x_Timestamp'].min()} → {df['x_Timestamp'].max()}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
