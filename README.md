# ⚡ Electricity Theft Detection

A machine learning system that detects electricity theft from smart meter data.
It combines supervised classification (Random Forest + XGBoost) with unsupervised
anomaly detection (Isolation Forest + LOF) and surfaces results through a
Streamlit dashboard.

---

## How it works

Smart meters record consumption every 3 minutes. Because real-world theft labels
rarely exist, this project **simulates** six theft patterns on real meter data,
trains models on those labelled windows, and uses an ensemble confidence score
to classify each 24-hour window as **Normal**, **Suspicious**, or **High Risk**.

### Theft patterns simulated

| Pattern | Behaviour |
|---|---|
| Meter bypass | Near-zero consumption (~5 % of normal) |
| Load suppression | Scaled down to 30–50 % of actual usage |
| Flat consumption | Constant value replacing real variation |
| Intermittent theft | Alternating hours reduced to 40 % |
| Sudden sustained drop | Consumption scaled to 20 % |
| Partial manipulation | Random combination of the above |

### Confidence scoring

Each 24-hour window receives a score (0–5) built from four independent signals:

| Signal | Points |
|---|---|
| Ensemble model probability ≥ 0.4 | +1 |
| Ensemble model probability ≥ 0.6 | +2 |
| One anomaly model flags the window | +1 |
| Both anomaly models flag the window | +2 |
| `zero_pct` ≥ 0.5 | +1 |
| `max_low_streak` ≥ 16 h **or** `cv` ≥ 4.0 | +1 |

Score 0 → **Normal** · Score 1–2 → **Suspicious** · Score ≥ 3 → **High Risk**

---

## Project structure

```
ElectricityTheftDetection/
├── data/
│   ├── raw/                  # original 3-min interval CSV (not in repo)
│   └── processed/            # intermediate and final feature files
│
├── notebooks/
│   ├── EDA.ipynb             # 1 — load, clean, resample raw data
│   ├── simulation_model.ipynb # 2 — simulate theft, label dataset
│   ├── feature_engineering.ipynb # 3 — extract 17 window features
│   ├── cleaning.ipynb        # 4 — remove all-zero normal windows
│   ├── anomaly_model.ipynb   # 5 — train Isolation Forest + LOF
│   ├── supervised_model.ipynb # 6 — train Random Forest + XGBoost
│   ├── ensemble.ipynb        # 7 — combine models, evaluate
│   └── models/               # saved .pkl files (not in repo)
│       ├── rf_model.pkl
│       ├── xgb_model.pkl
│       ├── isolation_forest.pkl
│       ├── lof_model.pkl
│       ├── scaler_anomaly.pkl
│       └── thresholds.pkl
│
├── app/
│   └── streamlit_app.py      # dashboard
│
├── requirements.txt
└── README.md
```

---

## Running end to end

### 1 — Prerequisites

```bash
python --version   # 3.8 or higher required
pip install -r requirements.txt
```

### 2 — Place raw data

Put your smart meter CSV at `data/raw/data.csv`.
Expected columns: `x_Timestamp`, `t_kWh`, `z_Avg Voltage (Volt)`,
`z_Avg Current (Amp)`, `y_Freq (Hz)`, `meter`.

### 3 — Run notebooks in order

Open each notebook in `notebooks/` and run all cells top to bottom:

| # | Notebook | Output |
|---|---|---|
| 1 | `EDA.ipynb` | `data/processed/clean_resampled.csv` |
| 2 | `simulation_model.ipynb` | `data/processed/simulated_theft_data.csv` |
| 3 | `feature_engineering.ipynb` | `data/processed/window_features.csv` |
| 4 | `cleaning.ipynb` | `data/processed/window_features_clean.csv` |
| 5 | `anomaly_model.ipynb` | `models/isolation_forest.pkl`, `models/lof_model.pkl`, `models/scaler_anomaly.pkl` |
| 6 | `supervised_model.ipynb` | `models/rf_model.pkl`, `models/xgb_model.pkl`, `models/thresholds.pkl` |
| 7 | `ensemble.ipynb` | validation metrics printed inline |

### 4 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` and upload `data/processed/window_features.csv`.

---

## Evaluation results

Measured on `window_features.csv` (7 299 windows, 11.4 % theft).

| Metric | Value |
|---|---|
| Recall (theft) | **95 %** |
| HIGH RISK precision | **~80 %** |
| Missed theft windows | < 5 % |

The system prioritises recall — minimising missed theft cases — while the
High Risk tier is kept precise enough for immediate field action.

---

## Tech stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy |
| Models | scikit-learn, xgboost, joblib |
| Dashboard | streamlit, plotly |
| EDA | matplotlib, seaborn |

Python 3.8+

---

## Notes

- All model `.pkl` files and data CSVs are excluded from version control (see `.gitignore`).
- Re-running `simulation_model.ipynb` regenerates theft labels with `random_state=42`,
  so results are reproducible but will differ if the raw data changes.
- To adjust detection sensitivity, change the score thresholds in `ensemble.ipynb`
  and `streamlit_app.py`: `score >= 3` for High Risk, `score >= 1` for Suspicious.

---

## Author

Omeir Kazi