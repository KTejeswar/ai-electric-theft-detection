# ⚡ AI-POWERED ELECTRICITY THEFT DETECTION SYSTEM
### A Hybrid Machine Learning, Geospatial AMI Telemetry & Closed-Loop Field Normalization Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20RF-green.svg)](https://xgboost.readthedocs.io/)
[![ReportLab](https://img.shields.io/badge/PDF-ReportLab%204.5-orange.svg)](https://www.reportlab.com/)
[![State Coverage](https://img.shields.io/badge/Grid-AP%20%26%20Telangana%20Discoms-cyan.svg)](#geospatial-grid-coverage)

---

## 📌 1. PROBLEM STATEMENT, ITS EFFECTS & OUR SOLUTION

### 🔴 The Problem Statement
In power distribution networks, **Non-Technical Losses (NTL)**—primarily caused by **electricity theft, illegal wire tapping, meter bypassing, current-transformer (CT) shunting, and physical seal tampering**—pose an existential threat to electrical utilities. Unlike technical losses (which are inherent heat dissipations in wires and transformers), electricity theft represents unrecorded, unmetered, and unbilled energy consumed maliciously.

Traditional theft mitigation in Indian power distribution companies relies on **sporadic, manual vigilance raids** guided by anonymous tip-offs or static rule-based alarms (e.g., flagging consumption if it drops below a fixed threshold). These methods suffer from:
- A hit rate under **15%**, wasting valuable workforce hours and squad logistics.
- Complete inability to detect intermittent, alternate-hour, or sophisticated partial-shunt tampering.
- Inability to distinguish between legitimate family vacations / seasonal closures and actual theft, leading to consumer harassment and legal disputes.

---

### ⚠️ The Real-World Effects & Impact
1. **Severe Financial Drain on State Discoms:** In India, Aggregate Technical and Commercial (AT&C) losses range between **15% and 25%**, causing cumulative annual revenue losses exceeding **₹20,000+ Crores**. Discoms operate under heavy deficits, restricting capital expenditure for grid modernization.
2. **Grid Overloading & Transformer Failures:** Unmetered power theft causes severe localized overloading on Distribution Transformers (DTRs), leading to frequent equipment burnouts, phase imbalances, and unscheduled power outages for innocent consumers.
3. **Tariff Inflation for Honest Consumers:** To recover non-technical losses, regulatory commissions are forced to approve higher retail tariffs, economically penalizing compliant households, farmers, and commercial establishments.
4. **Legal & Enforcement Friction:** Discom vigilance squads often face evidentiary challenges in special electricity courts due to the lack of tamper-evident, court-admissible electronic proof under Section 65B of the Indian Evidence Act.

---

### 💡 Our Solution: The AI-Powered Closed-Loop Forensic Platform
We have developed an **end-to-end, intelligent smart grid forensic system** that automates the entire lifecycle of power theft detection, legal prosecution, and physical grid normalization:

```
[9.7M Smart Meter Readings] 
         │
         ▼
[17 Time-Series Feature Extractors] 
         │
         ▼
[Hybrid ML Engine: XGBoost + Random Forest + Isolation Forest + LOF] 
         │
         ▼
[30-Feeder GIS Map (AP & Telangana) & Feeder Loss Recalculation] 
         │
         ▼
[Automated PDF Legal Inspection Warrants (Sec 135 & 138 Electricity Act)] 
         │
         ▼
[Field Squad Rectification & On-Site Meter Normalization Hub (Sec 152)] 
         │
         ▼
[Real-Time Status Restoration to Normal (Green) & Clearance Certificate PDF]
```

- **High-Frequency AMI Analytics:** Ingests and processes **9,705,600 raw smart meter readings** (3-minute intervals across 60 consumer profiles over a 365-day annual cycle).
- **Hybrid AI Engine:** Couples **Supervised Ensembles** (XGBoost + Random Forest) with **Unsupervised Anomaly Detectors** (Isolation Forest + Local Outlier Factor) to catch both known tampering tricks and novel zero-day attack vectors.
- **Explainable Time-Series Features (XAI):** Extracts **17 domain-specific indicators** (load factor, coefficient of variation, zero streak hours, day/night ratios) that provide transparent proof for electricity courts rather than uninterpretable black-box predictions.
- **Geospatial SCADA Command Center:** Maps **30 distribution substations** across **Telangana (TSSPDCL, TSNPDCL)** and **Andhra Pradesh (APEPDCL, APCPDCL, APSPDCL)**, calculating real-time power losses (MWh) and financial losses (₹ Lakhs).
- **Automated Statutory Legal Warrants:** Directly generates court-ready **PDF Field Inspection Warrants** citing Sections 135 & 138 of the Indian Electricity Act 2003 with exact GPS coordinates.
- **Closed-Loop Meter Normalization Workflow:** Features an on-site field verification hub where squads log tamper removal, new seal IDs, and compounding penalties (Section 152), instantly restoring the meter to **Normal** status across all SCADA maps, KPIs, and reports.

---

## 🛠️ 2. TECH STACK & TOOLS USED

| Layer | Tools & Libraries | Key Responsibility in the System |
|---|---|---|
| **Programming Language** | `Python 3.10+ / 3.12` | Core computing runtime for the entire pipeline and dashboard |
| **Data Processing** | `pandas`, `numpy` | Vectorized time-series resampling, window aggregation, feature computation |
| **Supervised Machine Learning** | `xgboost`, `scikit-learn` | Non-linear ensemble classification (Random Forest + XGBoost) |
| **Unsupervised Anomaly Detection** | `scikit-learn` | Multi-dimensional boundary isolation (`IsolationForest`) and density shift (`LocalOutlierFactor`) |
| **Model Serialization** | `joblib` | Model persistence, scaler states, and decision threshold dictionaries |
| **Frontend & Operational UI** | `streamlit` | Reactive SCADA dashboard, session state lifecycle, custom CSS container architecture |
| **Data Visualization & GIS Maps** | `plotly`, `plotly.express` | Interactive OpenStreetMap vector rendering, 365-day telemetry charts, polar radars |
| **Automated Legal PDF Engine** | `reportlab 4.5` | In-memory compilation of official Field Warrants and Restoration Certificates |
| **Presentation Deck Generator** | `python-pptx` | Programmatic compilation of the 12-slide executive presentation (`.pptx`) |

---

## 📁 3. PROJECT DIRECTORY STRUCTURE

```
ElectricityTheftDetection/
│
├── app/
│   └── streamlit_app.py               # Operational SCADA Dashboard (Map, Alerts, Normalization, Proof)
│
├── data/
│   ├── raw/
│   │   └── data.csv                   # 9,705,600 raw 3-min smart meter readings (60 meters x 365 days)
│   └── processed/
│       ├── clean_resampled.csv        # 485,400 hourly consumption readings
│       ├── simulated_theft_data.csv   # 6 mathematically modeled physical theft patterns
│       ├── window_features.csv        # 21,900 daily windows with 17 domain-engineered features
│       └── window_features_clean.csv  # Non-zero baseline feature dataset
│
├── models/
│   ├── rf_model.pkl                   # Trained Random Forest Classifier
│   ├── xgb_model.pkl                  # Trained XGBoost Classifier
│   ├── isolation_forest.pkl           # Fitted Isolation Forest Anomaly Detector
│   ├── lof_model.pkl                  # Fitted Local Outlier Factor Anomaly Detector
│   ├── scaler_anomaly.pkl             # Fitted StandardScaler for outlier features
│   └── thresholds.pkl                 # Tuned operational decision thresholds
│
├── run_pipeline.py                    # End-to-end automated pipeline (Ingestion -> Feature Extraction -> Training)
├── Electricity_Theft_Detection_Major_Project.pptx # Automated 12-Slide Executive Project Presentation
├── PROJECT_REPORT.md                  # Comprehensive 500+ line technical report (A-to-Z Project Binder)
├── requirements.txt                   # Project Python dependency manifest
└── README.md                          # Project documentation and quickstart guide
```

---

## ⚙️ 4. HOW TO RUN THE PROJECT (STEP-BY-STEP)

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/ElectricityTheftDetection.git
cd ElectricityTheftDetection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `streamlit`, `plotly`, `reportlab`, `python-pptx`)*

### Step 3: Run the End-to-End Pipeline (Optional if models are already trained)
To regenerate features, simulate physical attacks, and train all models:
```bash
python run_pipeline.py
```
*Outputs generated in `data/processed/` and serialized models in `models/`.*

### Step 4: Launch the Interactive SCADA Dashboard
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🔬 5. MATHEMATICAL THEFT ATTACK SIMULATION

Because real-world utilities cannot disclose labeled customer theft records due to privacy regulations, the system synthesizes **six physical tampering vectors** based on IEEE smart grid literature:

Let $x(t)$ denote genuine consumption, and $y(t)$ denote the recorded smart meter telemetry:

| Attack Model | Mathematical Formulation | Physical Tampering Mechanism |
|---|---|---|
| **1. Direct Bypass** | $y(t) = \alpha \cdot x(t), \; \alpha \in [0.0, 0.05]$ | Neutral bypass tap or current coil grounding; records near-zero power. |
| **2. Load Suppression** | $y(t) = \alpha \cdot x(t), \; \alpha \in [0.3, 0.5]$ | Shunt resistor across CT secondary terminals; registers only 30% to 50% load. |
| **3. Sudden Step-Down Drop** | $y(t) = 0.20 \cdot x(t) \text{ for } t \ge t_{\text{start}}$ | Permanent physical mechanical retardation or tap installation from a specific date. |
| **4. Alternate-Hour Theft** | $y(t) = x(t) \cdot [1 - \beta \sin^2(\pi t / 12)]$ | Bypass engaged during high-tariff / nighttime hours, returning to normal by day. |
| **5. Flatline Clamping** | $y(t) = 0.20 \cdot \mu_x + \epsilon(t)$ | Readout artificially clamped to a static baseline, masking heavy appliance usage. |
| **6. Erratic Swings** | $y(t) = x(t) \cdot \text{Uniform}(0.1, 1.2)$ | Loose, intermittent neutral connection generating extreme Coefficient of Variation ($CV \ge 4.0$). |

---

## 🎯 6. HYBRID DETECTION ENGINE & SCORING FORMULA

Each daily 24-hour window is evaluated by computing a composite **5-Point Hybrid Confidence Score ($S \in [0, 5]$)**:

$$S = 0$$
$$\text{If } P_{\text{Ensemble}} \ge 0.60 \implies S \leftarrow S + 2, \quad \text{else if } P_{\text{Ensemble}} \ge 0.40 \implies S \leftarrow S + 1$$
$$\text{If } \text{Iso}_{\text{flag}} = 1 \text{ and } \text{LOF}_{\text{flag}} = 1 \implies S \leftarrow S + 2, \quad \text{else if either} \implies S \leftarrow S + 1$$
$$\text{If } \text{zero\_pct} \ge 0.50 \implies S \leftarrow S + 1$$
$$\text{If } (\text{max\_low\_streak} \ge 16 \text{ or } CV \ge 4.0) \implies S \leftarrow S + 1$$
$$S_{\text{final}} = \min(S, 5)$$

### Multi-Tier Long-Term Aggregation (Prevents False Raid Dispatches)
Evaluating 365 days prevents single-day holidays from triggering false alarms:
- **$\mathbf{Theft\ Detected}$:** Theft days ratio $\ge 5\%$ ($\ge 18$ tampering days across the year).
- **$\mathbf{Suspicious}$:** Theft ratio $\ge 1.5\%$ (5–17 days) or intermittent fluctuations $\ge 30\%$.
- **$\mathbf{Normal}$:** Standard authentic consumer usage (0 to 4 isolated outlier days).

---

## 🗺️ 7. GEOSPATIAL GRID COVERAGE (30 DISTRIBUTION FEEDERS)

The platform models **30 real-world distribution corridors** across Andhra Pradesh and Telangana:

```
TELANGANA (TSSPDCL & TSNPDCL):
├── Hyderabad Metro: Gachibowli, Banjara Hills, Charminar, Secunderabad Station
├── Industrial Corridors: Patancheru Heavy Industrial (Sangareddy), Medchal Outer Ring, Jadcherla Park
├── Agro & Commercial: Miryalaguda Rice Hub (Nalgonda), Warangal Subedari, Karimnagar, Nizamabad, Khammam, Siddipet
└── Power & Mining: Ramagundam NTPC Power Belt, Mancherial Coal Corridor

ANDHRA PRADESH (APEPDCL, APCPDCL, APSPDCL):
├── Coastal & Ports: Visakhapatnam MVP Colony, Gajuwaka Steel Feeder, Kakinada Port Hub
├── North Coastal: Rajahmundry Morampudi Commercial, Srikakulam Palasa Agro, Vizianagaram Fort
├── Delta & Agriculture: Vijayawada Benz Circle, Guntur Arundelpet, Bhimavaram Aquaculture Delta
└── Rayalaseema & Industrial: Ongole Granite Mining, Tirupati Alipiri, Kurnool, Nellore Krishnapatnam, Anantapur, Kadapa
```

---

## ⚖️ 8. STATUTORY COMPLIANCE & LEGAL FRAMEWORK

The platform embeds legal compliance directly into its operational workflows:
- **Section 135, Indian Electricity Act 2003:** Issues formal inspection warrants empowering APTS squads to inspect, seize evidence, and register FIRs.
- **Section 138, Indian Electricity Act 2003:** Covers interference with metering infrastructure and seal tampering.
- **Section 152, Indian Electricity Act 2003:** Governs civil compounding penalties, enabling squads to accept settlements and restore regular service.
- **Section 65B, Indian Evidence Act:** Formats digital AMI smart meter logs with cryptographic timestamps and GPS coordinates for secondary evidence admissibility in court.

---

## 📊 9. EXPERIMENTAL RESULTS & BENCHMARKS

Evaluated on 21,900 daily windows with a realistic 6.7% theft ground truth:

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest (Supervised) | 91.4% | 88.2% | 89.8% | 0.946 | 3.2% |
| XGBoost Classifier (Supervised) | 93.1% | 89.6% | 91.3% | 0.958 | 2.9% |
| Isolation Forest (Unsupervised) | 76.5% | 84.1% | 80.1% | 0.872 | 7.8% |
| Local Outlier Factor (Unsupervised) | 74.2% | 81.5% | 77.7% | 0.849 | 8.4% |
| **Hybrid Ensemble (Our Platform)** | **94.8%** | **92.4%** | **93.6%** | **0.974** | **2.4%** |

- **Sub-15ms Latency:** In-memory caching (`@st.cache_data`) enables sub-second tab transitions and filter switches.
- **High Recall (92.4%):** Captures severe neutral bypasses as well as subtle partial-load tampering.
- **Low False Alarm Rate (2.4%):** Protects legitimate consumers from harassment during vacations or temporary grid outages.

---

## 👥 10. PROJECT CONTRIBUTORS & METADATA

- **Project:** Final Year Engineering Major Project
- **Domain:** Artificial Intelligence, Smart Power Grids & Utility Forensics
- **Target Utilities:** TSSPDCL, TSNPDCL, APEPDCL, APCPDCL, APSPDCL
- **License:** MIT License
