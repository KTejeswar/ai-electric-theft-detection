

```markdown
# ⚡ Electricity Theft Detection using Machine Learning

## 📌 Overview
Electricity theft is a major issue in power distribution systems, leading to financial losses and grid instability.

This project aims to detect electricity theft using **smart meter data** by combining:
- Behavioral feature engineering
- Simulated theft scenarios
- Supervised machine learning
- Unsupervised anomaly detection
- Ensemble decision-making

---

## 🧠 Problem Statement
Real-world electricity theft data lacks labeled examples of fraudulent behavior.

To address this:
- We simulate realistic theft scenarios using actual consumption data
- Train models to detect abnormal patterns
- Combine multiple models for robust detection

---

## 🔄 Project Pipeline

```

Smart Meter Data (3-min intervals)

↓

Data Aggregation (Daily Consumption)

↓

Data Preprocessing

↓

Feature Engineering (Behavior Patterns)

↓

Simulated Theft Generation

↓

┌───────────────────────┬───────────────────────┐

│                       │
Supervised Models                     Anomaly Detection

(RandomForest, XGBoost)             (Isolation Forest, LOF)

│                       │

└────────── Ensemble Decision ───────────┘

↓

Normal / Suspicious / High Risk

↓

Streamlit Dashboard

```

---

## ⚙️ Key Components

### 🔹 Feature Engineering
Captures electricity consumption behavior:
- Rolling statistics (mean, std)
- Consumption changes
- Z-score normalization
- Low usage detection
- Time-based features

---

### 🔹 Theft Simulation
Since real labels are unavailable, theft is simulated by modifying real data:

- Meter bypass (near-zero usage)
- Load suppression (scaled consumption)
- Flat consumption pattern
- Intermittent theft
- Sudden sustained drop
- Partial manipulation

---

### 🔹 Supervised Learning
Models trained on labeled (simulated) data:
- Random Forest
- XGBoost

---

### 🔹 Anomaly Detection
Detects unknown/unseen fraud patterns:
- Isolation Forest
- Local Outlier Factor (LOF)

---

### 🔹 Ensemble System
Combines outputs from:
- Supervised models
- Anomaly detection models

Final classification:
- ✅ Normal
- ⚠️ Suspicious
- 🚨 High Risk

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall (**most important for fraud detection**)

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  

---

## 📂 Project Structure

```

ElectricityTheftDetection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── simulate_theft.py
│   ├── train_models.py
│   ├── anomaly_models.py
│   ├── ensemble.py
│
├── app/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md

```

---

## 🚀 Future Enhancements

- Real-time data integration
- Deep learning (LSTM for time-series)
- Advanced anomaly detection
- Deployment on cloud

---

## 👨‍💻 Author
Omeir Kazi

---

## 📌 Note
This project uses **simulated theft scenarios** due to the lack of labeled real-world fraud data, ensuring realistic and practical model training.
```

---


