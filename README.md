# UIDAI Hackathon Dashboard - Backend Engine

A high-performance Python backend that processes Aadhaar **Enrolment**, **Biometric**, and **Demographic** data to generate actionable operational insights for government administrators.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **MBU Deficit Tracking** | Identifies pincodes where children are enrolled but lack biometric updates |
| **Fraud Detection** | 4-Model Ensemble (Isolation Forest, SVM, LOF, Autoencoder) to flag suspicious adult enrolments |
| **Migration Hotspots** | Detects areas with high in-migration patterns for urban planning |

---

## 📁 Project Structure

```
UIDAI/
├── Data/                          # CSV data (DO NOT COMMIT)
│   ├── api_data_aadhar_enrolment/
│   ├── api_data_aadhar_biometric/
│   └── api_data_aadhar_demographic/
├── generate_insights.py           # Main backend script
├── app_data.json                  # Generated output (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 3. Place Data Files

Place your CSV data files in the following structure inside the `Data/` folder:

| Folder | Pattern | Required Columns |
|--------|---------|------------------|
| `api_data_aadhar_enrolment/` | `*_enrolment_*.csv` | date, state, district, pincode, age_0_5, age_5_17, age_18_greater |
| `api_data_aadhar_biometric/` | `*_biometric_*.csv` | date, state, district, pincode, bio_age_5_17, bio_age_17_ |
| `api_data_aadhar_demographic/` | `*_demographic_*.csv` | date, state, district, pincode, demo_age_5_17, demo_age_17_ |

---

## ▶️ Usage

Run the insight generation script:

```bash
python generate_insights.py
```

**Output:** `app_data.json` containing:
- Summary statistics (total pincodes, critical alerts, MBU backlog)
- Compliance map data (deficit per pincode)
- Action tickets (MBU Camps, Fraud Audits, Urban Surveys)

---

## 📊 Output Format

```json
{
  "summary": {
    "total_pincodes": 1250,
    "critical_alerts": 15,
    "mbu_backlog_total": 50000
  },
  "compliance_map_data": [...],
  "action_tickets": [...]
}
```

---

## 🛠️ Tech Stack

- **Data Processing:** Pandas, NumPy
- **ML/Anomaly Detection:** Scikit-learn, TensorFlow/Keras

---

## 📄 License

UIDAI Hackathon Project - 2026
