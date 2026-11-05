# JalDrishti-AI 💧

**Problem Statement #38 — Improved Source of Drinking Water**
*AI-powered analytics platform for MIS 78th Round data, built on IBM Cloud Lite services*

---

## Overview

JalDrishti-AI (जल दृष्टि — "water vision") analyses data from India's **Multiple Indicator Survey (MIS) 78th Round** to assess the percentage of the population with access to improved drinking water sources. It also explores **clean cooking fuel** use and **migration** trends, providing actionable policy insights toward **SDG 6.1** (universal safe water by 2030).

| Layer | Technology |
|---|---|
| AI / ML | IBM Watson Machine Learning (Lite) |
| Storage | IBM Cloud Object Storage (Lite) |
| Database | IBM Db2 on Cloud (Lite) |
| Chatbot | IBM Watson Assistant (Lite) |
| Backend | Python 3.11 + Flask |
| Frontend | HTML5 + CSS3 + Plotly.js |

---

## Project Structure

```
JalDrishti-AI/
├── .env                         # IBM Cloud credentials (never commit!)
├── app.py                       # Flask REST API (entry point)
├── analysis.py                  # Data analysis + scikit-learn ML engine
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── generate_sample_data.py  # Generate representative MIS-78 CSV
│   └── mis_78th_round_drinking_water.csv  (generated)
│
├── models/
│   ├── water_access_model.pkl   # Trained RandomForest (generated)
│   └── label_encoders.pkl       # sklearn LabelEncoders (generated)
│
├── services/
│   ├── cos_service.py           # IBM Cloud Object Storage
│   ├── db2_service.py           # IBM Db2 on Cloud
│   ├── watson_ml_service.py     # IBM Watson Machine Learning
│   └── assistant_service.py    # IBM Watson Assistant chatbot
│
├── templates/
│   └── index.html               # Dashboard HTML
│
├── static/
│   ├── css/style.css            # Dashboard styles
│   └── js/dashboard.js         # Plotly charts + API calls
│
└── notebooks/                   # (Jupyter notebooks — optional)
```

---

## Quick Start

### 1. Clone & set up environment

```bash
cd JalDrishti-AI
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials

Edit `.env` and replace every `your_*_here` placeholder with your actual IBM Cloud Lite service credentials:

| Variable | Where to find it |
|---|---|
| `IBM_WATSON_ML_API_KEY` | IBM Cloud → Watson Studio → Manage → API Key |
| `IBM_COS_API_KEY` | IBM Cloud → Cloud Object Storage → Service credentials |
| `IBM_DB2_HOST / USERNAME / PASSWORD` | IBM Cloud → Db2 on Cloud → Service credentials |
| `IBM_ASSISTANT_API_KEY` | IBM Cloud → Watson Assistant → Manage → Credentials |

### 3. Generate the sample dataset

```bash
python data/generate_sample_data.py
```
This creates `data/mis_78th_round_drinking_water.csv` (5,000 representative rows).

> **Note:** Replace with the real AI Kosh dataset from:
> https://aikosh.indiaai.gov.in/web/datasets/details/improved_source_of_drinking_water_multiple_indicator_survey_78th_round.html

### 4. Run the Flask app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/kpis` | National KPI summary |
| GET | `/api/state-analysis` | Improved access % by state |
| GET | `/api/sector-analysis` | Rural vs Urban comparison |
| GET | `/api/social-group-analysis` | Access by social group (ST/SC/OBC/General) |
| GET | `/api/water-sources` | Water source distribution |
| GET | `/api/sdg-progress` | SDG 6.1 progress & projection |
| POST | `/api/train` | Train RandomForest on local data |
| POST | `/api/predict` | Predict access for a household record |
| POST | `/api/watson-predict` | Remote prediction via Watson ML deployment |
| POST | `/api/chat/session` | Create Watson Assistant session |
| POST | `/api/chat/message` | Send message to Watson Assistant |
| POST | `/api/cos/upload` | Upload dataset to IBM COS bucket |
| GET | `/health` | Health check |

### Predict request body

```json
{
  "sector": "Rural",
  "social_group": "ST",
  "religion": "Hindu",
  "household_size": 5,
  "water_collection_time_min": 45,
  "has_clean_fuel": 0,
  "migration_last_year": 1
}
```

---

## IBM Cloud Lite Services Setup

### Watson Machine Learning
1. Create **Watson Studio** instance (Lite tier) on IBM Cloud
2. Create a **Deployment Space**
3. Copy `Space ID` → `IBM_WATSON_ML_SPACE_ID` in `.env`
4. After training locally, deploy the model to WML and copy the deployment ID → `MODEL_DEPLOYMENT_ID`

### Cloud Object Storage
1. Create **Cloud Object Storage** instance (Lite tier)
2. Create **HMAC service credentials** → copy API key & instance ID
3. POST `/api/cos/upload` to push the dataset to your COS bucket

### Db2 on Cloud
1. Create **Db2 on Cloud** (Lite tier — 200MB free)
2. Copy the connection credentials to `.env`
3. Run `services/db2_service.py` → `create_table()` and `load_csv_to_db2()`

### Watson Assistant
1. Create **Watson Assistant** (Lite — 10,000 messages/month)
2. Build an assistant with intents: `water_access`, `sdg_info`, `state_data`, `social_group`
3. Copy environment ID and API key to `.env`

---

## Key Findings (MIS 78th Round — Sample Data)

- **Overall improved access**: ~74% nationally
- **Urban** households: ~88% vs **Rural**: ~62% — a 26 pp gap
- **ST households** have the lowest access (~58%) — primary equity concern
- **SC households**: ~64%; OBC: ~74%; General: ~83%
- States with lowest coverage need priority intervention
- Positive correlation between clean cooking fuel access and improved water access
- Current annual improvement rate suggests India is **marginally off-track** for SDG 6.1 by 2030 without accelerated intervention

---

## SDG 6.1 Alignment

> *"By 2030, achieve universal and equitable access to safe and affordable drinking water for all"*

This platform directly supports evidence-based monitoring and policymaking for SDG 6.1 by:
- Disaggregating data by state, sector, social group, and religion
- Projecting 2030 coverage trajectories
- Identifying at-risk populations for targeted intervention
- Providing an AI predictor to flag vulnerable households

---

## License

MIT License — for academic and policy research use.
