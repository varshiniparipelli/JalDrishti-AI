
# JalDrishti-AI 💧

**Live Demo:** [Click here to use JalDrishti-AI](https://jaldrishti-ai-4.onrender.com)



**Problem Statement #38 — Improved Source of Drinking Water**
*AI-powered analytics platform for MIS 78th Round data, built on IBM Cloud Lite services*

> Dataset: [AI Kosh — MIS 78th Round](https://aikosh.indiaai.gov.in/web/datasets/details/improved_source_of_drinking_water_multiple_indicator_survey_78th_round.html)

---

## Overview

JalDrishti-AI (जल दृष्टि — "water vision") analyses data from India's **Multiple Indicator Survey (MIS) 78th Round (NSS 78th Round)** to assess the percentage of the population with access to improved drinking water sources. It also explores **piped-water connectivity** and **rural–urban disparities**, providing actionable policy insights toward **SDG 6.1** (universal safe water by 2030).

| Layer | Technology |
|---|---|
| AI / LLM | IBM watsonx.ai — Granite 4 H Small |
| ML Engine | scikit-learn GradientBoostingRegressor |
| Storage | IBM Cloud Object Storage (Lite) |
| Database | IBM Db2 on Cloud (Lite) |
| Chatbot | IBM Watson Assistant (Lite) — fallback |
| Remote ML | IBM Watson Machine Learning (Lite) |
| Backend | Python 3.11 + Flask |
| Frontend | HTML5 + CSS3 + Plotly.js |

---

## Project Structure

```
JalDrishti-AI/
├── .env                         # IBM Cloud credentials (never commit!)
├── .env.example                 # Template — copy to .env and fill in
├── app.py                       # Flask REST API (entry point)
├── analysis.py                  # Data analysis + scikit-learn ML engine
├── requirements.txt             # Python dependencies
├── Procfile                     # Render / Heroku deployment config
├── render.yaml                  # Render.com deployment manifest
│
├── data/
│   ├── nss78_21.xlsx            # Real MIS 78th Round dataset (AI Kosh)
│   ├── mis_78th_round_drinking_water.csv  # (legacy generated sample)
│   └── generate_sample_data.py  # Synthetic data generator (for testing)
│
├── models/
│   ├── water_access_model.pkl   # Trained GradientBoostingRegressor
│   └── label_encoders.pkl       # sklearn LabelEncoders
│
├── services/
│   ├── watsonx_service.py       # IBM watsonx.ai (Granite chat + insights)
│   ├── cos_service.py           # IBM Cloud Object Storage
│   ├── db2_service.py           # IBM Db2 on Cloud
│   ├── watson_ml_service.py     # IBM Watson Machine Learning (remote)
│   └── assistant_service.py    # IBM Watson Assistant chatbot
│
├── templates/
│   └── index.html               # Dashboard HTML
│
└── static/
    ├── css/style.css            # Dashboard styles
    └── js/dashboard.js         # Plotly charts + API calls
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/varshiniparipelli/JalDrishti-AI.git
cd JalDrishti-AI
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials

Copy `.env.example` to `.env` and fill in your IBM Cloud Lite credentials:

```bash
cp .env.example .env
```

| Variable | Where to find it |
|---|---|
| `WATSONX_API_KEY` | IBM Cloud → Manage → API keys |
| `WATSONX_PROJECT_ID` | IBM Watson Studio → Project → Manage |
| `IBM_COS_API_KEY` | IBM Cloud → Cloud Object Storage → Service credentials (HMAC) |
| `IBM_DB2_HOST / USERNAME / PASSWORD` | IBM Cloud → Db2 on Cloud → Service credentials |
| `IBM_ASSISTANT_API_KEY` | IBM Cloud → Watson Assistant → Manage → Credentials |

### 3. Place the real dataset

Download `nss78_21.xlsx` from AI Kosh:
> https://aikosh.indiaai.gov.in/web/datasets/details/improved_source_of_drinking_water_multiple_indicator_survey_78th_round.html

Place it at `data/nss78_21.xlsx`.

### 4. Run the Flask app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 5. Train the ML model (first time)

```bash
curl -X POST http://localhost:5000/api/train
```

Or click **"Train Model & Predict"** in the dashboard.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/kpis` | National KPI summary |
| GET | `/api/state-analysis` | Improved access % by state (Rural / Urban / All) |
| GET | `/api/sector-analysis` | Rural vs Urban vs All-India comparison |
| GET | `/api/water-sources` | Piped vs Improved — All-India by sector |
| GET | `/api/state-gap` | Rural–Urban gap ranking by state |
| GET | `/api/top-states?n=10` | Top N states by improved access |
| GET | `/api/bottom-states?n=10` | Bottom N states by improved access |
| GET | `/api/sdg-progress` | SDG 6.1 projection to 2030 |
| POST | `/api/train` | Train GradientBoosting on real dataset |
| POST | `/api/predict` | Predict improved access % for a state + sector |
| POST | `/api/watson-predict` | Remote prediction via Watson ML deployment |
| POST | `/api/chat/session` | Create Watson Assistant session |
| POST | `/api/chat/message` | Send message to Watson Assistant |
| POST | `/api/cos/upload` | Upload dataset to IBM COS bucket |
| GET | `/api/watsonx/national-insight` | IBM Granite — national policy insight |
| POST | `/api/watsonx/state-insight` | IBM Granite — state-specific insight |
| POST | `/api/watsonx/chat` | IBM Granite chatbot (dataset-grounded) |
| GET | `/health` | Health check |

### Predict request body

```json
{
  "state": "Bihar",
  "sector": "Rural"
}
```

---

## IBM Cloud Lite Services Setup

### watsonx.ai (Granite 4 H Small)
1. Create **Watson Studio** instance on IBM Cloud
2. Create a project → copy `Project ID` → `WATSONX_PROJECT_ID` in `.env`
3. Generate an IBM Cloud **API Key** → `WATSONX_API_KEY` in `.env`

### Watson Machine Learning (remote deployment)
1. Create a **Deployment Space** in Watson Studio
2. Train locally, then deploy the model to WML
3. Copy deployment ID → `MODEL_DEPLOYMENT_ID` in `.env`

### Cloud Object Storage
1. Create **Cloud Object Storage** instance (Lite tier)
2. Create **HMAC service credentials** → copy API key & instance ID
3. POST `/api/cos/upload` to push the dataset to your COS bucket

### Db2 on Cloud
1. Create **Db2 on Cloud** (Lite tier — 200 MB free)
2. Copy the connection credentials to `.env`
3. Run `services/db2_service.py` → `create_table()` and `load_csv_to_db2()`

### Watson Assistant
1. Create **Watson Assistant** (Lite — 10,000 messages/month)
2. Build an assistant with intents: `water_access`, `sdg_info`, `state_data`
3. Copy environment ID and API key to `.env`

---

## Key Findings (MIS 78th Round — Real AI Kosh Data)

| Indicator | Value |
|---|---|
| National improved water access | **95.7%** |
| Rural improved access | **95.0%** |
| Urban improved access | **97.2%** |
| Rural–Urban access gap | **2.2 pp** |
| Piped water into dwelling (All India) | **35.5%** |
| Piped water — Rural | **24.8%** |
| Piped water — Urban | **61.5%** |
| States above 90% improved access | **31 / 36** |
| Highest: Delhi, Chandigarh | **100%** |
| Lowest: Kerala | **55.2%** *(well-reliant, see note)* |

> **Note on Kerala:** Kerala's lower "improved source" rank reflects the MIS 78th Round definition — households primarily relying on private wells (even clean ones) may not meet the WHO/JMP "improved source" classification. This is a measurement artefact, not an indication of poor water quality.

---

## ML Model Performance

| Metric | Value |
|---|---|
| Algorithm | GradientBoostingRegressor |
| Features | State, Sector (label-encoded) |
| Training samples | 72 (Rural + Urban per 36 states/UTs) |
| 5-Fold CV MAE | **4.97 pp** |
| Train MAE | **3.46 pp** |
| Within ±5 pp | **80.6 %** of states |
| Within ±10 pp | **90.3 %** of states |

---

## SDG 6.1 Alignment

> *"By 2030, achieve universal and equitable access to safe and affordable drinking water for all"*

| Milestone | Value |
|---|---|
| 2015 Baseline (est.) | 70.0% |
| Current coverage (MIS 78th Round, ~2021) | 95.7% |
| Annual improvement rate | +3.21 pp/yr |
| Projected 2030 coverage | 100% |
| On track? | **Yes** |

---

## License

MIT License — for academic and policy research use.
