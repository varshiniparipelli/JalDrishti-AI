
"""
JalDrishti-AI - IBM watsonx.ai Service
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# IBM WATSONX.AI CONFIGURATION
# ============================================================

WATSONX_PROJECT_ID = os.getenv(
    "WATSONX_PROJECT_ID",
    "d35a6f68-c129-4634-b823-ff9583f7fde6"
)

WATSONX_URL = os.getenv(
    "WATSONX_URL",
    "https://us-south.ml.cloud.ibm.com"
).rstrip("/")

WATSONX_API_KEY = (
    os.getenv("WATSONX_API_KEY")
    or os.getenv("IBM_WATSON_ML_API_KEY")
)

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
DEFAULT_MODEL = "ibm/granite-4-h-small"
WATSONX_API_VER = "2024-05-01"


# ============================================================
# HTTP SESSION
# ============================================================

def _get_session():
    session = requests.Session()
    session.trust_env = False
    return session


# ============================================================
# IBM IAM ACCESS TOKEN
# ============================================================

def _get_iam_token() -> str:

    if not WATSONX_API_KEY:
        raise ValueError(
            "WATSONX_API_KEY is not configured in environment variables."
        )

    response = _get_session().post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        timeout=30
    )

    response.raise_for_status()

    token = response.json().get("access_token")

    if not token:
        raise ValueError("IBM IAM did not return an access token.")

    return token


# ============================================================
# CLEAN GRANITE RESPONSE
# ============================================================

def _clean_response(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.strip()

    markers = [
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "assistant:"
    ]

    for marker in markers:
        text = text.replace(marker, "")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    suspicious = [
        "now answer the user's question",
        "================ verified project data",
        "important answering rules",
        "user question:"
    ]

    if any(item in text.lower() for item in suspicious):
        return ""

    return text


# ============================================================
# IBM GRANITE CHAT API
# ============================================================

def _call_granite(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300
) -> str:

    token = _get_iam_token()

    endpoint = (
        f"{WATSONX_URL}/ml/v1/text/chat"
        f"?version={WATSONX_API_VER}"
    )

    payload = {
        "model_id": DEFAULT_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.2,
            "repetition_penalty": 1.05
        }
    }

    response = _get_session().post(
        endpoint,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        timeout=60
    )

    response.raise_for_status()
    result = response.json()

    try:
        content = result["choices"][0]["message"]["content"]

        if isinstance(content, list):
            content = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        generated_text = _clean_response(content)

    except (KeyError, IndexError, TypeError):
        raise ValueError(
            "IBM Granite returned an unexpected response format."
        )

    if not generated_text:
        raise ValueError(
            "IBM Granite returned an empty or invalid response."
        )

    return generated_text


# ============================================================
# GENERAL TEXT GENERATION
# ============================================================

def generate_insight(prompt: str, max_tokens: int = 300) -> str:

    system_prompt = """
You are JalDrishti AI, specializing in drinking-water access,
water conservation, and SDG 6.1.

Use supplied statistics accurately. Never invent survey values,
state rankings, project features, or citations.

Use clear and concise English.
"""

    return _call_granite(system_prompt, prompt, max_tokens)


# ============================================================
# STATE-WISE INSIGHT
# ============================================================

def analyse_state(
    state: str,
    rural_pct: float,
    urban_pct: float,
    piped_pct: float
) -> str:

    prompt = f"""
Prepare a concise policy recommendation for {state}.

MIS 78th Round statistics:
Rural improved drinking-water access: {rural_pct}%
Urban improved drinking-water access: {urban_pct}%
Piped-water access: {piped_pct}%

Explain the rural and urban situation and suggest practical
interventions related to SDG 6.1.

Improved drinking-water access and piped-water access are
different indicators. Do not treat them as identical.
Do not invent statistics or citations.
"""

    return generate_insight(prompt, max_tokens=200)


# ============================================================
# NATIONAL INSIGHT
# ============================================================

def analyse_national(
    national_pct: float,
    gap_pp: float,
    projected_2030: float
) -> str:

    prompt = f"""
Prepare a concise national water-access policy insight.

National improved drinking-water access: {national_pct}%
Rural-urban access gap: {gap_pp} percentage points
Projected 2030 coverage: {projected_2030}%
SDG 6.1 target: 100% by 2030

Explain India's progress, the rural-urban gap, and practical
interventions. Identify the 2030 figure as a projection,
not an observed survey result. Do not invent statistics.
"""

    return generate_insight(prompt, max_tokens=200)


# ============================================================
# VERIFIED PROJECT ANSWERS
# ============================================================

def _project_features_answer():
    return (
        "The prediction model uses two input features: state and "
        "sector (rural or urban). Both are converted into numerical "
        "values using LabelEncoder before being supplied to the model."
    )


def _unused_features_answer():
    return (
        "No. The current JalDrishti-AI prediction model does not use "
        "rainfall or income as input features. It uses only state "
        "and sector to predict improved drinking-water access."
    )


def _actual_vs_predicted_answer():
    return (
        "Actual values are the drinking-water access percentages "
        "recorded in the MIS 78th Round dataset. Predicted values "
        "are estimates generated by the trained GradientBoostingRegressor "
        "for a selected state and sector. A prediction is a model estimate, "
        "not a new survey observation. The model's Mean Absolute Error "
        "(MAE) measures the average absolute difference between predictions "
        "and actual values used for evaluation. Dashboard survey statistics "
        "remain dataset values; they are not all AI-generated."
    )


def _project_model_answer():
    return (
        "JalDrishti-AI uses GradientBoostingRegressor from scikit-learn. "
        "It builds decision trees sequentially, with each new tree "
        "attempting to reduce the errors made by the existing model. "
        "The configuration is 50 estimators, maximum tree depth 2, "
        "learning rate 0.05, and random state 42. The model uses state "
        "and sector as inputs and predicts improved drinking-water "
        "access percentage. It is evaluated using 5-Fold cross-validation "
        "and Mean Absolute Error (MAE)."
    )


def _cross_validation_answer():
    return (
        "The project uses 5-Fold K-Fold cross-validation. The training "
        "data is divided into five parts. The model trains on four parts "
        "and validates on the remaining part, repeating this process "
        "five times. This helps evaluate its performance on unseen records."
    )


def _mae_answer():
    return (
        "The project evaluates prediction errors using Mean Absolute "
        "Error (MAE). MAE calculates the average absolute difference "
        "between actual and predicted values. A lower MAE indicates "
        "predictions are closer to actual values. The verified evaluation "
        "metric is MAE, not RMSE."
    )


def _project_architecture_answer():
    return (
        "JalDrishti-AI is a Flask-based drinking-water analytics "
        "application. Python and pandas process the MIS 78th Round "
        "Excel dataset, while the dashboard presents water-access "
        "statistics and visualizations. A scikit-learn "
        "GradientBoostingRegressor provides predictions, and IBM "
        "watsonx.ai with the Granite model powers the AI chatbot. "
        "Flask connects these components through backend API endpoints."
    )


def _dataset_answer():
    return (
        "JalDrishti-AI uses the MIS 78th Round dataset stored at "
        "data/nss78_21.xlsx. It contains fields such as state, "
        "age_group, sector, gender, indicator, sub_indicator, value, "
        "and total_count. The project uses these records to analyse "
        "improved drinking-water access, piped-water access, and "
        "rural-urban differences. The dataset also supplies training "
        "records for the prediction model."
    )


def _problem_statement_answer():
    return (
        "Problem Statement #38 asks us to analyse data from the MIS 78th "
        "Round to assess the percentage of India's population with access "
        "to improved drinking water sources, and to explore related "
        "indicators such as piped-water access and rural-urban disparities. "
        "The goal is to generate actionable insights for evidence-based "
        "policymaking and support India's progress toward SDG 6.1 — "
        "universal safe water access by 2030."
    )


def _national_kpi_answer():
    return (
        "According to the MIS 78th Round (nss78_21.xlsx), India's national "
        "improved drinking-water access is 95.7%. Rural access is 95.0% and "
        "urban access is 97.2%, with a rural-urban gap of 2.2 percentage "
        "points. Piped water into dwelling or yard/plot coverage is 35.5% "
        "nationally — 24.8% rural and 61.5% urban. 31 out of 36 states and "
        "UTs exceed the 90% improved-access threshold."
    )


def _sdg_answer():
    return (
        "SDG 6.1 aims for universal and equitable access to safe and "
        "affordable drinking water for all by 2030. According to MIS 78th "
        "Round data, India's current coverage is 95.7%. At the observed "
        "annual improvement rate of 3.21 percentage points per year — "
        "measured from the 2015 baseline of 70% — the projected 2030 "
        "coverage is 100%, indicating India is on track for SDG 6.1. "
        "However, piped-water access (35.5%) and state-level disparities "
        "remain challenges requiring targeted intervention."
    )


def _ibm_services_answer():
    return (
        "JalDrishti-AI uses four IBM Cloud Lite services: "
        "(1) IBM watsonx.ai with Granite 4 H Small for the AI chatbot and "
        "policy insights; "
        "(2) IBM Watson Machine Learning for remote model deployment and "
        "prediction via the /api/watson-predict endpoint; "
        "(3) IBM Cloud Object Storage for storing the MIS 78th Round dataset "
        "in a secure cloud bucket; "
        "(4) IBM Db2 on Cloud for relational database storage of survey "
        "records. IBM Watson Assistant provides a chatbot fallback layer."
    )


def _piped_water_answer():
    return (
        "Piped water into dwelling or yard/plot and improved source of "
        "drinking water are two separate indicators in MIS 78th Round. "
        "Improved source (95.7% nationally) includes tube wells, boreholes, "
        "protected dug wells, and piped water — any source meeting WHO/JMP "
        "safe-water criteria. Piped-water coverage (35.5% nationally) is a "
        "subset: only households with water piped directly to the home or "
        "yard. Rural piped coverage is just 24.8%, while urban is 61.5%, "
        "showing significant infrastructure disparity."
    )


def _kerala_answer():
    return (
        "Kerala has the lowest overall improved drinking-water access score "
        "(55.2%) in the MIS 78th Round dataset. This is a measurement "
        "artefact: Kerala households largely rely on private dug wells, "
        "which may not meet the WHO/JMP 'improved source' classification "
        "even if the water is clean. It does not indicate poor water quality "
        "— Kerala has very high overall water availability. The state's "
        "rural access is 54.4% and urban is 55.9%."
    )


# ============================================================
# QUESTION-SPECIFIC ANSWER ROUTER
# ============================================================

def _get_verified_answer(user_question: str):

    q = re.sub(r"\s+", " ", user_question.lower()).strip()

    # Actual dataset values vs predictions
    actual_prediction_terms = [
        "actual vs predicted",
        "actual and predicted",
        "actual values and predicted",
        "actual dataset values",
        "predicted values",
        "ai predicted",
        "ai-predicted",
        "prediction compared",
        "difference between actual",
        "survey values and predictions"
    ]

    if any(term in q for term in actual_prediction_terms):
        return _actual_vs_predicted_answer()

    # IMPORTANT: Check specific excluded features FIRST.
    unused_feature_terms = [
        "rainfall",
        "income",
        "literacy",
        "ration card",
        "ration-card",
        "governance score",
        "infrastructure budget"
    ]

    if (
        any(term in q for term in unused_feature_terms)
        and any(
            word in q
            for word in [
                "model",
                "input",
                "feature",
                "use",
                "predict"
            ]
        )
    ):
        return _unused_features_answer()

    # General model input features
    feature_question = (
        ("feature" in q or "input" in q)
        and any(
            word in q
            for word in [
                "model",
                "prediction",
                "predictor",
                "algorithm"
            ]
        )
    )

    if feature_question:
        return _project_features_answer()

    # Cross-validation
    if any(term in q for term in [
        "cross validation",
        "cross-validation",
        "5-fold",
        "five-fold",
        "k-fold",
        "kfold"
    ]):
        return _cross_validation_answer()

    # Evaluation metrics
    if (
        re.search(r"\bmae\b", q)
        or "mean absolute error" in q
        or re.search(r"\brmse\b", q)
        or "root mean squared error" in q
        or "evaluation metric" in q
        or "error metric" in q
    ):
        return _mae_answer()

    # Gradient Boosting / model details
    model_terms = [
        "gradient boosting",
        "gradientboosting",
        "regressor",
        "machine learning model",
        "prediction model",
        "model configuration",
        "how does the model work",
        "how does gradient",
        "what algorithm",
        "which algorithm",
        "algorithm used"
    ]

    if any(term in q for term in model_terms):
        return _project_model_answer()

    # Project architecture
    architecture_terms = [
        "project architecture",
        "system architecture",
        "architecture of jaldrishti",
        "technologies used",
        "technology stack",
        "tech stack",
        "how is jaldrishti built"
    ]

    if any(term in q for term in architecture_terms):
        return _project_architecture_answer()

    # Dataset information
    dataset_terms = [
        "which dataset",
        "dataset used",
        "data source",
        "mis 78th round dataset",
        "what is the mis 78th round",
        "what data does",
        "excel dataset"
    ]

    if any(term in q for term in dataset_terms):
        return _dataset_answer()

    # Problem statement
    if any(term in q for term in [
        "problem statement",
        "problem statement 38",
        "problem statement #38",
        "what is jaldrishti",
        "what does jaldrishti",
        "objective of",
        "goal of jaldrishti"
    ]):
        return _problem_statement_answer()

    # National KPIs / statistics
    if any(term in q for term in [
        "national improved",
        "national access",
        "all india",
        "national percentage",
        "overall access",
        "india's water",
        "india water access",
        "national kpi",
        "kpi",
        "95.7",
        "current coverage"
    ]):
        return _national_kpi_answer()

    # SDG 6 / SDG 6.1
    if any(term in q for term in [
        "sdg 6",
        "sdg6",
        "sdg goal 6",
        "sustainable development",
        "2030 target",
        "universal access",
        "on track",
        "projected 2030",
        "sdg progress"
    ]):
        return _sdg_answer()

    # IBM Cloud services
    if any(term in q for term in [
        "ibm cloud",
        "ibm services",
        "ibm watson",
        "ibm cos",
        "ibm db2",
        "watsonx",
        "granite",
        "cloud services",
        "watson ml",
        "object storage"
    ]):
        return _ibm_services_answer()

    # Piped water vs improved source
    if any(term in q for term in [
        "piped water",
        "piped vs",
        "difference between piped",
        "piped access",
        "piped coverage",
        "35.5",
        "24.8",
        "61.5"
    ]):
        return _piped_water_answer()

    # Kerala outlier explanation
    if "kerala" in q:
        return _kerala_answer()

    return None


# ============================================================
# JALDRISHTI-AI CHATBOT
# ============================================================

def chat_response(
    user_question: str,
    context_summary: str = ""
) -> str:

    if not user_question or not user_question.strip():
        return "Please enter a question."

    verified_answer = _get_verified_answer(user_question)

    if verified_answer:
        return verified_answer

    system_prompt = """
You are JalDrishti AI, an intelligent assistant for
India's drinking-water access and MIS 78th Round data.

You are powered by IBM Granite 4 H Small through IBM watsonx.ai.

VERIFIED PROJECT FACTS:

Project context:
- Problem Statement #38: Improved Source of Drinking Water.
- Dataset: MIS 78th Round (nss78_21.xlsx) from AI Kosh.
- 222 data points across 36 states/UTs, 2 sub-indicators.
- The project uses Flask as the backend framework.

National KPIs (real MIS 78th Round data):
- National improved drinking-water access: 95.7%.
- Rural improved access: 95.0%.
- Urban improved access: 97.2%.
- Rural-urban gap: 2.2 percentage points.
- Piped water into dwelling (all India): 35.5%.
- Piped water rural: 24.8%; urban: 61.5%.
- States above 90% improved access: 31 out of 36.
- Top states: Delhi and Chandigarh (100%).
- Lowest state: Kerala (55.2%) — well-reliance artefact.

SDG 6.1 progress:
- 2015 baseline: 70%. Current: 95.7%. Target: 100% by 2030.
- Annual improvement rate: 3.21 pp/yr.
- Projected 2030 coverage: 100%. India is on track.

ML model:
- Algorithm: scikit-learn GradientBoostingRegressor.
- 50 estimators, max_depth=2, learning_rate=0.05, random_state=42.
- Input features: state and sector (LabelEncoder encoded).
- Target: improved drinking-water access percentage.
- Validation: 5-Fold K-Fold cross-validation.
- Evaluation metric: Mean Absolute Error (MAE) = 4.97 pp (5-fold CV).
- Train MAE: 3.46 pp. Within 5 pp: 80.6%. Within 10 pp: 90.3%.
- Saved with joblib.

IBM Cloud Lite services used:
- IBM watsonx.ai (Granite 4 H Small) — AI chatbot and policy insights.
- IBM Watson Machine Learning — remote model deployment (/api/watson-predict).
- IBM Cloud Object Storage — dataset bucket storage (/api/cos/upload).
- IBM Db2 on Cloud — relational DB for survey records.
- IBM Watson Assistant — chatbot fallback.

The model does not use rainfall, income, literacy,
ration-card status, governance scores, or infrastructure budgets.

The model evaluation metric is MAE, not RMSE.

DATA ACCURACY:

1. Use supplied project statistics for state rankings,
   percentages, and water-access indicators.
2. Never invent state rankings or survey percentages.
3. Improved drinking-water access and piped-water access
   are two different indicators — never conflate them.
4. Never invent report quotations, page numbers, citations,
   or references.
5. If a requested statistic is not supplied, say so clearly.

GENERAL KNOWLEDGE:

For general questions about drinking water, pollution,
water conservation, health effects, and SDG 6.1,
provide useful explanations using general knowledge.
Do not claim general knowledge comes from the MIS dataset.

ANSWER STYLE:

- Answer the exact question directly.
- Use simple, clear English.
- Keep answers concise, usually 2 to 5 sentences.
- Avoid repeating the complete model configuration
  unless the user asks for model details.
- Do not invent project functionality.

Return only the final answer.
Do not return JSON, code, tool calls, or fake citations.
"""

    user_prompt = f"""
VERIFIED PROJECT DATA AND STATISTICS:

{context_summary if context_summary else
 "No additional dataset statistics were supplied for this question."}

USER QUESTION:

{user_question}

Answer the question accurately using the system instructions.
"""

    return _call_granite(
        system_prompt,
        user_prompt,
        max_tokens=300
    )
