
"""
JalDrishti-AI — IBM watsonx.ai Service

Provides:
1. IBM Granite 4 H Small integration
2. National water-access insights
3. State-wise water-access insights
4. Dataset-grounded chatbot responses
5. Verified project-specific answers
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
    """
    Create an HTTP session.

    trust_env=False prevents inherited proxy settings from
    interfering with requests to IBM Cloud.
    """

    session = requests.Session()
    session.trust_env = False

    return session


# ============================================================
# GET IBM IAM ACCESS TOKEN
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
        raise ValueError(
            "IBM IAM did not return an access token."
        )

    return token


# ============================================================
# CLEAN GRANITE RESPONSE
# ============================================================

def _clean_response(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove common model conversation markers.
    unwanted_markers = [
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "assistant:"
    ]

    for marker in unwanted_markers:
        text = text.replace(marker, "")

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    ).strip()

    # Reject obvious prompt leakage.
    suspicious_phrases = [
        "now answer the user's question",
        "================ VERIFIED PROJECT DATA",
        "important answering rules",
        "user question:"
    ]

    lowered = text.lower()

    if any(phrase in lowered for phrase in suspicious_phrases):
        return ""

    return text


# ============================================================
# GENERATE RESPONSE USING IBM GRANITE CHAT API
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
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.2,
            "repetition_penalty": 1.05
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = _get_session().post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    try:
        content = result["choices"][0]["message"]["content"]

        # Support both string and list-based message content.
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

def generate_insight(
    prompt: str,
    max_tokens: int = 300
) -> str:

    system_prompt = """
You are JalDrishti AI, an assistant specializing in
drinking-water access, water conservation, and SDG 6.1.

Follow the user's supplied instructions and statistics.

Never invent survey values, state rankings, citations,
project features, model configurations, or evaluation results.

Use clear, concise English.
"""

    return _call_granite(
        system_prompt,
        prompt,
        max_tokens
    )


# ============================================================
# STATE-WISE WATER ACCESS INSIGHT
# ============================================================

def analyse_state(
    state: str,
    rural_pct: float,
    urban_pct: float,
    piped_pct: float
) -> str:

    prompt = f"""
Prepare a concise policy recommendation for {state}.

Use only these supplied MIS 78th Round statistics:

State: {state}
Rural improved drinking-water access: {rural_pct}%
Urban improved drinking-water access: {urban_pct}%
Piped-water access: {piped_pct}%

Write approximately three sentences.

Explain the rural and urban water-access situation.
Suggest practical interventions related to SDG 6.1.

IMPORTANT:
Improved drinking-water access and piped-water access
are different indicators. Never treat them as identical.

Do not invent statistics, rankings, or citations.
"""

    return generate_insight(
        prompt,
        max_tokens=200
    )


# ============================================================
# NATIONAL WATER ACCESS INSIGHT
# ============================================================

def analyse_national(
    national_pct: float,
    gap_pp: float,
    projected_2030: float
) -> str:

    prompt = f"""
Prepare a concise national water-access policy insight.

Use only these supplied project statistics:

National improved drinking-water access: {national_pct}%
Rural-urban access gap: {gap_pp} percentage points
Projected 2030 coverage: {projected_2030}%
SDG 6.1 target: 100% by 2030

Explain:
1. India's progress toward SDG 6.1.
2. The importance of reducing the rural-urban gap.
3. Practical interventions to improve drinking-water access.

Write approximately three sentences.

Clearly identify the 2030 figure as a project projection,
not an observed survey result.

Do not invent additional statistics or citations.
"""

    return generate_insight(
        prompt,
        max_tokens=200
    )


# ============================================================
# VERIFIED JALDRISHTI-AI PROJECT ANSWERS
# ============================================================

def _project_model_answer() -> str:

    return """
JalDrishti-AI uses the GradientBoostingRegressor algorithm
from scikit-learn to predict improved drinking-water access
percentages.

The actual model configuration in analysis.py is:

- Algorithm: GradientBoostingRegressor
- Number of estimators: 50
- Maximum tree depth: 2
- Learning rate: 0.05
- Random state: 42
- Validation: 5-Fold K-Fold cross-validation
- Evaluation metric: Mean Absolute Error (MAE)

The model uses two input features: state and sector.
Both are converted into numerical values using LabelEncoder.

The target variable is the improved drinking-water
access percentage from the MIS 78th Round dataset.

Gradient Boosting builds decision trees sequentially.
Each new tree attempts to reduce the remaining prediction
errors of the existing ensemble.

The model is evaluated using 5-Fold cross-validation
and then fitted on the available training records.
The trained model and label encoders are saved using joblib.

IMPORTANT:
The current implementation does not use rainfall, income,
literacy, ration-card status, governance scores, or
infrastructure budgets as model input features.
It evaluates predictions using MAE, not RMSE.
"""


def _project_architecture_answer() -> str:

    return """
JalDrishti-AI is a Flask-based drinking-water analytics
application.

Its main components are:

1. Data processing:
   Python and pandas process the MIS 78th Round dataset.

2. Dashboard:
   The application displays drinking-water access
   statistics, state comparisons, and visualizations.

3. AI Predictor:
   A scikit-learn GradientBoostingRegressor predicts
   improved drinking-water access percentages.

4. AI Chatbot:
   IBM watsonx.ai integrates the Granite model to answer
   general water-related questions.

5. Backend:
   Flask provides API endpoints connecting the dashboard,
   prediction functionality, and chatbot.

The project uses the MIS 78th Round Excel dataset
as its primary source of drinking-water access data.
"""


def _dataset_answer() -> str:

    return """
JalDrishti-AI uses the MIS 78th Round drinking-water
dataset stored in:

data/nss78_21.xlsx

The dataset contains fields such as:

- State
- Age group
- Sector
- Gender
- Indicator
- Sub-indicator
- Value
- Total count

The project uses these records to analyze improved
drinking-water access, piped-water access, rural and
urban differences, and state-level performance.

The dataset is also used to prepare the training records
for the machine-learning prediction model.

The dataset contains statistical values; it does not
provide the original report's written paragraphs.
"""


def _get_verified_answer(user_question: str):

    q = user_question.lower()

    # --------------------------------------------------------
    # GRADIENT BOOSTING / MACHINE LEARNING
    # --------------------------------------------------------

    model_keywords = [
        "gradient boosting",
        "gradientboosting",
        "regressor",
        "machine learning model",
        "prediction model",
        "model configuration",
        "model features",
        "model inputs",
        "how does the model",
        "how does it predict",
        "what algorithm"
    ]

    if any(keyword in q for keyword in model_keywords):
        return _project_model_answer()

    # --------------------------------------------------------
    # PROJECT ARCHITECTURE
    # --------------------------------------------------------

    architecture_keywords = [
        "project architecture",
        "system architecture",
        "architecture of jaldrishti",
        "technologies used",
        "technology stack",
        "tech stack",
        "how is jaldrishti built"
    ]

    if any(keyword in q for keyword in architecture_keywords):
        return _project_architecture_answer()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    dataset_keywords = [
        "which dataset",
        "dataset used",
        "data source",
        "mis 78th round dataset",
        "what data does",
        "excel dataset"
    ]

    if any(keyword in q for keyword in dataset_keywords):
        return _dataset_answer()

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

    # First check verified project-specific information.
    verified_answer = _get_verified_answer(user_question)

    if verified_answer:
        return verified_answer.strip()

    # --------------------------------------------------------
    # IBM GRANITE SYSTEM INSTRUCTIONS
    # --------------------------------------------------------

    system_prompt = """
You are JalDrishti AI, an intelligent assistant for
India's drinking-water access and MIS 78th Round data.

You are powered by IBM Granite through IBM watsonx.ai.

PROJECT FACTS — FOLLOW THESE EXACTLY:

- The project uses Flask.
- The primary dataset is data/nss78_21.xlsx.
- The dataset is from the MIS 78th Round.
- The prediction algorithm is scikit-learn GradientBoostingRegressor.
- The model has 50 estimators.
- Maximum tree depth is 2.
- Learning rate is 0.05.
- Random state is 42.
- Input features are state and sector.
- Both input features are encoded using LabelEncoder.
- The target is improved drinking-water access percentage.
- Model validation uses 5-Fold K-Fold cross-validation.
- The evaluation metric is Mean Absolute Error (MAE).
- The trained model and encoders are saved using joblib.

NEVER claim that the model uses:
- Rainfall
- Income
- Literacy
- Ration-card status
- Governance scores
- Infrastructure budgets

NEVER claim that the project evaluates the model using RMSE.
The verified evaluation metric is MAE.

DATA ACCURACY RULES:

1. Use supplied project statistics for questions about
   state rankings, percentages, and water-access indicators.

2. Do not invent state rankings or survey percentages.

3. Improved drinking-water access and piped-water access
   are different indicators.

4. Do not invent report quotations, page numbers,
   citations, or references.

5. If a requested statistic is not available in the
   supplied context, clearly say so.

GENERAL KNOWLEDGE:

For general questions about drinking water, pollution,
water conservation, health effects, and SDG 6.1,
provide a useful explanation using general knowledge.

Do not falsely claim that general knowledge comes
from the MIS 78th Round dataset.

ANSWER STYLE:

- Use simple, clear English.
- Answer the actual question directly.
- Keep most answers between 2 and 5 sentences.
- Use bullets when explaining multiple points.
- Avoid unnecessary technical terminology.
- Do not invent project functionality.

OUTPUT:

Return only the final answer.
Do not return JSON, code, tool calls, or fake citations.
"""

    # --------------------------------------------------------
    # INCLUDE AVAILABLE DATASET CONTEXT
    # --------------------------------------------------------

    user_prompt = f"""
VERIFIED PROJECT DATA AND STATISTICS:

{context_summary if context_summary else
 "No additional dataset statistics were supplied for this question."}

USER QUESTION:

{user_question}

Answer the question accurately using the system instructions.
"""

    # --------------------------------------------------------
    # CALL IBM GRANITE
    # --------------------------------------------------------

    return _call_granite(
        system_prompt,
        user_prompt,
        max_tokens=300
    )
