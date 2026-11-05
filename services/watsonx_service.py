"""
JalDrishti-AI — IBM watsonx.ai Service
Uses the IBM watsonx.ai foundation model API (ibm/granite-4-h-small) to generate
natural-language insights from MIS 78th Round drinking water data.
Project ID: d35a6f68-c129-4634-b823-ff9583f7fde6
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "d35a6f68-c129-4634-b823-ff9583f7fde6")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_WATSON_ML_API_KEY")
IAM_TOKEN_URL      = "https://iam.cloud.ibm.com/identity/token"

# IBM Granite-4 small — confirmed available on watsonx Lite
DEFAULT_MODEL    = "ibm/granite-4-h-small"
WATSONX_API_VER  = "2024-05-01"


def _get_iam_token() -> str:
    """Exchange IBM API Key for a short-lived IAM bearer token."""
    resp = requests.post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def generate_insight(prompt: str, max_tokens: int = 300) -> str:
    """
    Call IBM watsonx.ai text generation API with the given prompt.
    Returns the generated text string.
    """
    token    = _get_iam_token()
    endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version={WATSONX_API_VER}"

    payload = {
        "model_id":   DEFAULT_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "input":      prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens":  max_tokens,
            "min_new_tokens":  20,
            "stop_sequences":  [],
            "repetition_penalty": 1.1,
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    return result["results"][0]["generated_text"].strip()


def analyse_state(state: str, rural_pct: float, urban_pct: float, piped_pct: float) -> str:
    """Generate a policy insight for a specific state using IBM Granite."""
    prompt = f"""You are a water policy analyst for India. Based on MIS 78th Round data:

State: {state}
- Rural improved water access: {rural_pct}%
- Urban improved water access: {urban_pct}%
- Piped water access: {piped_pct}%

Provide a concise 3-sentence policy recommendation to improve drinking water access in {state}, 
focusing on the rural-urban gap and SDG 6.1 targets."""
    return generate_insight(prompt, max_tokens=200)


def analyse_national(national_pct: float, gap_pp: float, projected_2030: float) -> str:
    """Generate a national-level insight using IBM Granite."""
    prompt = f"""You are a water policy analyst for India. Based on MIS 78th Round (2023) data:

- National improved water access: {national_pct}%
- Rural-Urban gap: {gap_pp} percentage points
- Projected 2030 coverage: {projected_2030}%
- SDG 6.1 target: 100% by 2030

Write a 3-sentence policy insight on India's progress toward SDG 6.1 and what interventions 
are most needed to close the remaining gap."""
    return generate_insight(prompt, max_tokens=200)


def chat_response(user_question: str, context_summary: str = "") -> str:
    """
    Generate a chatbot response using IBM Granite for water-access questions.
    Falls back gracefully if API key is not set.
    """
    if not WATSONX_API_KEY:
        raise ValueError("WATSONX_API_KEY not set in .env")

    prompt = f"""You are JalDrishti AI, an expert on India's drinking water access data from MIS 78th Round.
{f'Context: {context_summary}' if context_summary else ''}

User question: {user_question}

Answer concisely and factually in 2-3 sentences, referencing MIS 78th Round data where relevant."""
    return generate_insight(prompt, max_tokens=150)
