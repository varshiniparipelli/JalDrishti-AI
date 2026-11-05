"""
JalDrishti-AI — IBM Watson Machine Learning Service
Deploy, score, and retrieve predictions from IBM Watson ML (Lite).
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

WML_API_KEY     = os.getenv("IBM_WATSON_ML_API_KEY")
WML_URL         = os.getenv("IBM_WATSON_ML_URL", "https://us-south.ml.cloud.ibm.com")
WML_SPACE_ID    = os.getenv("IBM_WATSON_ML_SPACE_ID")
DEPLOYMENT_ID   = os.getenv("MODEL_DEPLOYMENT_ID")
IAM_TOKEN_URL   = "https://iam.cloud.ibm.com/identity/token"


def _get_iam_token() -> str:
    """Exchange IBM API Key for a short-lived IAM bearer token."""
    resp = requests.post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WML_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def score_record(input_record: dict) -> dict:
    """
    Send a single household record to the deployed Watson ML model
    and return the prediction (improved_source probability + label).

    Parameters
    ----------
    input_record : dict
        Keys: sector, social_group, religion, household_size,
              water_collection_time_min, has_clean_fuel, migration_last_year

    Returns
    -------
    dict with keys: predicted_label, probability_improved, probability_not_improved
    """
    token = _get_iam_token()
    endpoint = f"{WML_URL}/ml/v4/deployments/{DEPLOYMENT_ID}/predictions?version=2021-05-01"

    payload = {
        "input_data": [
            {
                "fields": list(input_record.keys()),
                "values": [list(input_record.values())],
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    predictions = result["predictions"][0]
    values      = predictions["values"][0]
    fields      = predictions["fields"]

    label_idx    = fields.index("prediction")
    prob_idx     = fields.index("probability")
    probabilities = values[prob_idx]

    return {
        "predicted_label": values[label_idx],
        "probability_improved":     round(probabilities[1], 4),
        "probability_not_improved": round(probabilities[0], 4),
    }


def list_deployments() -> list:
    """List all active WML deployments in the configured space."""
    token = _get_iam_token()
    url   = f"{WML_URL}/ml/v4/deployments?space_id={WML_SPACE_ID}&version=2021-05-01"
    resp  = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    resources = resp.json().get("resources", [])
    return [
        {
            "id":     r["metadata"]["id"],
            "name":   r["metadata"]["name"],
            "status": r["entity"]["status"]["state"],
        }
        for r in resources
    ]
