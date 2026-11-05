"""
JalDrishti-AI — IBM Watson Assistant Chatbot Service
Connect to Watson Assistant (Lite) to power the on-page Q&A chatbot
about drinking water access insights.
"""

import os
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv

load_dotenv()

_assistant_client = None


def _get_client() -> AssistantV2:
    global _assistant_client
    if _assistant_client is None:
        auth = IAMAuthenticator(os.getenv("IBM_ASSISTANT_API_KEY"))
        client = AssistantV2(version="2023-06-15", authenticator=auth)
        client.set_service_url(os.getenv("IBM_ASSISTANT_URL"))
        _assistant_client = client
    return _assistant_client


def create_session() -> str:
    """Create a new Watson Assistant session and return the session_id."""
    client = _get_client()
    env_id = os.getenv("IBM_ASSISTANT_ENVIRONMENT_ID")
    resp   = client.create_session(assistant_id=env_id).get_result()
    return resp["session_id"]


def send_message(session_id: str, user_text: str) -> dict:
    """
    Send a message to Watson Assistant and return the structured response.

    Returns
    -------
    dict with keys: response_text (str), session_id (str)
    """
    client  = _get_client()
    env_id  = os.getenv("IBM_ASSISTANT_ENVIRONMENT_ID")

    resp = client.message(
        assistant_id=env_id,
        session_id=session_id,
        input={"message_type": "text", "text": user_text},
    ).get_result()

    texts = []
    for generic in resp.get("output", {}).get("generic", []):
        if generic.get("response_type") == "text":
            texts.append(generic.get("text", ""))

    return {
        "response_text": " ".join(texts) or "I'm sorry, I didn't understand that.",
        "session_id": session_id,
    }


def delete_session(session_id: str):
    """Delete a Watson Assistant session after use."""
    client = _get_client()
    env_id = os.getenv("IBM_ASSISTANT_ENVIRONMENT_ID")
    client.delete_session(assistant_id=env_id, session_id=session_id)
