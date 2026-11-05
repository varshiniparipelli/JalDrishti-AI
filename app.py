"""
JalDrishti-AI — Flask Application Entry Point
REST API serving the MIS 78th Round water-access analytics dashboard.
Uses IBM Cloud Lite: Watson ML, COS, Db2, Watson Assistant.
"""

import os
from flask import Flask, jsonify, request, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from analysis import (
    load_data, compute_kpis, state_analysis, sector_analysis,
    water_source_distribution, state_gap_analysis,
    bottom_states, top_states,
    train_model, predict_access, sdg_progress,
)

WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "d35a6f68-c129-4634-b823-ff9583f7fde6")

app  = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "jaldrishti-dev-key")

# ── Cache dataset in memory ───────────────────────────────────────────────
_df: pd.DataFrame = None

def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = load_data()
    return _df


# ── Pages ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── KPI endpoint ──────────────────────────────────────────────────────────

@app.route("/api/kpis")
def api_kpis():
    df   = get_df()
    kpis = compute_kpis(df)
    return jsonify({"status": "ok", "data": kpis})


# ── State Analysis ────────────────────────────────────────────────────────

@app.route("/api/state-analysis")
def api_state():
    df   = get_df()
    data = state_analysis(df)
    return jsonify({"status": "ok", "data": data})


# ── Sector Analysis ───────────────────────────────────────────────────────

@app.route("/api/sector-analysis")
def api_sector():
    df   = get_df()
    data = sector_analysis(df)
    return jsonify({"status": "ok", "data": data})


# ── Water Source Distribution ─────────────────────────────────────────────

@app.route("/api/water-sources")
def api_water_sources():
    df   = get_df()
    data = water_source_distribution(df)
    return jsonify({"status": "ok", "data": data})


# ── State Gap Analysis ────────────────────────────────────────────────────

@app.route("/api/state-gap")
def api_state_gap():
    df   = get_df()
    data = state_gap_analysis(df)
    return jsonify({"status": "ok", "data": data})


# ── Top / Bottom States ───────────────────────────────────────────────────

@app.route("/api/top-states")
def api_top_states():
    df  = get_df()
    n   = int(request.args.get("n", 10))
    return jsonify({"status": "ok", "data": top_states(df, n)})


@app.route("/api/bottom-states")
def api_bottom_states():
    df  = get_df()
    n   = int(request.args.get("n", 10))
    return jsonify({"status": "ok", "data": bottom_states(df, n)})


# ── SDG Progress ──────────────────────────────────────────────────────────

@app.route("/api/sdg-progress")
def api_sdg():
    df   = get_df()
    data = sdg_progress(df)
    return jsonify({"status": "ok", "data": data})


# ── Train Model ───────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    df     = get_df()
    result = train_model(df)
    return jsonify({"status": "ok", "data": result})


# ── Predict (local sklearn model) ─────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    missing = [k for k in ["state", "sector"] if k not in body]
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {missing}"}), 400
    try:
        result = predict_access(body)
        return jsonify({"status": "ok", "data": result})
    except FileNotFoundError:
        return jsonify({"status": "error",
                        "message": "Model not trained yet. POST /api/train first."}), 503


# ── Watson ML remote predict ──────────────────────────────────────────────

@app.route("/api/watson-predict", methods=["POST"])
def api_watson_predict():
    try:
        from services.watson_ml_service import score_record
        body   = request.get_json(force=True)
        result = score_record(body)
        return jsonify({"status": "ok", "data": result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Watson Assistant Chat ─────────────────────────────────────────────────

@app.route("/api/chat/session", methods=["POST"])
def chat_session():
    try:
        from services.assistant_service import create_session
        sid = create_session()
        session["watson_session_id"] = sid
        return jsonify({"status": "ok", "session_id": sid})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/chat/message", methods=["POST"])
def chat_message():
    try:
        from services.assistant_service import send_message
        body = request.get_json(force=True)
        sid  = body.get("session_id") or session.get("watson_session_id")
        text = body.get("text", "")
        if not sid:
            return jsonify({"status": "error", "message": "No session_id provided"}), 400
        result = send_message(sid, text)
        return jsonify({"status": "ok", "data": result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── COS Upload ────────────────────────────────────────────────────────────

@app.route("/api/cos/upload", methods=["POST"])
def cos_upload():
    try:
        from services.cos_service import ensure_bucket_exists, upload_file
        ensure_bucket_exists()
        xlsx_path = os.path.join(os.path.dirname(__file__), "data", "nss78_21.xlsx")
        ok = upload_file(xlsx_path)
        return jsonify({"status": "ok" if ok else "error",
                        "message": "Uploaded to IBM COS" if ok else "Upload failed"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── watsonx.ai — National Insight ────────────────────────────────────────

@app.route("/api/watsonx/national-insight", methods=["GET"])
def watsonx_national():
    try:
        from services.watsonx_service import analyse_national
        df   = get_df()
        sdg  = sdg_progress(df)
        from analysis import compute_kpis
        kpis = compute_kpis(df)
        text = analyse_national(
            national_pct   = kpis["national_improved_pct"],
            gap_pp         = kpis["rural_urban_gap"],
            projected_2030 = sdg["projected_2030_pct"],
        )
        return jsonify({"status": "ok", "insight": text,
                        "project_id": WATSONX_PROJECT_ID})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── watsonx.ai — State Insight ────────────────────────────────────────────

@app.route("/api/watsonx/state-insight", methods=["POST"])
def watsonx_state():
    try:
        from services.watsonx_service import analyse_state
        body  = request.get_json(force=True)
        state = body.get("state", "Bihar")
        df    = get_df()
        from analysis import _improved, _piped, SUB_IMPROVED, SUB_PIPED
        imp   = _improved(df)
        piped = _piped(df)

        rural_row = imp[(imp["state"] == state) & (imp["sector"] == "Rural")]
        urban_row = imp[(imp["state"] == state) & (imp["sector"] == "Urban")]
        piped_row = piped[(piped["state"] == state) & (piped["sector"] == "All")]

        rural_pct = float(rural_row["value"].values[0]) if len(rural_row) else 0.0
        urban_pct = float(urban_row["value"].values[0]) if len(urban_row) else 0.0
        piped_pct = float(piped_row["value"].values[0]) if len(piped_row) else 0.0

        text = analyse_state(state, rural_pct, urban_pct, piped_pct)
        return jsonify({"status": "ok", "state": state, "insight": text,
                        "rural_pct": rural_pct, "urban_pct": urban_pct,
                        "piped_pct": piped_pct,
                        "project_id": WATSONX_PROJECT_ID})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── watsonx.ai — Chat ─────────────────────────────────────────────────────

@app.route("/api/watsonx/chat", methods=["POST"])
def watsonx_chat():
    try:
        from services.watsonx_service import chat_response
        body = request.get_json(force=True)
        text = body.get("text", "")
        if not text:
            return jsonify({"status": "error", "message": "text is required"}), 400
        reply = chat_response(text)
        return jsonify({"status": "ok", "response": reply,
                        "model": "ibm/granite-4-h-small",
                        "project_id": WATSONX_PROJECT_ID})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── Health ────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "project": "JalDrishti-AI", "survey_round": 78,
                    "dataset": "nss78_21.xlsx (MIS 78th Round)",
                    "watsonx_project_id": WATSONX_PROJECT_ID})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=os.getenv("FLASK_DEBUG", "True") == "True", host="0.0.0.0", port=port)
