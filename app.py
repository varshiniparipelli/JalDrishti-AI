
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
    load_data,
    compute_kpis,
    state_analysis,
    sector_analysis,
    water_source_distribution,
    state_gap_analysis,
    bottom_states,
    top_states,
    train_model,
    predict_access,
    sdg_progress,
)

WATSONX_PROJECT_ID = os.getenv(
    "WATSONX_PROJECT_ID",
    "d35a6f68-c129-4634-b823-ff9583f7fde6"
)

app = Flask(__name__)
CORS(app)

app.secret_key = os.getenv("SECRET_KEY", "jaldrishti-dev-key")


# ============================================================
# DATASET CACHE
# ============================================================

_df: pd.DataFrame = None


def get_df() -> pd.DataFrame:
    """Load the dataset once and reuse it."""
    global _df

    if _df is None:
        _df = load_data()

    return _df


# ============================================================
# CHATBOT DATA CONTEXT
# ============================================================

def build_chat_context(df):
    """
    Build chatbot context using actual MIS 78th Round
    dataset calculations.
    """

    kpis = compute_kpis(df)
    states = state_analysis(df)
    top = top_states(df, 10)
    bottom = bottom_states(df, 10)
    sectors = sector_analysis(df)
    sources = water_source_distribution(df)
    gaps = state_gap_analysis(df)
    sdg = sdg_progress(df)

    context = f"""
You are JalDrishti AI, an assistant for India's drinking-water
access analytics project.

DATA SOURCE:
MIS 78th Round dataset (2020-2021).

IMPORTANT:
Use the supplied dataset results for all questions about
survey statistics and state rankings.

NATIONAL KPI RESULTS:
{kpis}

STATE ANALYSIS:
{states}

TOP 10 STATES BY IMPROVED DRINKING-WATER ACCESS:
{top}

BOTTOM 10 STATES BY IMPROVED DRINKING-WATER ACCESS:
{bottom}

SECTOR ANALYSIS:
{sectors}

WATER SOURCE DISTRIBUTION:
{sources}

RURAL-URBAN STATE GAPS:
{gaps}

SDG PROGRESS:
{sdg}

IMPORTANT DEFINITIONS AND RULES:

1. Improved drinking-water access and piped-water access
   are two different indicators.

2. Improved drinking-water access can include eligible
   improved sources other than piped household connections.

3. Do not describe improved drinking-water access as
   tap-water connectivity.

4. For state rankings, use overall (All sector)
   improved drinking-water access.

5. Do not invent state names, percentages, rankings,
   survey results, citations or references.

6. Do not use placeholders such as [percentage].

7. If the supplied dataset does not contain the requested
   information, clearly say that the information is unavailable.

8. The dataset represents the survey period, not real-time
   current water-access conditions.

9. Explain SDG 6.1 using the supplied progress figures.
   Clearly distinguish survey values from projections.

10. Answer in clear, simple language in approximately
    2-4 sentences.

11. Do not claim to have real-time browsing capability.

12. Return plain text, not JSON.
"""

    return context


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# KPI ENDPOINT
# ============================================================

@app.route("/api/kpis")
def api_kpis():

    df = get_df()
    kpis = compute_kpis(df)

    return jsonify({
        "status": "ok",
        "data": kpis
    })


# ============================================================
# STATE ANALYSIS
# ============================================================

@app.route("/api/state-analysis")
def api_state():

    df = get_df()
    data = state_analysis(df)

    return jsonify({
        "status": "ok",
        "data": data
    })


# ============================================================
# SECTOR ANALYSIS
# ============================================================

@app.route("/api/sector-analysis")
def api_sector():

    df = get_df()
    data = sector_analysis(df)

    return jsonify({
        "status": "ok",
        "data": data
    })


# ============================================================
# WATER SOURCE DISTRIBUTION
# ============================================================

@app.route("/api/water-sources")
def api_water_sources():

    df = get_df()
    data = water_source_distribution(df)

    return jsonify({
        "status": "ok",
        "data": data
    })


# ============================================================
# STATE GAP ANALYSIS
# ============================================================

@app.route("/api/state-gap")
def api_state_gap():

    df = get_df()
    data = state_gap_analysis(df)

    return jsonify({
        "status": "ok",
        "data": data
    })


# ============================================================
# TOP AND BOTTOM STATES
# ============================================================

@app.route("/api/top-states")
def api_top_states():

    df = get_df()
    n = int(request.args.get("n", 10))

    return jsonify({
        "status": "ok",
        "data": top_states(df, n)
    })


@app.route("/api/bottom-states")
def api_bottom_states():

    df = get_df()
    n = int(request.args.get("n", 10))

    return jsonify({
        "status": "ok",
        "data": bottom_states(df, n)
    })


# ============================================================
# SDG PROGRESS
# ============================================================

@app.route("/api/sdg-progress")
def api_sdg():

    df = get_df()
    data = sdg_progress(df)

    return jsonify({
        "status": "ok",
        "data": data
    })


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@app.route("/api/train", methods=["POST"])
def api_train():

    df = get_df()
    result = train_model(df)

    return jsonify({
        "status": "ok",
        "data": result
    })


# ============================================================
# LOCAL MACHINE LEARNING PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    body = request.get_json(force=True) or {}

    missing = [
        key for key in ["state", "sector"]
        if key not in body
    ]

    if missing:
        return jsonify({
            "status": "error",
            "message": f"Missing fields: {missing}"
        }), 400

    try:
        result = predict_access(body)

        return jsonify({
            "status": "ok",
            "data": result
        })

    except FileNotFoundError:

        return jsonify({
            "status": "error",
            "message": "Model not trained yet. POST /api/train first."
        }), 503


# ============================================================
# IBM WATSON MACHINE LEARNING REMOTE PREDICTION
# ============================================================

@app.route("/api/watson-predict", methods=["POST"])
def api_watson_predict():

    try:
        from services.watson_ml_service import score_record

        body = request.get_json(force=True)
        result = score_record(body)

        return jsonify({
            "status": "ok",
            "data": result
        })

    except Exception as exc:

        app.logger.exception("Watson ML prediction error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# IBM WATSON ASSISTANT CHAT
# ============================================================

@app.route("/api/chat/session", methods=["POST"])
def chat_session():

    try:
        from services.assistant_service import create_session

        sid = create_session()
        session["watson_session_id"] = sid

        return jsonify({
            "status": "ok",
            "session_id": sid
        })

    except Exception as exc:

        app.logger.exception("Watson Assistant session error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.route("/api/chat/message", methods=["POST"])
def chat_message():

    try:
        from services.assistant_service import send_message

        body = request.get_json(force=True) or {}

        sid = (
            body.get("session_id")
            or session.get("watson_session_id")
        )

        text = body.get("text", "")

        if not sid:
            return jsonify({
                "status": "error",
                "message": "No session_id provided"
            }), 400

        result = send_message(sid, text)

        return jsonify({
            "status": "ok",
            "data": result
        })

    except Exception as exc:

        app.logger.exception("Watson Assistant message error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# IBM CLOUD OBJECT STORAGE UPLOAD
# ============================================================

@app.route("/api/cos/upload", methods=["POST"])
def cos_upload():

    try:
        from services.cos_service import (
            ensure_bucket_exists,
            upload_file
        )

        ensure_bucket_exists()

        xlsx_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "nss78_21.xlsx"
        )

        ok = upload_file(xlsx_path)

        return jsonify({
            "status": "ok" if ok else "error",
            "message": (
                "Uploaded to IBM COS"
                if ok else "Upload failed"
            )
        })

    except Exception as exc:

        app.logger.exception("COS upload error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# WATSONX.AI — NATIONAL INSIGHT
# ============================================================

@app.route("/api/watsonx/national-insight", methods=["GET"])
def watsonx_national():

    try:
        from services.watsonx_service import analyse_national

        df = get_df()

        sdg = sdg_progress(df)
        kpis = compute_kpis(df)

        text = analyse_national(
            national_pct=kpis["national_improved_pct"],
            gap_pp=kpis["rural_urban_gap"],
            projected_2030=sdg["projected_2030_pct"]
        )

        return jsonify({
            "status": "ok",
            "insight": text,
            "project_id": WATSONX_PROJECT_ID
        })

    except Exception as exc:

        app.logger.exception("Watsonx national insight error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# WATSONX.AI — STATE INSIGHT
# ============================================================

@app.route("/api/watsonx/state-insight", methods=["POST"])
def watsonx_state():

    try:
        from services.watsonx_service import analyse_state

        body = request.get_json(force=True) or {}
        state = body.get("state", "Bihar")

        df = get_df()

        from analysis import _improved, _piped

        imp = _improved(df)
        piped = _piped(df)

        rural_row = imp[
            (imp["state"] == state)
            & (imp["sector"] == "Rural")
        ]

        urban_row = imp[
            (imp["state"] == state)
            & (imp["sector"] == "Urban")
        ]

        piped_row = piped[
            (piped["state"] == state)
            & (piped["sector"] == "All")
        ]

        rural_pct = (
            float(rural_row["value"].values[0])
            if len(rural_row) else 0.0
        )

        urban_pct = (
            float(urban_row["value"].values[0])
            if len(urban_row) else 0.0
        )

        piped_pct = (
            float(piped_row["value"].values[0])
            if len(piped_row) else 0.0
        )

        text = analyse_state(
            state,
            rural_pct,
            urban_pct,
            piped_pct
        )

        return jsonify({
            "status": "ok",
            "state": state,
            "insight": text,
            "rural_pct": rural_pct,
            "urban_pct": urban_pct,
            "piped_pct": piped_pct,
            "project_id": WATSONX_PROJECT_ID
        })

    except Exception as exc:

        app.logger.exception("Watsonx state insight error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# WATSONX.AI — JALDRISHTI AI CHATBOT
# ============================================================

@app.route("/api/watsonx/chat", methods=["POST"])
def watsonx_chat():

    try:
        from services.watsonx_service import chat_response

        body = request.get_json(force=True) or {}
        question = str(body.get("text", "")).strip()

        if not question:
            return jsonify({
                "status": "error",
                "message": "text is required"
            }), 400

        df = get_df()

        # Calculate rankings directly from the dataset
        bottom = bottom_states(df, 10)
        top = top_states(df, 10)

        q = question.lower()

        # ----------------------------------------------------
        # TOP VS BOTTOM COMPARISON
        # ----------------------------------------------------

        if (
            "difference" in q
            and "top" in q
            and (
                "bottom" in q
                or "lowest" in q
            )
        ):

            lowest = bottom[0]
            highest = top[0]

            difference = round(
                highest["improved_pct"]
                - lowest["improved_pct"],
                1
            )

            reply = (
                f"{highest['state']} has the highest improved "
                f"drinking-water access at "
                f"{highest['improved_pct']}%, while "
                f"{lowest['state']} has the lowest at "
                f"{lowest['improved_pct']}%. "
                f"The difference between them is "
                f"{difference} percentage points."
            )

            return jsonify({
                "status": "ok",
                "response": reply,
                "model": "dataset-analysis",
                "source": "MIS 78th Round"
            })

        # ----------------------------------------------------
        # LOWEST STATE
        # ----------------------------------------------------

        lowest_keywords = [
            "lowest",
            "least access",
            "least accessibility",
            "bottom-performing",
            "worst performing"
        ]

        if any(word in q for word in lowest_keywords):

            result = bottom[0]

            reply = (
                f"According to the MIS 78th Round dataset, "
                f"{result['state']} has the lowest improved "
                f"drinking-water access at "
                f"{result['improved_pct']}%."
            )

            return jsonify({
                "status": "ok",
                "response": reply,
                "model": "dataset-analysis",
                "source": "MIS 78th Round"
            })

        # ----------------------------------------------------
        # HIGHEST STATE
        # ----------------------------------------------------

        highest_keywords = [
            "highest",
            "best-performing",
            "top-performing",
            "most accessible"
        ]

        if any(word in q for word in highest_keywords):

            result = top[0]

            reply = (
                f"According to the MIS 78th Round dataset, "
                f"{result['state']} has the highest improved "
                f"drinking-water access at "
                f"{result['improved_pct']}%."
            )

            return jsonify({
                "status": "ok",
                "response": reply,
                "model": "dataset-analysis",
                "source": "MIS 78th Round"
            })

        # ----------------------------------------------------
        # OTHER QUESTIONS — IBM GRANITE
        # ----------------------------------------------------

        context = build_chat_context(df)

        reply = chat_response(
            user_question=question,
            context_summary=context
        )

        return jsonify({
            "status": "ok",
            "response": reply,
            "model": "ibm/granite-4-h-small",
            "project_id": WATSONX_PROJECT_ID,
            "source": "MIS 78th Round"
        })

    except Exception as exc:

        app.logger.exception("Watsonx chatbot error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "project": "JalDrishti-AI",
        "survey_round": 78,
        "dataset": "nss78_21.xlsx (MIS 78th Round)",
        "watsonx_project_id": WATSONX_PROJECT_ID
    })


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", 5000))

    app.run(
        debug=os.getenv("FLASK_DEBUG", "True") == "True",
        host="0.0.0.0",
        port=port
    )
