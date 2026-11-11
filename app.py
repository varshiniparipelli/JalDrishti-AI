
"""
JalDrishti-AI — Flask Application Entry Point
REST API serving MIS 78th Round water-access analytics dashboard.
"""

import os
import re

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
    age_group_analysis,
    gender_analysis,
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
    """Build chatbot context using actual MIS 78th Round calculations."""

    kpis = compute_kpis(df)
    states = state_analysis(df)
    top = top_states(df, 10)
    bottom = bottom_states(df, 10)
    sectors = sector_analysis(df)
    sources = water_source_distribution(df)
    gaps = state_gap_analysis(df)
    sdg = sdg_progress(df)

    return f"""
You are JalDrishti AI, an assistant for India's drinking-water
access analytics project.

DATA SOURCE:
MIS 78th Round dataset (2020-2021).

NATIONAL KPI RESULTS:
{kpis}

STATE ANALYSIS:
{states}

TOP 10 STATES:
{top}

BOTTOM 10 STATES:
{bottom}

SECTOR ANALYSIS:
{sectors}

WATER SOURCE DISTRIBUTION:
{sources}

RURAL-URBAN STATE GAPS:
{gaps}

SDG PROGRESS:
{sdg}

IMPORTANT RULES:
1. Use supplied dataset results for survey statistics.
2. Improved drinking-water access and piped-water access
   are different indicators.
3. Improved access does not mean tap-water connectivity.
4. For state rankings, use overall improved access.
5. Never invent state names, percentages or rankings.
6. Never use placeholders such as [percentage].
7. If information is unavailable, say so clearly.
8. Dataset values represent the survey period, not
   real-time current conditions.
9. Clearly distinguish survey estimates from projections.
10. Answer in clear, simple English.
11. Do not invent citations or report quotations.
12. Return plain text, not JSON.
"""


# ============================================================
# HOME PAGE
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
    return jsonify({"status": "ok", "data": compute_kpis(df)})


# ============================================================
# STATE ANALYSIS
# ============================================================

@app.route("/api/state-analysis")
def api_state():
    return jsonify({
        "status": "ok",
        "data": state_analysis(get_df())
    })


# ============================================================
# SECTOR ANALYSIS
# ============================================================

@app.route("/api/sector-analysis")
def api_sector():
    return jsonify({
        "status": "ok",
        "data": sector_analysis(get_df())
    })


# ============================================================
# WATER SOURCE DISTRIBUTION
# ============================================================

@app.route("/api/water-sources")
def api_water_sources():
    return jsonify({
        "status": "ok",
        "data": water_source_distribution(get_df())
    })


# ============================================================
# STATE GAP ANALYSIS
# ============================================================

@app.route("/api/state-gap")
def api_state_gap():
    return jsonify({
        "status": "ok",
        "data": state_gap_analysis(get_df())
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
    return jsonify({
        "status": "ok",
        "data": sdg_progress(get_df())
    })


# ============================================================
# AGE GROUP ANALYSIS
# ============================================================

@app.route("/api/age-group-analysis")
def api_age_group():
    return jsonify({
        "status": "ok",
        "data": age_group_analysis(get_df())
    })


# ============================================================
# GENDER ANALYSIS
# ============================================================

@app.route("/api/gender-analysis")
def api_gender():
    return jsonify({
        "status": "ok",
        "data": gender_analysis(get_df())
    })


# ============================================================
# FULL REPORT — single call returns everything
# ============================================================

@app.route("/api/full-report")
def api_full_report():
    df = get_df()
    return jsonify({
        "status": "ok",
        "data": {
            "kpis":             compute_kpis(df),
            "sdg_progress":     sdg_progress(df),
            "sector_analysis":  sector_analysis(df),
            "top_states":       top_states(df, 10),
            "bottom_states":    bottom_states(df, 10),
            "state_gap":        state_gap_analysis(df),
            "water_sources":    water_source_distribution(df),
            "age_group":        age_group_analysis(df),
            "gender":           gender_analysis(df),
        }
    })


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@app.route("/api/train", methods=["POST"])
def api_train():
    return jsonify({
        "status": "ok",
        "data": train_model(get_df())
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
        return jsonify({
            "status": "ok",
            "data": predict_access(body)
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

        return jsonify({"status": "ok", "data": result})

    except Exception as exc:
        app.logger.exception("Watson ML prediction error")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ============================================================
# IBM WATSON ASSISTANT CHAT
# ============================================================

@app.route("/api/chat/session", methods=["POST"])
def chat_session():
    try:
        from services.assistant_service import create_session

        sid = create_session()
        session["watson_session_id"] = sid

        return jsonify({"status": "ok", "session_id": sid})

    except Exception as exc:
        app.logger.exception("Watson Assistant session error")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/chat/message", methods=["POST"])
def chat_message():
    try:
        from services.assistant_service import send_message

        body = request.get_json(force=True) or {}
        sid = body.get("session_id") or session.get("watson_session_id")
        text = body.get("text", "")

        if not sid:
            return jsonify({
                "status": "error",
                "message": "No session_id provided"
            }), 400

        return jsonify({
            "status": "ok",
            "data": send_message(sid, text)
        })

    except Exception as exc:
        app.logger.exception("Watson Assistant message error")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ============================================================
# IBM CLOUD OBJECT STORAGE UPLOAD
# ============================================================

@app.route("/api/cos/upload", methods=["POST"])
def cos_upload():
    try:
        from services.cos_service import ensure_bucket_exists, upload_file

        ensure_bucket_exists()

        xlsx_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "nss78_21.xlsx"
        )

        ok = upload_file(xlsx_path)

        return jsonify({
            "status": "ok" if ok else "error",
            "message": "Uploaded to IBM COS" if ok else "Upload failed"
        })

    except Exception as exc:
        app.logger.exception("COS upload error")
        return jsonify({"status": "error", "message": str(exc)}), 500


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
        return jsonify({"status": "error", "message": str(exc)}), 500


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

        rural_pct = float(rural_row["value"].values[0]) if len(rural_row) else 0.0
        urban_pct = float(urban_row["value"].values[0]) if len(urban_row) else 0.0
        piped_pct = float(piped_row["value"].values[0]) if len(piped_row) else 0.0

        text = analyse_state(state, rural_pct, urban_pct, piped_pct)

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
        return jsonify({"status": "error", "message": str(exc)}), 500


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
        q = question.lower()

        bottom = bottom_states(df, 10)
        top = top_states(df, 10)

        # ----------------------------------------------------
        # HIGHEST AND LOWEST IN THE SAME QUESTION
        # This must run before individual highest/lowest rules.
        # ----------------------------------------------------

        asks_both = (
            ("highest" in q and "lowest" in q)
            or ("highest" in q and "least" in q)
            or ("top" in q and "bottom" in q)
            or ("best" in q and "worst" in q)
        )

        if asks_both:
            highest = top[0]
            lowest = bottom[0]

            difference = round(
                float(highest["improved_pct"])
                - float(lowest["improved_pct"]),
                1
            )

            reply = (
                "According to the MIS 78th Round dataset:\n\n"
                f"Highest: {highest['state']} — "
                f"{highest['improved_pct']}% improved drinking-water access.\n\n"
                f"Lowest: {lowest['state']} — "
                f"{lowest['improved_pct']}% improved drinking-water access.\n\n"
                f"The difference is {difference} percentage points."
            )

            return jsonify({
                "status": "ok",
                "response": reply,
                "model": "dataset-analysis",
                "source": "MIS 78th Round"
            })

        # ----------------------------------------------------
        # NAMED STATE COMPARISON
        # ----------------------------------------------------

        comparison_intent = any(term in q for term in [
            "compare",
            "comparison",
            " versus ",
            " vs ",
            "difference between",
            "between"
        ])

        if comparison_intent:
            all_states = state_analysis(df)
            matched_states = []

            for state_row in all_states:
                state_name = str(state_row.get("state", "")).strip()

                if state_name and re.search(
                    r"\b" + re.escape(state_name.lower()) + r"\b",
                    q
                ):
                    matched_states.append(state_row)

            unique_states = []
            seen_names = set()

            for state_row in matched_states:
                name_key = str(state_row.get("state", "")).strip().lower()

                if name_key not in seen_names:
                    unique_states.append(state_row)
                    seen_names.add(name_key)

            if len(unique_states) == 2:
                first, second = unique_states

                first_overall = float(first["improved_all_pct"])
                second_overall = float(second["improved_all_pct"])
                overall_difference = round(
                    abs(first_overall - second_overall), 1
                )

                reply = (
                    "MIS 78th Round state comparison:\n\n"
                    f"{first['state']}:\n"
                    f"- Rural improved drinking-water access: "
                    f"{first.get('improved_rural_pct')}%\n"
                    f"- Urban improved drinking-water access: "
                    f"{first.get('improved_urban_pct')}%\n"
                    f"- Overall improved drinking-water access: "
                    f"{first_overall}%\n\n"
                    f"{second['state']}:\n"
                    f"- Rural improved drinking-water access: "
                    f"{second.get('improved_rural_pct')}%\n"
                    f"- Urban improved drinking-water access: "
                    f"{second.get('improved_urban_pct')}%\n"
                    f"- Overall improved drinking-water access: "
                    f"{second_overall}%\n\n"
                )

                if first_overall > second_overall:
                    reply += (
                        f"{first['state']} has {overall_difference} "
                        "percentage points higher overall improved "
                        f"drinking-water access than {second['state']}."
                    )
                elif second_overall > first_overall:
                    reply += (
                        f"{second['state']} has {overall_difference} "
                        "percentage points higher overall improved "
                        f"drinking-water access than {first['state']}."
                    )
                else:
                    reply += (
                        "Both states have the same overall improved "
                        "drinking-water access in the supplied dataset."
                    )

                return jsonify({
                    "status": "ok",
                    "response": reply,
                    "model": "dataset-analysis",
                    "source": "MIS 78th Round"
                })

        # ----------------------------------------------------
        # TOP VS BOTTOM DIFFERENCE
        # ----------------------------------------------------

        if (
            "difference" in q
            and "top" in q
            and ("bottom" in q or "lowest" in q)
        ):
            lowest = bottom[0]
            highest = top[0]

            difference = round(
                float(highest["improved_pct"])
                - float(lowest["improved_pct"]),
                1
            )

            reply = (
                f"{highest['state']} has the highest improved "
                f"drinking-water access at {highest['improved_pct']}%, "
                f"while {lowest['state']} has the lowest at "
                f"{lowest['improved_pct']}%. The difference is "
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
                f"drinking-water access at {result['improved_pct']}%."
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
                f"drinking-water access at {result['improved_pct']}%."
            )

            return jsonify({
                "status": "ok",
                "response": reply,
                "model": "dataset-analysis",
                "source": "MIS 78th Round"
            })

        # ----------------------------------------------------
        # GENERAL / EDUCATIONAL QUESTIONS — IBM GRANITE
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
        return jsonify({"status": "error", "message": str(exc)}), 500


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
