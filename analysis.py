
"""
JalDrishti-AI — Data Analysis Engine
Works with the REAL MIS 78th Round dataset (nss78_21.xlsx / AI Kosh).

Dataset structure:
  state | age_group | sector | gender | indicator | sub_indicator | value | total_count
  sub_indicator values:
    - "Piped Water into Dwelling or Yard/plot"
    - "Improved Source of Drinking Water"
  sector values: Rural | Urban | All
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.getenv("DATASET_PATH", "data/nss78_21.xlsx")

SUB_IMPROVED = "Improved Source of Drinking Water"
SUB_PIPED = "Piped Water into Dwelling or Yard/plot"

MODEL_PATH = os.path.join(BASE_DIR, "models", "water_access_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")


# ── Load Dataset ────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATASET_PATH if os.path.isabs(DATASET_PATH) else os.path.join(BASE_DIR, DATASET_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place nss78_21.xlsx in the data/ folder."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=0, header=0)
    else:
        df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ── Helper Functions ────────────────────────────────────────────────────────

def _improved(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for Improved Source of Drinking Water."""
    return df[df["sub_indicator"] == SUB_IMPROVED].copy()


def _piped(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for Piped Water into Dwelling or Yard/plot."""
    return df[df["sub_indicator"] == SUB_PIPED].copy()


# ── KPI Summary ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    imp = _improved(df)

    # All-India row for each sector
    ai_all = imp[(imp["state"] == "All India") & (imp["sector"] == "All")]
    ai_rural = imp[(imp["state"] == "All India") & (imp["sector"] == "Rural")]
    ai_urban = imp[(imp["state"] == "All India") & (imp["sector"] == "Urban")]

    national_pct = float(ai_all["value"].values[0]) if len(ai_all) else imp[imp["sector"] == "All"]["value"].mean()
    rural_pct = float(ai_rural["value"].values[0]) if len(ai_rural) else imp[imp["sector"] == "Rural"]["value"].mean()
    urban_pct = float(ai_urban["value"].values[0]) if len(ai_urban) else imp[imp["sector"] == "Urban"]["value"].mean()

    piped = _piped(df)
    ai_piped = piped[(piped["state"] == "All India") & (piped["sector"] == "All")]
    piped_pct = float(ai_piped["value"].values[0]) if len(ai_piped) else piped[piped["sector"] == "All"]["value"].mean()

    # States (exclude All India)
    states_imp = imp[
        (imp["sub_indicator"] == SUB_IMPROVED) &
        (imp["state"] != "All India") &
        (imp["sector"] == "All")
    ]

    n_states = states_imp["state"].nunique()
    above_90 = int((states_imp["value"] >= 90).sum())

    return {
        "national_improved_pct": round(national_pct, 1),
        "rural_improved_pct": round(rural_pct, 1),
        "urban_improved_pct": round(urban_pct, 1),
        "piped_water_pct": round(piped_pct, 1),
        "rural_urban_gap": round(urban_pct - rural_pct, 1),
        "states_covered": n_states,
        "states_above_90pct": above_90,
        "survey_round": 78,
    }


# ── State Analysis ──────────────────────────────────────────────────────────

def state_analysis(df: pd.DataFrame) -> list:
    """Improved water access percentage by state — Rural, Urban, All sectors."""

    imp = _improved(df)
    imp = imp[imp["state"] != "All India"]

    result = []

    for state, grp in imp.groupby("state"):
        row = {"state": state}

        for sector in ["Rural", "Urban", "All"]:
            sub = grp[grp["sector"] == sector]

            row[f"improved_{sector.lower()}_pct"] = (
                round(float(sub["value"].values[0]), 1) if len(sub) else None
            )

        result.append(row)

    # Sort by All-sector descending
    result.sort(key=lambda x: x.get("improved_all_pct") or 0, reverse=True)

    return result


# ── Sector Analysis ─────────────────────────────────────────────────────────

def sector_analysis(df: pd.DataFrame) -> list:
    """Rural vs Urban improved access and piped water — All-India."""

    imp = _improved(df)
    piped = _piped(df)

    rows = []

    for sector in ["Rural", "Urban", "All"]:
        imp_row = imp[
            (imp["state"] == "All India") &
            (imp["sector"] == sector)
        ]

        piped_row = piped[
            (piped["state"] == "All India") &
            (piped["sector"] == sector)
        ]

        rows.append({
            "sector": sector,
            "improved_pct": round(float(imp_row["value"].values[0]), 1) if len(imp_row) else None,
            "piped_pct": round(float(piped_row["value"].values[0]), 1) if len(piped_row) else None,
        })

    return rows


# ── Water Source Distribution ───────────────────────────────────────────────

def water_source_distribution(df: pd.DataFrame) -> list:
    """Compare both sub-indicators at All-India level across sectors."""

    rows = []

    for sub in [SUB_IMPROVED, SUB_PIPED]:
        sub_df = df[
            (df["state"] == "All India") &
            (df["sub_indicator"] == sub)
        ]

        for _, r in sub_df.iterrows():
            rows.append({
                "source": sub,
                "sector": r["sector"],
                "value_pct": round(float(r["value"]), 1),
            })

    return rows


# ── State Gap Analysis ──────────────────────────────────────────────────────

def state_gap_analysis(df: pd.DataFrame) -> list:
    """States with the largest Rural–Urban gap in improved water access."""

    imp = _improved(df)
    imp = imp[imp["state"] != "All India"]

    gaps = []

    for state, grp in imp.groupby("state"):
        rural = grp[grp["sector"] == "Rural"]["value"]
        urban = grp[grp["sector"] == "Urban"]["value"]

        if len(rural) and len(urban):
            gaps.append({
                "state": state,
                "rural_pct": round(float(rural.values[0]), 1),
                "urban_pct": round(float(urban.values[0]), 1),
                "gap": round(float(urban.values[0]) - float(rural.values[0]), 1),
            })

    gaps.sort(key=lambda x: x["gap"], reverse=True)

    return gaps


# ── Bottom States ────────────────────────────────────────────────────────────

def bottom_states(df: pd.DataFrame, n: int = 10) -> list:
    """States / UTs with lowest overall improved water access."""

    imp = _improved(df)
    imp = imp[
        (imp["state"] != "All India") &
        (imp["sector"] == "All")
    ]

    imp_sorted = imp.sort_values("value")

    return [
        {
            "state": row["state"],
            "improved_pct": round(float(row["value"]), 1)
        }
        for _, row in imp_sorted.head(n).iterrows()
    ]


# ── Top States ───────────────────────────────────────────────────────────────

def top_states(df: pd.DataFrame, n: int = 10) -> list:
    """States / UTs with highest overall improved water access."""

    imp = _improved(df)
    imp = imp[
        (imp["state"] != "All India") &
        (imp["sector"] == "All")
    ]

    imp_sorted = imp.sort_values("value", ascending=False)

    return [
        {
            "state": row["state"],
            "improved_pct": round(float(row["value"]), 1)
        }
        for _, row in imp_sorted.head(n).iterrows()
    ]


# ── SDG Progress ────────────────────────────────────────────────────────────

def sdg_progress(df: pd.DataFrame) -> dict:
    """Estimate progress toward SDG 6.1 (universal safe water by 2030)."""

    imp = _improved(df)

    ai = imp[
        (imp["state"] == "All India") &
        (imp["sector"] == "All")
    ]

    current_pct = float(ai["value"].values[0]) if len(ai) else imp[imp["sector"] == "All"]["value"].mean()

    baseline = 70.0
    survey_year = 2023
    target_year = 2030

    years_elapsed = survey_year - 2015
    years_remaining = target_year - survey_year

    annual_rate = (current_pct - baseline) / years_elapsed if years_elapsed else 0

    projected_2030 = min(
        current_pct + annual_rate * years_remaining,
        100.0
    )

    return {
        "current_coverage_pct": round(current_pct, 1),
        "sdg_target_pct": 100.0,
        "gap_pct": round(100.0 - current_pct, 1),
        "annual_improvement_rate": round(annual_rate, 2),
        "projected_2030_pct": round(projected_2030, 1),
        "on_track": bool(projected_2030 >= 95),
        "baseline_2015_pct": baseline,
    }


# ── Age Group Analysis ───────────────────────────────────────────────────────

def age_group_analysis(df: pd.DataFrame) -> list:
    """Improved water access by age group — All India, All sectors."""

    imp = _improved(df)
    ai = imp[
        (imp["state"] == "All India") &
        (imp["sector"] == "All")
    ]

    rows = []
    for _, r in ai.iterrows():
        age = str(r.get("age_group", "")).strip()
        if age and age.lower() not in ("all", "total", "nan", ""):
            rows.append({
                "age_group": age,
                "improved_pct": round(float(r["value"]), 1),
            })

    rows.sort(key=lambda x: x["improved_pct"], reverse=True)
    return rows


# ── Gender Analysis ──────────────────────────────────────────────────────────

def gender_analysis(df: pd.DataFrame) -> list:
    """Improved water access by gender — All India, All sectors."""

    imp = _improved(df)
    ai = imp[
        (imp["state"] == "All India") &
        (imp["sector"] == "All")
    ]

    rows = []
    for _, r in ai.iterrows():
        gender = str(r.get("gender", "")).strip()
        if gender and gender.lower() not in ("all", "total", "persons", "nan", ""):
            rows.append({
                "gender": gender,
                "improved_pct": round(float(r["value"]), 1),
            })

    rows.sort(key=lambda x: x["improved_pct"], reverse=True)
    return rows


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a GradientBoosting regressor on MIS 78th Round data.

    Features: state (encoded) + sector (encoded).
    Uses 5-fold cross-validation to estimate generalisation error.
    """

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
    import joblib

    imp = _improved(df)

    imp = imp[
        (imp["state"] != "All India") &
        (imp["sector"].isin(["Rural", "Urban"]))
    ].copy()

    imp = imp.dropna(subset=["state", "sector", "value"])

    if len(imp) < 5:
        raise ValueError(
            f"Not enough records to train the model: {len(imp)}"
        )

    le_state = LabelEncoder()
    le_sector = LabelEncoder()

    imp["state_enc"]  = le_state.fit_transform(imp["state"])
    imp["sector_enc"] = le_sector.fit_transform(imp["sector"])

    X = imp[["state_enc", "sector_enc"]].values
    y = imp["value"].values

    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )

    cv_mae = -cv_scores.mean()

    model.fit(X, y)

    train_mae = mean_absolute_error(y, model.predict(X))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"state": le_state, "sector": le_sector}, ENCODER_PATH)

    print(
        f"[ML] Model trained — "
        f"5-Fold MAE: {cv_mae:.2f} pp | "
        f"Train MAE: {train_mae:.2f} pp"
    )

    return {
        "loo_mae_pp":   round(float(cv_mae),      2),
        "train_mae_pp": round(float(train_mae),   2),
        "n_samples":    len(y),
        "features":     ["state", "sector"],
        "model":        "GradientBoostingRegressor",
        "note":         "MAE = mean absolute error in percentage points (5-fold CV)"
    }


# ── Prediction ──────────────────────────────────────────────────────────────

def predict_access(record: dict) -> dict:
    """
    Predict improved water access percentage for a given state + sector.

    record keys:
        state (str)
        sector (str — 'Rural' or 'Urban')
    """

    import joblib

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)

    le_state = encoders["state"]
    le_sector = encoders["sector"]

    try:
        state_enc = le_state.transform([record["state"]])[0]
    except ValueError:
        state_enc = int(
            le_state.transform([le_state.classes_[0]])[0]
        )

    sector_enc = le_sector.transform([record["sector"]])[0]

    X = [[state_enc, sector_enc]]

    predicted = float(model.predict(X)[0])
    predicted = max(0.0, min(100.0, predicted))

    return {
        "state": record["state"],
        "sector": record["sector"],
        "predicted_improved_pct": round(predicted, 1),
        "interpretation": (
            "High access — above 90%" if predicted >= 90 else
            "Moderate access — 70-90%" if predicted >= 70 else
            "Low access — below 70% (priority intervention needed)"
        ),
    }
