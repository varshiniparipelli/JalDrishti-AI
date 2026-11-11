"""
JalDrishti-AI — local validation script
Run with: python validate.py
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

passed = []
failed = []

def chk(label, fn):
    try:
        result = fn()
        passed.append(label)
        return result
    except Exception as e:
        failed.append((label, str(e)))
        return None

# ── Dataset ──────────────────────────────────────────────────────────
from analysis import (
    load_data, compute_kpis, state_analysis, sector_analysis,
    water_source_distribution, state_gap_analysis, bottom_states,
    top_states, sdg_progress, train_model, predict_access,
    age_group_analysis, gender_analysis
)

df = chk("load_data", load_data)
if df is None:
    print("FATAL: dataset could not be loaded")
    sys.exit(1)

chk("compute_kpis",               lambda: compute_kpis(df))
chk("state_analysis",             lambda: state_analysis(df))
chk("sector_analysis",            lambda: sector_analysis(df))
chk("water_source_distribution",  lambda: water_source_distribution(df))
chk("state_gap_analysis",         lambda: state_gap_analysis(df))
chk("bottom_states",              lambda: bottom_states(df, 5))
chk("top_states",                 lambda: top_states(df, 5))
chk("sdg_progress",               lambda: sdg_progress(df))
chk("age_group_analysis",         lambda: age_group_analysis(df))
chk("gender_analysis",            lambda: gender_analysis(df))

train_result = chk("train_model", lambda: train_model(df))
chk("predict_access_rural",       lambda: predict_access({"state": "Bihar", "sector": "Rural"}))
chk("predict_access_urban",       lambda: predict_access({"state": "Delhi", "sector": "Urban"}))
chk("predict_access_kerala",      lambda: predict_access({"state": "Kerala", "sector": "Rural"}))

# ── KPI spot check ───────────────────────────────────────────────────
kpis = compute_kpis(df)
def check_kpis():
    assert 90 <= kpis["national_improved_pct"] <= 100, f"national KPI out of range: {kpis}"
    assert kpis["states_covered"] >= 30, f"too few states: {kpis}"
chk("kpi_sanity", check_kpis)

# ── Model accuracy ───────────────────────────────────────────────────
if train_result:
    def check_accuracy():
        assert train_result["loo_mae_pp"] < 15, f"CV MAE too high: {train_result['loo_mae_pp']}"
        assert train_result["n_samples"] >= 50, f"too few samples: {train_result['n_samples']}"
    chk("model_accuracy_sanity", check_accuracy)

# ── Report ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("JalDrishti-AI Local Validation")
print("=" * 60)
print(f"PASSED: {len(passed)}")
for p in passed:
    print(f"  OK  {p}")
print()
if failed:
    print(f"FAILED: {len(failed)}")
    for label, err in failed:
        print(f"  FAIL {label}: {err}")
else:
    print("All checks passed.")

if train_result:
    print()
    print("Model stats:")
    print(f"  CV MAE  : {train_result['loo_mae_pp']} pp")
    print(f"  Train MAE: {train_result['train_mae_pp']} pp")
    print(f"  Samples : {train_result['n_samples']}")

print()
kpis = compute_kpis(df)
print("National KPIs:")
for k, v in kpis.items():
    print(f"  {k}: {v}")

sys.exit(0 if not failed else 1)
