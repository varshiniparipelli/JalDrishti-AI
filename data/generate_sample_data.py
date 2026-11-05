"""
JalDrishti-AI — Sample Data Generator
Generates a representative sample dataset mirroring the
MIS 78th Round (Improved Source of Drinking Water) structure.
Run this script once to create mis_78th_round_drinking_water.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu & Kashmir", "Goa", "Puducherry"
]

IMPROVED_SOURCES = [
    "Piped water into dwelling",
    "Piped water to yard/plot",
    "Public tap/standpipe",
    "Tube well/borehole",
    "Protected dug well",
    "Protected spring",
    "Rainwater collection"
]

UNIMPROVED_SOURCES = [
    "Unprotected dug well",
    "Unprotected spring",
    "Surface water",
    "Tanker-truck/cart",
    "Bottled water (only)"
]

SECTORS = ["Rural", "Urban"]
RELIGION = ["Hindu", "Muslim", "Christian", "Sikh", "Others"]
SOCIAL_GROUP = ["ST", "SC", "OBC", "General"]
HOUSEHOLD_SIZE = list(range(1, 10))
CLEAN_FUEL = ["LPG/PNG", "Electricity", "Biogas", "None"]

n = 5000
records = []

for _ in range(n):
    state = np.random.choice(STATES)
    sector = np.random.choice(SECTORS, p=[0.65, 0.35])
    social_group = np.random.choice(SOCIAL_GROUP, p=[0.12, 0.18, 0.43, 0.27])
    religion = np.random.choice(RELIGION, p=[0.79, 0.14, 0.02, 0.02, 0.03])

    # Urban households more likely to have improved water
    if sector == "Urban":
        improved_prob = np.random.uniform(0.82, 0.98)
    elif social_group in ["ST", "SC"]:
        improved_prob = np.random.uniform(0.45, 0.72)
    else:
        improved_prob = np.random.uniform(0.58, 0.88)

    has_improved = np.random.random() < improved_prob
    if has_improved:
        water_source = np.random.choice(IMPROVED_SOURCES)
    else:
        water_source = np.random.choice(UNIMPROVED_SOURCES)

    # Clean cooking fuel access
    if sector == "Urban":
        fuel_prob = np.random.uniform(0.75, 0.95)
    else:
        fuel_prob = np.random.uniform(0.30, 0.65)
    has_clean_fuel = np.random.random() < fuel_prob
    cooking_fuel = np.random.choice(CLEAN_FUEL[:3]) if has_clean_fuel else "None"

    # Migration
    migration_member = np.random.choice([0, 1], p=[0.72, 0.28])

    hh_size = np.random.choice(HOUSEHOLD_SIZE, p=[0.05,0.10,0.15,0.20,0.20,0.12,0.09,0.05,0.04])

    # Water collection time (minutes, 0 = on premises)
    if water_source in ["Piped water into dwelling"]:
        water_time = 0
    elif water_source in ["Piped water to yard/plot"]:
        water_time = np.random.randint(0, 5)
    else:
        water_time = np.random.randint(5, 120)

    records.append({
        "hhid": f"MIS78-{_ + 1:05d}",
        "state": state,
        "sector": sector,
        "social_group": social_group,
        "religion": religion,
        "household_size": hh_size,
        "main_drinking_water_source": water_source,
        "improved_source": int(has_improved),
        "water_collection_time_min": water_time,
        "water_on_premises": int(water_time == 0),
        "clean_cooking_fuel": cooking_fuel,
        "has_clean_fuel": int(has_clean_fuel),
        "migration_last_year": migration_member,
        "year": 2023,
        "survey_round": 78
    })

df = pd.DataFrame(records)
out_path = os.path.join(os.path.dirname(__file__), "mis_78th_round_drinking_water.csv")
df.to_csv(out_path, index=False)
print(f"[OK] Sample dataset created: {out_path}  ({len(df)} rows)")
