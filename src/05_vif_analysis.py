
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "spotify_modeling_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

GROUP_A_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "tempo",
    "log_duration",
    "explicit",
    "key",
    "mode"
]

GROUP_B_FEATURES = GROUP_A_FEATURES + ["year"]


# ============================================================
# 2. LOAD DATA AND REPRODUCE TRAINING SPLIT
# ============================================================

print("=" * 70)
print("MULTICOLLINEARITY ANALYSIS - VIF")
print("=" * 70)

data = pd.read_csv(INPUT_FILE)

indices = np.arange(len(data))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_data = data.iloc[train_indices].copy()

print("\n[1] Dataset")
print(f"Total observations: {len(data)}")
print(f"Training observations: {len(train_data)}")


# ============================================================
# 3. VIF CALCULATION FUNCTION
# ============================================================

def calculate_vif(data, features, group_name):

    print("\n" + "=" * 70)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 70)

    X = data[features].copy()

    # Treat musical key as categorical
    X = pd.get_dummies(
        X,
        columns=["key"],
        drop_first=True,
        dtype=float
    )

    # Ensure numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    # Median imputation
    X = X.fillna(X.median())

    # Remove constant or zero-variance columns if any
    X = X.loc[:, X.nunique() > 1]

    # Add intercept
    X = sm_add_constant(X)

    # Calculate VIF for each predictor
    vif_results = []

    for i, column in enumerate(X.columns):

        if column == "const":
            continue

        vif_value = variance_inflation_factor(
            X.values,
            i
        )

        vif_results.append({
            "Feature_Group": group_name,
            "Variable": column,
            "VIF": vif_value
        })

    vif_table = pd.DataFrame(vif_results)

    # Classification
    def classify_vif(value):
        if value >= 10:
            return "High multicollinearity"
        elif value >= 5:
            return "Moderate multicollinearity"
        else:
            return "Low multicollinearity"

    vif_table["Interpretation"] = (
        vif_table["VIF"].apply(classify_vif)
    )

    vif_table = vif_table.sort_values(
        "VIF",
        ascending=False
    )

    print("\nVIF Results:")
    print(vif_table.to_string(index=False))

    return vif_table


def sm_add_constant(X):
    # Add intercept without importing the full statsmodels API
    import statsmodels.api as sm
    return sm.add_constant(X, has_constant="add")


# ============================================================
# 4. RUN BOTH FEATURE GROUPS
# ============================================================

vif_a = calculate_vif(
    train_data,
    GROUP_A_FEATURES,
    "Group_A_Audio"
)

vif_b = calculate_vif(
    train_data,
    GROUP_B_FEATURES,
    "Group_B_Audio_Year"
)

all_vif = pd.concat(
    [vif_a, vif_b],
    ignore_index=True
)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

output_file = OUTPUT_DIR / "vif_results.csv"

all_vif.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# Save a comparison summary
summary = (
    all_vif.groupby("Feature_Group")
    .agg(
        Maximum_VIF=("VIF", "max"),
        Mean_VIF=("VIF", "mean"),
        Variables_VIF_Above_5=("VIF", lambda x: (x >= 5).sum()),
        Variables_VIF_Above_10=("VIF", lambda x: (x >= 10).sum())
    )
    .reset_index()
)

summary_file = OUTPUT_DIR / "vif_summary.csv"

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("VIF ANALYSIS COMPLETED")
print("=" * 70)

print("\nSummary:")
print(summary.to_string(index=False))

print(f"\nSaved: {output_file}")
print(f"Saved: {summary_file}")
