
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "spotify_modeling_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Audio features
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

# Audio features + release year
GROUP_B_FEATURES = GROUP_A_FEATURES + ["year"]

TARGET = "popularity"


# ============================================================
# 2. LOAD DATA AND REPRODUCE THE TRAIN-TEST SPLIT
# ============================================================

print("=" * 70)
print("OLS STATISTICAL INFERENCE")
print("=" * 70)

data = pd.read_csv(INPUT_FILE)

print("\n[1] Dataset")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")

# Reproduce exactly the same split as model_training.py
indices = np.arange(len(data))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_data = data.iloc[train_indices].copy()

print("\n[2] Training data")
print(f"Training observations: {len(train_data)}")
print(f"Testing observations: {len(test_indices)}")


# ============================================================
# 3. OLS INFERENCE FUNCTION
# ============================================================

def run_ols_inference(data, features, group_name):

    print("\n" + "=" * 70)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 70)

    # Select predictors and target
    X = data[features].copy()
    y = data[TARGET].copy()

    # Treat musical key as a categorical variable.
    # One category is dropped to avoid the dummy variable trap.
    X = pd.get_dummies(
        X,
        columns=["key"],
        drop_first=True,
        dtype=float
    )

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Median imputation using training data only
    medians = X.median()
    X = X.fillna(medians)

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    # Fit OLS with HC3 heteroscedasticity-robust standard errors
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")

    # --------------------------------------------------------
    # Coefficient table
    # --------------------------------------------------------

    confidence_intervals = results.conf_int()

    coefficient_table = pd.DataFrame({
        "Feature_Group": group_name,
        "Variable": results.params.index,
        "Coefficient": results.params.values,
        "Robust_SE": results.bse.values,
        "t_statistic": results.tvalues.values,
        "p_value": results.pvalues.values,
        "CI_Lower_95": confidence_intervals.iloc[:, 0].values,
        "CI_Upper_95": confidence_intervals.iloc[:, 1].values
    })

    # Significance labels
    def significance(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "Not significant"

    coefficient_table["Significance"] = (
        coefficient_table["p_value"].apply(significance)
    )

    # --------------------------------------------------------
    # Robust joint significance test
    # H0: all slope coefficients = 0
    # --------------------------------------------------------

    number_of_parameters = len(results.params)

    restriction_matrix = np.zeros(
        (number_of_parameters - 1, number_of_parameters)
    )

    restriction_matrix[:, 1:] = np.eye(
        number_of_parameters - 1
    )

    wald_result = results.wald_test(
        restriction_matrix,
        use_f=True
    )

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    summary = {
        "Feature_Group": group_name,
        "Observations": int(results.nobs),
        "Number_of_Predictors_Including_Dummies": number_of_parameters - 1,
        "R_squared": results.rsquared,
        "Adjusted_R_squared": results.rsquared_adj,
        "Robust_F_statistic": float(np.asarray(wald_result.fvalue).squeeze()),
        "Robust_F_p_value": float(np.asarray(wald_result.pvalue).squeeze())
    }

    print("\nModel statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nTop coefficient results:")
    print(
        coefficient_table
        .sort_values("p_value")
        .head(20)
        .to_string(index=False)
    )

    return coefficient_table, summary


# ============================================================
# 4. RUN BOTH FEATURE GROUPS
# ============================================================

all_coefficients = []
all_summaries = []

for group_name, features in [
    ("Group_A_Audio", GROUP_A_FEATURES),
    ("Group_B_Audio_Year", GROUP_B_FEATURES)
]:

    coefficients, summary = run_ols_inference(
        train_data,
        features,
        group_name
    )

    all_coefficients.append(coefficients)
    all_summaries.append(summary)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

coefficient_results = pd.concat(
    all_coefficients,
    ignore_index=True
)

summary_results = pd.DataFrame(all_summaries)

coefficient_file = OUTPUT_DIR / "ols_coefficient_results.csv"
summary_file = OUTPUT_DIR / "ols_model_statistics.csv"

coefficient_results.to_csv(
    coefficient_file,
    index=False,
    encoding="utf-8-sig"
)

summary_results.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

# Save a readable text report
report_file = OUTPUT_DIR / "ols_inference_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("OLS STATISTICAL INFERENCE REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("Standard errors: HC3 heteroscedasticity-robust\n")
    f.write("Training split: 80%, random_state=42\n")
    f.write("Categorical variable: key (one-hot encoded)\n\n")

    f.write("MODEL STATISTICS\n")
    f.write(summary_results.to_string(index=False))
    f.write("\n\n")

    f.write("COEFFICIENT RESULTS\n")
    f.write(coefficient_results.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 70)
print("OLS INFERENCE COMPLETED")
print("=" * 70)

print(f"Saved: {coefficient_file}")
print(f"Saved: {summary_file}")
print(f"Saved: {report_file}")
