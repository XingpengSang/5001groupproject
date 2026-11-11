
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "spotify_modeling_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
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
    "mode",
    "year"
]

TARGET = "popularity"


# ============================================================
# 2. LOAD DATA AND SPLIT
# ============================================================

print("=" * 70)
print("MODEL DIAGNOSTICS")
print("=" * 70)

data = pd.read_csv(INPUT_FILE)

train_indices, test_indices = train_test_split(
    np.arange(len(data)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_data = data.iloc[train_indices].copy()
test_data = data.iloc[test_indices].copy()

print("\n[1] Dataset")
print(f"Total observations: {len(data)}")
print(f"Training observations: {len(train_data)}")
print(f"Testing observations: {len(test_data)}")


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

X_train = train_data[FEATURES].copy()
X_test = test_data[FEATURES].copy()

y_train = train_data[TARGET].copy()
y_test = test_data[TARGET].copy()

# Convert key into categorical dummy variables
X_train = pd.get_dummies(
    X_train,
    columns=["key"],
    drop_first=True,
    dtype=float
)

X_test = pd.get_dummies(
    X_test,
    columns=["key"],
    drop_first=True,
    dtype=float
)

# Ensure test columns match training columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Numeric conversion
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Impute using training medians only
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Add intercept
X_train = sm.add_constant(X_train, has_constant="add")
X_test = sm.add_constant(X_test, has_constant="add")

# Align test columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# 4. FIT OLS MODEL
# ============================================================

print("\n[2] Fitting OLS model...")

model = sm.OLS(y_train, X_train)
results = model.fit()

# Predictions
train_predictions = results.predict(X_train)
test_predictions = results.predict(X_test)

train_residuals = y_train - train_predictions
test_residuals = y_test - test_predictions

print("OLS model fitted successfully.")


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted, dataset_name):

    return {
        "Dataset": dataset_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


metrics = pd.DataFrame([
    calculate_metrics(
        y_train, train_predictions, "Training"
    ),
    calculate_metrics(
        y_test, test_predictions, "Testing"
    )
])

print("\n[3] Model performance")
print(metrics.to_string(index=False))


# ============================================================
# 6. BREUSCH-PAGAN HETEROSCEDASTICITY TEST
# ============================================================

print("\n[4] Breusch-Pagan Test")

bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
    train_residuals,
    X_train
)

bp_results = {
    "LM Statistic": bp_lm,
    "LM p-value": bp_lm_pvalue,
    "F Statistic": bp_f,
    "F p-value": bp_f_pvalue
}

for key, value in bp_results.items():
    print(f"{key}: {value:.8g}")

if bp_lm_pvalue < 0.05:
    print("Conclusion: Evidence of heteroscedasticity.")
else:
    print("Conclusion: No significant evidence of heteroscedasticity.")


# ============================================================
# 7. JARQUE-BERA NORMALITY TEST
# ============================================================

print("\n[5] Jarque-Bera Test")

jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(
    train_residuals
)

jb_results = {
    "JB Statistic": jb_stat,
    "JB p-value": jb_pvalue,
    "Skewness": skewness,
    "Kurtosis": kurtosis
}

for key, value in jb_results.items():
    print(f"{key}: {value:.8g}")

if jb_pvalue < 0.05:
    print("Conclusion: Residuals deviate from normality.")
else:
    print("Conclusion: No significant evidence against normality.")

print("Note: With a very large sample, this test is highly sensitive.")


# ============================================================
# 8. RESIDUAL DIAGNOSTIC PLOTS
# ============================================================

print("\n[6] Creating diagnostic plots...")

# Plot 1: Residuals vs Fitted
plt.figure(figsize=(9, 6))
plt.scatter(
    train_predictions,
    train_residuals,
    alpha=0.15,
    s=8
)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted Popularity")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values - Training Set")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "residuals_vs_fitted.png",
    dpi=300
)
plt.close()


# Plot 2: Residual distribution
plt.figure(figsize=(9, 6))
plt.hist(
    train_residuals,
    bins=60,
    edgecolor="black",
    alpha=0.75
)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution - Training Set")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300
)
plt.close()


# Plot 3: Q-Q plot
fig = sm.qqplot(
    train_residuals,
    line="45",
    fit=True
)
plt.title("Q-Q Plot of OLS Residuals")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "residual_qq_plot.png",
    dpi=300
)
plt.close()


# Plot 4: Actual vs Predicted - Test Set
plt.figure(figsize=(8, 8))
plt.scatter(
    y_test,
    test_predictions,
    alpha=0.2,
    s=8
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Popularity - Test Set")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)
plt.close()


# Plot 5: Test residuals vs predicted
plt.figure(figsize=(9, 6))
plt.scatter(
    test_predictions,
    test_residuals,
    alpha=0.15,
    s=8
)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Popularity")
plt.ylabel("Prediction Residuals")
plt.title("Test Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "test_residuals_vs_predicted.png",
    dpi=300
)
plt.close()


# ============================================================
# 9. SAVE RESULTS
# ============================================================

metrics.to_csv(
    OUTPUT_DIR / "diagnostic_model_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

test_predictions_output = pd.DataFrame({
    "Actual_Popularity": y_test.values,
    "Predicted_Popularity": test_predictions,
    "Residual": test_residuals.values
})

test_predictions_output.to_csv(
    OUTPUT_DIR / "diagnostic_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

test_residuals_output = pd.DataFrame({
    "Actual_Popularity": y_test.values,
    "Predicted_Popularity": test_predictions,
    "Residual": test_residuals.values
})

bp_output = pd.DataFrame([bp_results])
jb_output = pd.DataFrame([jb_results])

bp_output.to_csv(
    OUTPUT_DIR / "breusch_pagan_test.csv",
    index=False,
    encoding="utf-8-sig"
)

jb_output.to_csv(
    OUTPUT_DIR / "jarque_bera_test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("MODEL DIAGNOSTICS COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("1. residuals_vs_fitted.png")
print("2. residual_distribution.png")
print("3. residual_qq_plot.png")
print("4. actual_vs_predicted.png")
print("5. test_residuals_vs_predicted.png")
print("6. diagnostic_model_metrics.csv")
print("7. diagnostic_test_predictions.csv")
print("8. breusch_pagan_test.csv")
print("9. jarque_bera_test.csv")

print(f"\nOutput directory: {OUTPUT_DIR}")
