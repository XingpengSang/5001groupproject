
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
OUTPUT_DIR = PROJECT_DIR / "output"

DATA_FILE = OUTPUT_DIR / "spotify_modeling_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "popularity"

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND TRAIN-TEST SPLIT
# ============================================================

print("=" * 70)
print("NONLINEAR MODEL COMPARISON")
print("=" * 70)

data = pd.read_csv(DATA_FILE)

print("\n[1] Dataset")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")

required_columns = list(
    set(GROUP_B_FEATURES + [TARGET])
)

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

train_indices, test_indices = train_test_split(
    np.arange(len(data)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_data = data.iloc[train_indices].copy()
test_data = data.iloc[test_indices].copy()

print("\n[2] Train-test split")
print(f"Training observations: {len(train_data)}")
print(f"Testing observations: {len(test_data)}")
print(f"Random state: {RANDOM_STATE}")


# ============================================================
# 3. FEATURE PREPROCESSING
# ============================================================

def prepare_features(train_df, test_df, features):
    """
    Prepare numerical features and one-hot encode key.

    The encoding is fitted using training data only.
    Test data is aligned to the training feature columns.
    """

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    # Separate numeric and categorical variables
    numeric_features = [
        col for col in features
        if col != "key"
    ]

    # Convert numeric columns and impute missing values
    for col in numeric_features:

        X_train[col] = pd.to_numeric(
            X_train[col],
            errors="coerce"
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            errors="coerce"
        )

        median_value = X_train[col].median()

        if pd.isna(median_value):
            median_value = 0

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

    # Treat key as a categorical variable
    X_train["key"] = X_train["key"].astype("string")
    X_test["key"] = X_test["key"].astype("string")

    key_mode = X_train["key"].mode()

    if len(key_mode) > 0:
        key_fill_value = key_mode.iloc[0]
    else:
        key_fill_value = "Unknown"

    X_train["key"] = X_train["key"].fillna(key_fill_value)
    X_test["key"] = X_test["key"].fillna(key_fill_value)

    # One-hot encoding
    X_train = pd.get_dummies(
        X_train,
        columns=["key"],
        prefix="key",
        dtype=float
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["key"],
        prefix="key",
        dtype=float
    )

    # Align test columns with training columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Ensure consistent numeric data types
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Final validation
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing features do not match.")

    if X_train.isnull().any().any():
        raise ValueError("Missing values remain in training features.")

    if X_test.isnull().any().any():
        raise ValueError("Missing values remain in testing features.")

    return X_train, X_test


# ============================================================
# 4. DEFINE MODELS
# ============================================================

def create_models():

    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
prediction_records = []
fitted_models = {}
feature_names_by_group = {}

y_train = train_data[TARGET].astype(float)
y_test = test_data[TARGET].astype(float)

feature_groups = {
    "Group_A_Audio": GROUP_A_FEATURES,
    "Group_B_Audio_Year": GROUP_B_FEATURES
}

for group_name, features in feature_groups.items():

    print("\n" + "=" * 70)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 70)

    # Prepare features independently for each group
    X_train, X_test = prepare_features(
        train_data,
        test_data,
        features
    )

    feature_names = X_train.columns.tolist()

    feature_names_by_group[group_name] = feature_names

    print(f"Original predictors: {len(features)}")
    print(f"Encoded predictors: {len(feature_names)}")

    # Store feature matrices for this group
    for model_name, model in create_models().items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        r2 = r2_score(y_test, y_pred)

        results.append({
            "Feature_Group": group_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        prediction_records.append(
            pd.DataFrame({
                "Feature_Group": group_name,
                "Model": model_name,
                "Year": test_data["year"].values,
                "Actual_Popularity": y_test.values,
                "Predicted_Popularity": y_pred,
                "Residual": y_test.values - y_pred,
                "Absolute_Error": np.abs(
                    y_test.values - y_pred
                )
            })
        )

        fitted_models[(group_name, model_name)] = model

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")


# ============================================================
# 6. SAVE MODEL EVALUATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE",
    ascending=True
)

print("\n" + "=" * 70)
print("NONLINEAR MODEL EVALUATION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "nonlinear_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

all_predictions = pd.concat(
    prediction_records,
    ignore_index=True
)

all_predictions.to_csv(
    OUTPUT_DIR / "nonlinear_model_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. ERROR ANALYSIS BY POPULARITY GROUP
# ============================================================

print("\n[3] Error analysis by popularity group")

all_predictions["Popularity_Group"] = pd.cut(
    all_predictions["Actual_Popularity"],
    bins=[-0.001, 20, 40, 60, 80, 100],
    labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
    include_lowest=True
)

group_error_results = (
    all_predictions
    .groupby(
        ["Feature_Group", "Model", "Popularity_Group"],
        observed=False
    )
    .agg(
        Sample_Size=("Actual_Popularity", "size"),
        Mean_Actual=("Actual_Popularity", "mean"),
        Mean_Predicted=("Predicted_Popularity", "mean"),
        MAE=("Absolute_Error", "mean"),
        RMSE=(
            "Residual",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        Mean_Error=("Residual", "mean")
    )
    .reset_index()
)

group_error_results["Prediction_Bias"] = np.where(
    group_error_results["Mean_Error"] < 0,
    "Overprediction",
    "Underprediction"
)

group_error_results.to_csv(
    OUTPUT_DIR / "nonlinear_error_by_popularity.csv",
    index=False,
    encoding="utf-8-sig"
)

print(group_error_results.to_string(index=False))


# ============================================================
# 8. MODEL PERFORMANCE PLOT
# ============================================================

plot_data = results_df.copy()

plot_labels = (
    plot_data["Feature_Group"]
    + "\n"
    + plot_data["Model"]
)

x = np.arange(len(plot_data))

plt.figure(figsize=(11, 6))

plt.bar(x, plot_data["RMSE"])

plt.xticks(
    x,
    plot_labels,
    rotation=20,
    ha="right"
)

plt.ylabel("Test RMSE")
plt.title("Nonlinear Model Comparison - Test RMSE")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "nonlinear_model_rmse_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. HIGH-POPULARITY PREDICTION BIAS
# ============================================================

high_popularity = group_error_results[
    group_error_results["Popularity_Group"].isin(
        ["60-80", "80-100"]
    )
].copy()

high_popularity["Label"] = (
    high_popularity["Feature_Group"]
    + " | "
    + high_popularity["Model"]
    + " | "
    + high_popularity["Popularity_Group"].astype(str)
)

plt.figure(figsize=(13, 7))

plt.bar(
    high_popularity["Label"],
    high_popularity["Mean_Error"]
)

plt.axhline(0, linestyle="--")

plt.ylabel("Mean Error (Actual - Predicted)")
plt.title("Prediction Bias for High-Popularity Songs")

plt.xticks(rotation=35, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "nonlinear_high_popularity_bias.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n[4] Extracting Random Forest feature importance")

importance_records = []

for group_name in feature_groups:

    model = fitted_models[
        (group_name, "Random Forest")
    ]

    importances = model.feature_importances_

    feature_names = feature_names_by_group[group_name]

    print(f"\n{group_name}")
    print(f"Feature names: {len(feature_names)}")
    print(f"Feature importances: {len(importances)}")
    print(f"Model input features: {model.n_features_in_}")

    # Strict dimension validation
    if not (
        len(feature_names)
        == len(importances)
        == model.n_features_in_
    ):
        raise ValueError(
            f"Feature dimension mismatch in {group_name}."
        )

    importance_df = pd.DataFrame({
        "Feature_Group": group_name,
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_records.append(importance_df)

    print(f"\nTop 10 features: {group_name}")
    print(importance_df.head(10).to_string(index=False))


importance_results = pd.concat(
    importance_records,
    ignore_index=True
)

importance_results.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. PLOT TOP 15 FEATURE IMPORTANCE
# ============================================================

group_b_importance = (
    importance_results[
        importance_results["Feature_Group"]
        == "Group_B_Audio_Year"
    ]
    .nlargest(15, "Importance")
    .sort_values("Importance")
)

plt.figure(figsize=(10, 7))

plt.barh(
    group_b_importance["Feature"],
    group_b_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Random Forest - Top 15 Feature Importance (Audio + Year)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_top15_feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("NONLINEAR MODEL COMPARISON COMPLETED")
print("=" * 70)

print("\nGenerated files:")

output_files = [
    "nonlinear_model_results.csv",
    "nonlinear_model_predictions.csv",
    "nonlinear_error_by_popularity.csv",
    "random_forest_feature_importance.csv",
    "nonlinear_model_rmse_comparison.png",
    "nonlinear_high_popularity_bias.png",
    "random_forest_top15_feature_importance.png"
]

for i, filename in enumerate(output_files, start=1):
    print(f"{i}. {filename}")

print(f"\nOutput directory: {OUTPUT_DIR}")
