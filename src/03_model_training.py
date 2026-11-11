
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. Path configuration
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "spotify_modeling_data.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load processed dataset
# ============================================================

print("=" * 70)
print("SPOTIFY MODEL TRAINING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

target = "popularity"

print("\n[1] Dataset")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# 3. Define feature groups
# ============================================================

group_a_features = [
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

group_b_features = group_a_features + ["year"]

feature_groups = {
    "Group_A_Audio": group_a_features,
    "Group_B_Audio_Year": group_b_features
}

# Check required columns
required_columns = group_b_features + [target]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# ============================================================
# 4. Train-test split
# ============================================================

print("\n[2] Train-test split")

# Generate indices once so both groups use exactly
# the same training and testing observations.
all_indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42
)

y_train = df.iloc[train_indices][target]
y_test = df.iloc[test_indices][target]

print(f"Training observations: {len(train_indices)}")
print(f"Testing observations: {len(test_indices)}")
print("Random state: 42")

# ============================================================
# 5. Define preprocessing pipeline
# ============================================================

def build_preprocessor(features):

    # Key is categorical, not continuous.
    categorical_features = ["key"]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


# ============================================================
# 6. Define evaluation function
# ============================================================

def evaluate_model(model_name, group_name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "Feature_Group": group_name,
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 7. Model configuration
# ============================================================

alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
best_parameters = []
all_predictions = []

# ============================================================
# 8. Train models for each feature group
# ============================================================

for group_name, features in feature_groups.items():

    print("\n" + "=" * 70)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 70)

    X_train = df.iloc[train_indices][features].copy()
    X_test = df.iloc[test_indices][features].copy()

    preprocessor = build_preprocessor(features)

    # --------------------------------------------------------
    # Model 1: Mean Baseline
    # --------------------------------------------------------

    print("\n[3] Training Mean Baseline...")

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)

    results.append(
        evaluate_model(
            "Mean Baseline",
            group_name,
            y_test,
            baseline_pred
        )
    )

    # --------------------------------------------------------
    # Model 2: Ordinary Least Squares
    # --------------------------------------------------------

    print("\n[4] Training OLS...")

    ols_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    ols_pipeline.fit(X_train, y_train)
    ols_pred = ols_pipeline.predict(X_test)

    results.append(
        evaluate_model(
            "OLS",
            group_name,
            y_test,
            ols_pred
        )
    )

    # --------------------------------------------------------
    # Model 3: Ridge Regression with 5-fold CV
    # --------------------------------------------------------

    print("\n[5] Training Ridge Regression with 5-fold CV...")

    ridge_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("model", Ridge())
    ])

    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid={
            "model__alpha": alpha_values
        },
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )

    ridge_search.fit(X_train, y_train)
    ridge_pred = ridge_search.predict(X_test)

    results.append(
        evaluate_model(
            "Ridge",
            group_name,
            y_test,
            ridge_pred
        )
    )

    best_parameters.append({
        "Feature_Group": group_name,
        "Model": "Ridge",
        "Best_Alpha": ridge_search.best_params_["model__alpha"],
        "Best_CV_RMSE": -ridge_search.best_score_
    })

    print(f"Best Ridge alpha: {ridge_search.best_params_['model__alpha']}")
    print(f"Best CV RMSE: {-ridge_search.best_score_:.4f}")

    # --------------------------------------------------------
    # Model 4: Lasso Regression with 5-fold CV
    # --------------------------------------------------------

    print("\n[6] Training Lasso Regression with 5-fold CV...")

    lasso_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(features)),
        ("model", Lasso(max_iter=10000, tol=0.001))
    ])

    lasso_search = GridSearchCV(
        estimator=lasso_pipeline,
        param_grid={
            "model__alpha": alpha_values
        },
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )

    lasso_search.fit(X_train, y_train)
    lasso_pred = lasso_search.predict(X_test)

    results.append(
        evaluate_model(
            "Lasso",
            group_name,
            y_test,
            lasso_pred
        )
    )

    best_parameters.append({
        "Feature_Group": group_name,
        "Model": "Lasso",
        "Best_Alpha": lasso_search.best_params_["model__alpha"],
        "Best_CV_RMSE": -lasso_search.best_score_
    })

    print(f"Best Lasso alpha: {lasso_search.best_params_['model__alpha']}")
    print(f"Best CV RMSE: {-lasso_search.best_score_:.4f}")

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Actual_Popularity": y_test.to_numpy(),
        "Mean_Baseline": baseline_pred,
        "OLS": ols_pred,
        "Ridge": ridge_pred,
        "Lasso": lasso_pred
    })

    prediction_df.insert(0, "Original_Index", test_indices)
    prediction_df.insert(0, "Feature_Group", group_name)

    all_predictions.append(prediction_df)

# ============================================================
# 9. Save evaluation results
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df.sort_values(
        ["Feature_Group", "RMSE"]
    ).to_string(index=False)
)

results_df.to_csv(
    OUTPUT_DIR / "model_evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 10. Save best hyperparameters
# ============================================================

params_df = pd.DataFrame(best_parameters)

print("\nBest hyperparameters:")
print(params_df.to_string(index=False))

params_df.to_csv(
    OUTPUT_DIR / "best_model_parameters.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 11. Save test predictions
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    OUTPUT_DIR / "model_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 12. Final summary
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print(f"Results saved to: {OUTPUT_DIR}")
print("1. model_evaluation_results.csv")
print("2. best_model_parameters.csv")
print("3. model_test_predictions.csv")
