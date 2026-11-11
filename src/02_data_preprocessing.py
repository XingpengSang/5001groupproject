
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. Path configuration
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
DATA_DIR = PROJECT_DIR / "Dataset"
OUTPUT_DIR = PROJECT_DIR / "output"

INPUT_FILE = DATA_DIR / "data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load original dataset
# ============================================================

print("=" * 65)
print("SPOTIFY DATA PREPROCESSING")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print("\n[1] Original dataset")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Keep the original dataset unchanged.
data = df.copy()

# ============================================================
# 3. Select variables
# ============================================================

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "tempo"
]

additional_features = [
    "duration_ms",
    "explicit",
    "key",
    "mode"
]

target = "popularity"
year_feature = "year"

required_columns = (
    audio_features
    + additional_features
    + [target, year_feature]
)

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep only variables required for modeling.
data = data[required_columns].copy()

print("\n[2] Selected variables")
print(data.columns.tolist())

# ============================================================
# 4. Handle invalid tempo values
# ============================================================

print("\n[3] Handle invalid tempo values")

invalid_tempo_count = (data["tempo"] <= 0).sum()

print(f"Non-positive tempo values: {invalid_tempo_count}")

# Replace non-positive tempo with NaN.
# Median imputation will be performed later inside
# the machine learning pipeline.
data.loc[data["tempo"] <= 0, "tempo"] = np.nan

print(f"Missing tempo after processing: {data['tempo'].isna().sum()}")

# ============================================================
# 5. Transform track duration
# ============================================================

print("\n[4] Transform track duration")

print("Original duration statistics:")
print(data["duration_ms"].describe())

# Convert milliseconds to seconds.
data["duration_seconds"] = data["duration_ms"] / 1000

# Log transformation reduces right skewness.
data["log_duration"] = np.log1p(data["duration_ms"])

print("\nLog-transformed duration statistics:")
print(data["log_duration"].describe())

# Remove original duration_ms from modeling variables.
data = data.drop(columns=["duration_ms", "duration_seconds"])

# ============================================================
# 6. Check categorical and binary variables
# ============================================================

print("\n[5] Check categorical variables")

print("\nKey distribution:")
print(data["key"].value_counts().sort_index())

print("\nMode distribution:")
print(data["mode"].value_counts().sort_index())

print("\nExplicit distribution:")
print(data["explicit"].value_counts().sort_index())

# Key will be treated as a categorical variable
# during model construction.
# Mode and Explicit are binary variables.

# ============================================================
# 7. Check data validity
# ============================================================

print("\n[6] Data validity checks")

checks = []

def record_check(name, count):
    checks.append({
        "check": name,
        "problem_count": int(count)
    })
    print(f"{name}: {int(count)}")

record_check(
    "Missing target values",
    data[target].isna().sum()
)

record_check(
    "Popularity outside [0, 100]",
    ((data[target] < 0) | (data[target] > 100)).sum()
)

record_check(
    "Invalid year outside [1921, 2020]",
    ((data[year_feature] < 1921) |
     (data[year_feature] > 2020)).sum()
)

record_check(
    "Invalid key outside [0, 11]",
    ((data["key"] < 0) | (data["key"] > 11)).sum()
)

record_check(
    "Invalid mode values",
    (~data["mode"].isin([0, 1])).sum()
)

record_check(
    "Invalid explicit values",
    (~data["explicit"].isin([0, 1])).sum()
)

record_check(
    "Non-positive duration",
    (data["log_duration"] <= 0).sum()
)

checks_df = pd.DataFrame(checks)

checks_df.to_csv(
    OUTPUT_DIR / "preprocessing_checks.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 8. Define modeling feature groups
# ============================================================

print("\n[7] Define modeling feature groups")

# Group A: Audio features and track-related metadata.
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

# Group B: Group A + release year.
group_b_features = group_a_features + ["year"]

print(f"Group A features ({len(group_a_features)}):")
print(group_a_features)

print(f"\nGroup B features ({len(group_b_features)}):")
print(group_b_features)

# ============================================================
# 9. Save modeling dataset
# ============================================================

print("\n[8] Save processed dataset")

output_file = OUTPUT_DIR / "spotify_modeling_data.csv"

data.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved: {output_file}")

# ============================================================
# 10. Save feature configuration
# ============================================================

feature_config = pd.DataFrame({
    "feature": group_b_features,
    "group_a": [
        feature in group_a_features
        for feature in group_b_features
    ],
    "group_b": [True] * len(group_b_features)
})

feature_config.to_csv(
    OUTPUT_DIR / "model_feature_config.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 11. Save preprocessing summary
# ============================================================

summary = [
    "SPOTIFY DATA PREPROCESSING SUMMARY",
    "=" * 50,
    f"Original rows: {df.shape[0]}",
    f"Processed rows: {data.shape[0]}",
    f"Processed columns: {data.shape[1]}",
    f"Invalid tempo values converted to missing: {invalid_tempo_count}",
    f"Popularity zero values retained: {(data[target] == 0).sum()}",
    f"Popularity mean: {data[target].mean():.4f}",
    f"Popularity standard deviation: {data[target].std():.4f}",
    f"Group A feature count: {len(group_a_features)}",
    f"Group B feature count: {len(group_b_features)}",
    "",
    "Important notes:",
    "1. Original data was not modified.",
    "2. Popularity = 0 observations were retained.",
    "3. Non-positive tempo values were converted to missing.",
    "4. Duration was log-transformed.",
    "5. Key will be one-hot encoded during modeling.",
    "6. Standardization and imputation will be performed",
    "   inside the modeling pipeline to avoid data leakage."
]

with open(
    OUTPUT_DIR / "preprocessing_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(summary))

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Processed shape: {data.shape}")
print(f"Output directory: {OUTPUT_DIR}")
