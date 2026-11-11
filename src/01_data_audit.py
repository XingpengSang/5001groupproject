
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Path configuration
# ============================================================

PROJECT_DIR = Path(r"D:\5001groupproject")
DATA_DIR = PROJECT_DIR / "Dataset"
OUTPUT_DIR = PROJECT_DIR / "output"

DATA_FILE = DATA_DIR / "data.csv"

# Create output folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the input file exists
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Cannot find dataset: {DATA_FILE}")

# ============================================================
# 2. Load dataset
# ============================================================

print("=" * 60)
print("SPOTIFY DATASET AUDIT")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\n[1] Dataset dimensions")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n[2] Column names")
print(df.columns.tolist())

print("\n[3] Data types")
print(df.dtypes)

print("\n[4] First five rows")
print(df.head())

print("\n[5] Dataset information")
df.info()

# ============================================================
# 3. Missing values
# ============================================================

print("\n[6] Missing values")

missing = df.isnull().sum()
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing / len(df) * 100
})

print(missing_df)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

# ============================================================
# 4. Duplicate records
# ============================================================

print("\n[7] Duplicate records")

duplicate_rows = df.duplicated().sum()
print(f"Fully duplicated rows: {duplicate_rows}")

if "id" in df.columns:
    duplicate_ids = df["id"].duplicated().sum()
    print(f"Duplicate track IDs: {duplicate_ids}")

    df[df["id"].duplicated(keep=False)].to_csv(
        OUTPUT_DIR / "duplicate_tracks.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 5. Descriptive statistics
# ============================================================

print("\n[8] Descriptive statistics")

desc = df.describe(include="all").T
print(desc)

desc.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv",
    encoding="utf-8-sig"
)

# ============================================================
# 6. Popularity analysis
# ============================================================

print("\n[9] Popularity analysis")

if "popularity" not in df.columns:
    raise ValueError("The dataset does not contain a popularity column.")

popularity = df["popularity"]

print(popularity.describe())
print(f"Popularity = 0: {(popularity == 0).sum()}")
print(f"Popularity > 80: {(popularity > 80).sum()}")

print(
    f"Percentage with popularity = 0: "
    f"{(popularity == 0).mean() * 100:.2f}%"
)

plt.figure(figsize=(9, 5))
plt.hist(popularity.dropna(), bins=30, edgecolor="black")
plt.title("Distribution of Spotify Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Tracks")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "popularity_distribution.png", dpi=300)
plt.close()

# ============================================================
# 7. Release year analysis
# ============================================================

print("\n[10] Release year analysis")

if "year" in df.columns:
    print(f"Minimum year: {df['year'].min()}")
    print(f"Maximum year: {df['year'].max()}")

    year_counts = df["year"].value_counts().sort_index()
    year_counts.to_csv(
        OUTPUT_DIR / "year_distribution.csv",
        header=["track_count"],
        encoding="utf-8-sig"
    )

    year_summary = df.groupby("year")["popularity"].agg(
        ["count", "mean", "median", "std"]
    )

    print(year_summary.head())
    print(year_summary.tail())

    year_summary.to_csv(
        OUTPUT_DIR / "year_popularity_summary.csv",
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 5))
    year_counts.plot()
    plt.title("Number of Tracks by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tracks_by_year.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    year_summary["mean"].plot()
    plt.title("Average Track Popularity by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Average Popularity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_popularity_by_year.png", dpi=300)
    plt.close()

# ============================================================
# 8. Data quality checks
# ============================================================

print("\n[11] Data quality checks")

quality_results = []

def add_check(name, condition):
    count = int(condition.sum())
    quality_results.append({
        "check": name,
        "problem_count": count
    })
    print(f"{name}: {count}")

if "duration_ms" in df.columns:
    add_check("Non-positive duration", df["duration_ms"] <= 0)

if "tempo" in df.columns:
    add_check("Non-positive tempo", df["tempo"] <= 0)

for col in [
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence"
]:
    if col in df.columns:
        add_check(
            f"{col} outside [0, 1]",
            (df[col] < 0) | (df[col] > 1)
        )

add_check(
    "Popularity outside [0, 100]",
    (df["popularity"] < 0) | (df["popularity"] > 100)
)

quality_df = pd.DataFrame(quality_results)
quality_df.to_csv(
    OUTPUT_DIR / "data_quality_checks.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 9. Correlation analysis
# ============================================================

print("\n[12] Correlation with popularity")

numeric_df = df.select_dtypes(include=[np.number])

correlations = numeric_df.corr()["popularity"].sort_values(
    ascending=False
)

print(correlations)

correlations.to_csv(
    OUTPUT_DIR / "popularity_correlations.csv",
    header=["correlation"],
    encoding="utf-8-sig"
)

# ============================================================
# 10. Save audit summary
# ============================================================

summary = [
    "SPOTIFY DATASET AUDIT SUMMARY",
    "=" * 40,
    f"Rows: {df.shape[0]}",
    f"Columns: {df.shape[1]}",
    f"Missing cells: {df.isnull().sum().sum()}",
    f"Fully duplicated rows: {duplicate_rows}",
    f"Popularity mean: {popularity.mean():.2f}",
    f"Popularity median: {popularity.median():.2f}",
    f"Popularity standard deviation: {popularity.std():.2f}",
    f"Popularity = 0 count: {(popularity == 0).sum()}",
]

if "year" in df.columns:
    summary.append(f"Minimum year: {df['year'].min()}")
    summary.append(f"Maximum year: {df['year'].max()}")

summary_path = OUTPUT_DIR / "audit_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n" + "=" * 60)
print("AUDIT COMPLETED SUCCESSFULLY")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
