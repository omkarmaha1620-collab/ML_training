import pandas as pd
import numpy as np

INPUT_FILE = "zenodo_raw_filtered.csv"
CHUNK_SIZE = 100_000

print("=" * 70)
print("RL AIS TRAJECTORY GAP ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# STEP 1: Read required columns
# ------------------------------------------------------------

print("\nReading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["MMSI", "TIMESTAMP"]
)

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------------------
# STEP 2: Convert timestamp
# ------------------------------------------------------------

df["TIMESTAMP"] = pd.to_numeric(
    df["TIMESTAMP"],
    errors="coerce"
)

df = df.dropna(
    subset=["MMSI", "TIMESTAMP"]
)

# ------------------------------------------------------------
# STEP 3: Sort by vessel and time
# ------------------------------------------------------------

print("\nSorting trajectories...")

df = df.sort_values(
    ["MMSI", "TIMESTAMP"]
)

# ------------------------------------------------------------
# STEP 4: Calculate time difference
# ------------------------------------------------------------

print("\nCalculating time gaps...")

df["gap_seconds"] = (
    df.groupby("MMSI")["TIMESTAMP"]
      .diff()
)

gaps = df["gap_seconds"].dropna()

# ------------------------------------------------------------
# STEP 5: Overall statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL GAP STATISTICS")
print("=" * 70)

print(f"Minimum gap : {gaps.min():.2f} seconds")
print(f"Maximum gap : {gaps.max():.2f} seconds")
print(f"Mean gap    : {gaps.mean():.2f} seconds")
print(f"Median gap  : {gaps.median():.2f} seconds")

print("\nGap percentiles:")

for p in [50, 75, 90, 95, 99]:

    value = np.percentile(gaps, p)

    print(
        f"{p:>2}th percentile : "
        f"{value:.2f} seconds "
        f"({value / 60:.2f} minutes)"
    )

# ------------------------------------------------------------
# STEP 6: Gap distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GAP DISTRIBUTION")
print("=" * 70)

thresholds = [
    10,
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    21600
]

for seconds in thresholds:

    count = (gaps <= seconds).sum()

    percentage = (
        count / len(gaps)
    ) * 100

    print(
        f"<= {seconds:6} sec : "
        f"{count:10,} gaps | "
        f"{percentage:6.2f}%"
    )

# ------------------------------------------------------------
# STEP 7: Large gaps
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LARGE GAPS")
print("=" * 70)

for hours in [1, 3, 6, 12, 24]:

    threshold = hours * 3600

    count = (gaps > threshold).sum()

    percentage = (
        count / len(gaps)
    ) * 100

    print(
        f"> {hours:2} hour(s) : "
        f"{count:10,} gaps | "
        f"{percentage:6.2f}%"
    )

# ------------------------------------------------------------
# STEP 8: Per-vessel gap statistics
# ------------------------------------------------------------

print("\nCalculating vessel gap statistics...")

vessel_stats = (
    df.groupby("MMSI")["gap_seconds"]
      .agg(
          median_gap="median",
          mean_gap="mean",
          max_gap="max",
          gap_count="count"
      )
      .reset_index()
)

print("\n" + "=" * 70)
print("VESSEL GAP SUMMARY")
print("=" * 70)

print(
    f"Vessels analyzed : "
    f"{len(vessel_stats):,}"
)

print(
    f"Median vessel gap : "
    f"{vessel_stats['median_gap'].median():.2f} seconds"
)

print(
    f"Mean vessel gap : "
    f"{vessel_stats['mean_gap'].mean():.2f} seconds"
)

print(
    f"Median maximum gap : "
    f"{vessel_stats['max_gap'].median():.2f} seconds"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)