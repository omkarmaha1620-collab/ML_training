import pandas as pd
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "zenodo_raw_filtered.csv"
STATS_FILE = "rl_trajectory_statistics.csv"

OUTPUT_FILE = "rl_training_dataset_v2.csv"

MIN_DAYS = 7
MIN_RECORDS = 5000

MAX_GAP_SECONDS = 3600       # 1 hour

TARGET_ROWS = 1_500_000

# Maximum records taken from one vessel
MAX_RECORDS_PER_VESSEL = 20_000

# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING RL TRAINING DATASET V2")
print("=" * 70)

# ============================================================
# LOAD TRAJECTORY STATISTICS
# ============================================================

if not os.path.exists(STATS_FILE):
    raise FileNotFoundError(
        f"{STATS_FILE} not found."
    )

stats = pd.read_csv(STATS_FILE)

print("\nTrajectory statistics loaded.")
print(f"Total vessels: {len(stats):,}")

# ============================================================
# SELECT GOOD VESSELS
# ============================================================

selected = stats[
    (stats["duration_days"] >= MIN_DAYS) &
    (stats["records"] >= MIN_RECORDS)
].copy()

# Sort by record count
selected = selected.sort_values(
    "records",
    ascending=False
)

print("\n" + "=" * 70)
print("VESSEL SELECTION")
print("=" * 70)

print(f"Minimum trajectory : {MIN_DAYS} days")
print(f"Minimum records    : {MIN_RECORDS:,}")
print(f"Eligible vessels   : {len(selected):,}")

# ============================================================
# READ RAW DATA
# ============================================================

selected_mmsi = set(
    selected["MMSI"].astype(str)
)

columns = [
    "MMSI",
    "UTC",
    "TIMESTAMP",
    "LON",
    "LAT",
    "SOG",
    "COG",
    "HEADING",
    "STATUS",
    "STATUS_DESC"
]

print("\n" + "=" * 70)
print("READING RAW DATA")
print("=" * 70)

chunks = []

for number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        chunksize=100_000
    ),
    start=1
):

    print(f"Processing chunk {number}...")

    chunk["MMSI"] = chunk["MMSI"].astype(str)

    chunk = chunk[
        chunk["MMSI"].isin(selected_mmsi)
    ]

    if len(chunk) > 0:
        chunks.append(chunk)

# ============================================================
# COMBINE
# ============================================================

print("\nCombining selected records...")

df = pd.concat(
    chunks,
    ignore_index=True
)

print(
    f"Selected raw rows: {len(df):,}"
)

# ============================================================
# CLEAN
# ============================================================

print("\nCleaning data...")

df["TIMESTAMP"] = pd.to_numeric(
    df["TIMESTAMP"],
    errors="coerce"
)

df["LAT"] = pd.to_numeric(
    df["LAT"],
    errors="coerce"
)

df["LON"] = pd.to_numeric(
    df["LON"],
    errors="coerce"
)

df["SOG"] = pd.to_numeric(
    df["SOG"],
    errors="coerce"
)

df["COG"] = pd.to_numeric(
    df["COG"],
    errors="coerce"
)

df["HEADING"] = pd.to_numeric(
    df["HEADING"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "MMSI",
        "TIMESTAMP",
        "LAT",
        "LON",
        "SOG",
        "COG"
    ]
)

# Valid geographical coordinates
df = df[
    df["LAT"].between(-90, 90)
]

df = df[
    df["LON"].between(-180, 180)
]

df = df[
    df["SOG"] >= 0
]

df = df[
    df["COG"].between(0, 360)
]

# ============================================================
# SORT TRAJECTORIES
# ============================================================

print("Sorting trajectories...")

df = df.sort_values(
    ["MMSI", "TIMESTAMP"]
).reset_index(drop=True)

# ============================================================
# REMOVE LARGE GAPS
# ============================================================

print("Checking trajectory gaps...")

df["gap_seconds"] = (
    df.groupby("MMSI")["TIMESTAMP"]
      .diff()
)

# Keep first record and records with <= 1 hour gap
df = df[
    df["gap_seconds"].isna() |
    (df["gap_seconds"] <= MAX_GAP_SECONDS)
].copy()

print(
    f"Rows after gap filtering: {len(df):,}"
)

# ============================================================
# LIMIT EACH VESSEL
# ============================================================

print("\n" + "=" * 70)
print("BALANCING VESSEL REPRESENTATION")
print("=" * 70)

# Number of records per vessel
counts = (
    df.groupby("MMSI")
      .size()
      .sort_values(ascending=False)
)

print(
    f"Vessels available: {len(counts):,}"
)

print(
    f"Maximum records/vessel: "
    f"{MAX_RECORDS_PER_VESSEL:,}"
)

# We will take chronological records from each vessel.
# No random row deletion.

parts = []

total_rows = 0

for mmsi in counts.index:

    vessel = df[
        df["MMSI"] == mmsi
    ].copy()

    # Keep maximum allowed records
    vessel = vessel.head(
        MAX_RECORDS_PER_VESSEL
    )

    # Don't exceed target
    remaining = TARGET_ROWS - total_rows

    if remaining <= 0:
        break

    if len(vessel) > remaining:
        vessel = vessel.head(remaining)

    parts.append(vessel)

    total_rows += len(vessel)

    print(
        f"Vessel {mmsi}: "
        f"{len(vessel):,} records | "
        f"Total: {total_rows:,}"
    )

# ============================================================
# COMBINE VESSELS
# ============================================================

df_final = pd.concat(
    parts,
    ignore_index=True
)

# Remove helper column
df_final = df_final.drop(
    columns=["gap_seconds"],
    errors="ignore"
)

# Final chronological ordering
df_final = df_final.sort_values(
    ["MMSI", "TIMESTAMP"]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("SAVING RL DATASET V2")
print("=" * 70)

df_final.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RL DATASET V2 SUMMARY")
print("=" * 70)

print(
    f"Rows             : {len(df_final):,}"
)

print(
    f"Vessels           : "
    f"{df_final['MMSI'].nunique():,}"
)

print(
    f"Columns           : "
    f"{len(df_final.columns)}"
)

print(
    f"Records/vessel    : "
    f"{len(df_final) / df_final['MMSI'].nunique():,.1f} average"
)

print(
    f"Start timestamp   : "
    f"{df_final['TIMESTAMP'].min()}"
)

print(
    f"End timestamp     : "
    f"{df_final['TIMESTAMP'].max()}"
)

print("\nColumns:")

for column in df_final.columns:
    print(f" - {column}")

print("\nSaved:")
print(f"1. {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("RL DATASET V2 CREATED SUCCESSFULLY")
print("=" * 70)