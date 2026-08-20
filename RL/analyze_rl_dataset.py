import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# RL DATASET ANALYSIS
# ============================================================

DATA_FILE = Path("zenodo_raw_filtered.csv")

# Read in chunks so the 1 GB file does not need to fit in RAM
CHUNK_SIZE = 100_000

print("=" * 70)
print("RL DATASET ANALYSIS")
print("=" * 70)

if not DATA_FILE.exists():
    print(f"\nERROR: File not found:")
    print(DATA_FILE.resolve())
    raise SystemExit(1)

print(f"\nDataset: {DATA_FILE.resolve()}")
print(f"File size: {DATA_FILE.stat().st_size / (1024**3):.2f} GB")

# ============================================================
# 1. READ HEADER
# ============================================================

print("\n" + "=" * 70)
print("READING HEADER")
print("=" * 70)

header = pd.read_csv(DATA_FILE, nrows=5)

print("\nColumns:")
for i, col in enumerate(header.columns, 1):
    print(f"{i:2}. {col}")

# ============================================================
# 2. FIND VESSEL ID COLUMN
# ============================================================

possible_vessel_columns = [
    "MMSI",
    "mmsi",
    "VESSEL_ID",
    "vessel_id",
    "SHIP_ID",
    "ship_id",
    "IMO",
    "imo",
    "SHIP_MMSI",
    "ship_mmsi"
]

vessel_col = None

for col in possible_vessel_columns:
    if col in header.columns:
        vessel_col = col
        break

if vessel_col is None:
    print("\nWARNING: Could not automatically identify vessel ID column.")
    print("Available columns:")
    print(list(header.columns))
    print("\nPlease tell me which column identifies the vessel.")
    raise SystemExit(1)

print(f"\nVessel ID column: {vessel_col}")

# ============================================================
# 3. FIND TIMESTAMP COLUMN
# ============================================================

possible_time_columns = [
    "TIMESTAMP",
    "timestamp",
    "Timestamp",
    "UTC",
    "utc",
    "DATE",
    "date",
    "DATETIME",
    "datetime"
]

time_col = None

for col in possible_time_columns:
    if col in header.columns:
        time_col = col
        break

if time_col is None:
    print("\nWARNING: Could not automatically identify timestamp column.")
    raise SystemExit(1)

print(f"Timestamp column: {time_col}")

# ============================================================
# 4. SAMPLE RAW TIMESTAMP VALUES
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP CHECK")
print("=" * 70)

sample = pd.read_csv(
    DATA_FILE,
    usecols=[time_col],
    nrows=20
)

print("\nRaw timestamp values:")
print(sample[time_col].head(20).to_string(index=False))

# Try numeric conversion
numeric_time = pd.to_numeric(sample[time_col], errors="coerce")

if numeric_time.notna().sum() > 0:

    median_value = numeric_time.dropna().median()

    print(f"\nTypical raw timestamp value: {median_value}")

    # Automatically determine likely Unix timestamp unit
    if median_value > 1e17:
        time_unit = "ns"
    elif median_value > 1e14:
        time_unit = "us"
    elif median_value > 1e11:
        time_unit = "ms"
    elif median_value > 1e8:
        time_unit = "s"
    else:
        time_unit = None

    if time_unit:
        converted = pd.to_datetime(
            numeric_time,
            unit=time_unit,
            errors="coerce"
        )

        print(f"Detected timestamp unit: {time_unit}")
        print("\nConverted timestamps:")
        print(converted.head(20).to_string(index=False))
    else:
        print("Timestamp does not appear to be Unix time.")
        time_unit = None

else:
    time_unit = None
    converted = pd.to_datetime(
        sample[time_col],
        errors="coerce"
    )

    print("\nTimestamp appears to be a datetime string.")

# ============================================================
# 5. FULL DATASET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING FULL DATASET")
print("=" * 70)

print("\nThis may take several minutes because the dataset has")
print("approximately 9.6 million rows.")

total_rows = 0
vessel_counts = Counter()

global_min_time = None
global_max_time = None

missing_counts = Counter()

sog_values = []
cog_values = []

chunk_number = 0

use_columns = [vessel_col, time_col]

# Add movement columns if available
for col in ["LON", "LAT", "SOG", "COG", "HEADING", "STATUS"]:
    if col in header.columns:
        use_columns.append(col)

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=use_columns,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk_number += 1
    total_rows += len(chunk)

    # --------------------------------------------------------
    # Vessel counts
    # --------------------------------------------------------

    counts = chunk[vessel_col].value_counts(dropna=True)

    for vessel, count in counts.items():
        vessel_counts[vessel] += int(count)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for col in chunk.columns:
        missing_counts[col] += int(chunk[col].isna().sum())

    # --------------------------------------------------------
    # Timestamp range
    # --------------------------------------------------------

    raw_time = chunk[time_col]

    if time_unit:

        numeric = pd.to_numeric(raw_time, errors="coerce")

        converted_time = pd.to_datetime(
            numeric,
            unit=time_unit,
            errors="coerce"
        )

    else:

        converted_time = pd.to_datetime(
            raw_time,
            errors="coerce"
        )

    valid_time = converted_time.dropna()

    if len(valid_time) > 0:

        chunk_min = valid_time.min()
        chunk_max = valid_time.max()

        if global_min_time is None or chunk_min < global_min_time:
            global_min_time = chunk_min

        if global_max_time is None or chunk_max > global_max_time:
            global_max_time = chunk_max

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if chunk_number % 10 == 0:
        print(
            f"Processed approximately "
            f"{total_rows:,} rows..."
        )

# ============================================================
# 6. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal rows       : {total_rows:,}")
print(f"Total columns    : {len(header.columns)}")
print(f"Unique vessels   : {len(vessel_counts):,}")

print("\nTime range:")

if global_min_time is not None:
    print(f"Start            : {global_min_time}")
    print(f"End              : {global_max_time}")
else:
    print("Could not determine timestamp range.")

# ============================================================
# 7. VESSEL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("VESSEL STATISTICS")
print("=" * 70)

counts_array = np.array(list(vessel_counts.values()))

print(f"\nMinimum records/vessel : {counts_array.min():,}")
print(f"Maximum records/vessel : {counts_array.max():,}")
print(f"Mean records/vessel    : {counts_array.mean():,.1f}")
print(f"Median records/vessel  : {np.median(counts_array):,.1f}")

thresholds = [
    10,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000
]

print("\nVessels meeting minimum record count:")

for threshold in thresholds:

    number_of_vessels = sum(
        count >= threshold
        for count in vessel_counts.values()
    )

    rows_from_vessels = sum(
        count
        for count in vessel_counts.values()
        if count >= threshold
    )

    print(
        f">= {threshold:>7,} records : "
        f"{number_of_vessels:>5,} vessels | "
        f"{rows_from_vessels:>10,} rows"
    )

# ============================================================
# 8. TOP 30 VESSELS
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 VESSELS BY NUMBER OF RECORDS")
print("=" * 70)

top_vessels = vessel_counts.most_common(30)

for rank, (vessel, count) in enumerate(top_vessels, 1):
    print(
        f"{rank:2}. Vessel {vessel} : "
        f"{count:,} records"
    )

# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

for col in header.columns:

    missing = missing_counts[col]

    percentage = (
        missing / total_rows * 100
        if total_rows > 0
        else 0
    )

    print(
        f"{col:20} : "
        f"{missing:>10,} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# 10. SAVE VESSEL STATISTICS
# ============================================================

stats_df = pd.DataFrame(
    [
        {
            "vessel_id": vessel,
            "record_count": count
        }
        for vessel, count in vessel_counts.items()
    ]
)

stats_df = stats_df.sort_values(
    "record_count",
    ascending=False
)

stats_file = "rl_vessel_statistics.csv"

stats_df.to_csv(
    stats_file,
    index=False
)

print("\nSaved:")
print(f"1. {stats_file}")

# ============================================================
# 11. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print("""
DO NOT REDUCE THE DATASET YET.

We now need to look at:

1. Vessel record distribution
2. Correct timestamp range
3. Number of vessels with enough trajectory history
4. Missing-value percentage

Then we will create a smaller RL dataset containing
only useful vessel trajectories.

The target will probably be somewhere around
500,000 - 1,500,000 rows for the first RL experiment,
but the exact number will be decided from the statistics above.
""")

print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)