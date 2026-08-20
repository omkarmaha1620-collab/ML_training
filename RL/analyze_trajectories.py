import pandas as pd
import numpy as np

INPUT_FILE = "zenodo_raw_filtered.csv"
OUTPUT_FILE = "rl_trajectory_statistics.csv"

CHUNK_SIZE = 100_000

print("=" * 70)
print("RL TRAJECTORY ANALYSIS")
print("=" * 70)

print("\nReading dataset in chunks...")

stats = {}

for i, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=["MMSI", "TIMESTAMP"],
        chunksize=CHUNK_SIZE
    )
):
    print(f"Processing chunk {i + 1}...")

    chunk["TIMESTAMP"] = pd.to_numeric(
        chunk["TIMESTAMP"],
        errors="coerce"
    )

    grouped = chunk.groupby("MMSI")["TIMESTAMP"].agg(
        ["count", "min", "max"]
    )

    for mmsi, row in grouped.iterrows():

        if mmsi not in stats:
            stats[mmsi] = {
                "records": 0,
                "min_timestamp": np.inf,
                "max_timestamp": -np.inf
            }

        stats[mmsi]["records"] += int(row["count"])

        if row["min"] < stats[mmsi]["min_timestamp"]:
            stats[mmsi]["min_timestamp"] = row["min"]

        if row["max"] > stats[mmsi]["max_timestamp"]:
            stats[mmsi]["max_timestamp"] = row["max"]


print("\nBuilding trajectory statistics...")

result = []

for mmsi, values in stats.items():

    duration_seconds = (
        values["max_timestamp"] -
        values["min_timestamp"]
    )

    duration_days = duration_seconds / 86400

    result.append({
        "MMSI": mmsi,
        "records": values["records"],
        "start_timestamp": values["min_timestamp"],
        "end_timestamp": values["max_timestamp"],
        "duration_days": duration_days
    })


df = pd.DataFrame(result)

df = df.sort_values(
    by="records",
    ascending=False
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("TRAJECTORY SUMMARY")
print("=" * 70)

print(f"Vessels analyzed : {len(df):,}")
print(f"Total records    : {df['records'].sum():,}")

print("\nDuration statistics:")
print(f"Minimum : {df['duration_days'].min():.2f} days")
print(f"Maximum : {df['duration_days'].max():.2f} days")
print(f"Mean    : {df['duration_days'].mean():.2f} days")
print(f"Median  : {df['duration_days'].median():.2f} days")

print("\n" + "=" * 70)
print("VESSELS BY TRAJECTORY DURATION")
print("=" * 70)

for days in [1, 3, 7, 14, 21, 30]:

    selected = df[df["duration_days"] >= days]

    print(
        f">= {days:2} days : "
        f"{len(selected):5,} vessels | "
        f"{selected['records'].sum():,} rows"
    )


print("\n" + "=" * 70)
print("VESSELS BY RECORD COUNT")
print("=" * 70)

for records in [1000, 5000, 10000, 20000, 50000]:

    selected = df[df["records"] >= records]

    print(
        f">= {records:6,} records : "
        f"{len(selected):5,} vessels | "
        f"{selected['records'].sum():,} rows"
    )


print("\n" + "=" * 70)
print("TOP 20 TRAJECTORIES")
print("=" * 70)

print(
    df.head(20).to_string(index=False)
)

print("\nSaved:")
print(f"1. {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("TRAJECTORY ANALYSIS COMPLETED")
print("=" * 70)