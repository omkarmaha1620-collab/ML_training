import pandas as pd
import numpy as np

DATA_FILE = "rl_training_dataset_v2.csv"

print("=" * 70)
print("RL V2 ROUTE OPTIMIZATION ANALYSIS")
print("=" * 70)

print("\nLoading RL V2 dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ------------------------------------------------------------
# TIMESTAMP
# ------------------------------------------------------------

df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")

df["datetime"] = pd.to_datetime(
    df["TIMESTAMP"],
    unit="s",
    errors="coerce"
)

# ------------------------------------------------------------
# BASIC DATA QUALITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print(f"Missing values:")
print(df.isna().sum())

print("\nUnique vessels :", df["MMSI"].nunique())

print("\nTime range:")
print("Start:", df["datetime"].min())
print("End  :", df["datetime"].max())

# ------------------------------------------------------------
# VESSEL TRAJECTORIES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VESSEL TRAJECTORIES")
print("=" * 70)

vessel_stats = (
    df.groupby("MMSI")
    .agg(
        records=("MMSI", "size"),
        start_time=("datetime", "min"),
        end_time=("datetime", "max"),
        start_lat=("LAT", "first"),
        start_lon=("LON", "first"),
        end_lat=("LAT", "last"),
        end_lon=("LON", "last"),
    )
    .reset_index()
)

vessel_stats["duration_hours"] = (
    vessel_stats["end_time"] -
    vessel_stats["start_time"]
).dt.total_seconds() / 3600

print(
    vessel_stats[
        [
            "records",
            "duration_hours",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
        ]
    ].describe()
)

# ------------------------------------------------------------
# START / END DISTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("START → END MOVEMENT")
print("=" * 70)

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


vessel_stats["start_end_distance_km"] = haversine(
    vessel_stats["start_lat"],
    vessel_stats["start_lon"],
    vessel_stats["end_lat"],
    vessel_stats["end_lon"],
)

print(
    vessel_stats[
        "start_end_distance_km"
    ].describe()
)

print("\nVessels by start → end distance:")

bins = [
    0,
    10,
    50,
    100,
    250,
    500,
    1000,
    2000,
    np.inf
]

labels = [
    "<10 km",
    "10-50 km",
    "50-100 km",
    "100-250 km",
    "250-500 km",
    "500-1000 km",
    "1000-2000 km",
    ">2000 km"
]

distance_groups = pd.cut(
    vessel_stats["start_end_distance_km"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(distance_groups.value_counts().sort_index())

# ------------------------------------------------------------
# SPEED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPEED STATISTICS")
print("=" * 70)

print(df["SOG"].describe())

print("\nZero-speed records:",
      (df["SOG"] == 0).sum())

print("Records with SOG > 30:",
      (df["SOG"] > 30).sum())

# ------------------------------------------------------------
# HEADING / COURSE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NAVIGATION STATISTICS")
print("=" * 70)

print("\nCOG:")
print(df["COG"].describe())

print("\nHEADING:")
print(df["HEADING"].describe())

# ------------------------------------------------------------
# GEOGRAPHIC BOUNDARIES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GEOGRAPHIC COVERAGE")
print("=" * 70)

print(f"Latitude minimum : {df['LAT'].min()}")
print(f"Latitude maximum : {df['LAT'].max()}")

print(f"Longitude minimum: {df['LON'].min()}")
print(f"Longitude maximum: {df['LON'].max()}")

# ------------------------------------------------------------
# SAMPLE VESSELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 ROUTES BY START → END DISTANCE")
print("=" * 70)

top_routes = vessel_stats.sort_values(
    "start_end_distance_km",
    ascending=False
).head(20)

print(
    top_routes[
        [
            "MMSI",
            "records",
            "duration_hours",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "start_end_distance_km",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# SAVE ANALYSIS
# ------------------------------------------------------------

vessel_stats.to_csv(
    "rl_v2_vessel_route_statistics.csv",
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved:")
print("1. rl_v2_vessel_route_statistics.csv")