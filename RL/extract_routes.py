import pandas as pd
import numpy as np

DATA_FILE = "rl_training_dataset_v2.csv"
OUTPUT_FILE = "rl_route_episodes_clean.csv"

# Maximum physically plausible vessel speed for our dataset.
# We use a generous threshold so legitimate fast vessels are not removed.
MAX_IMPLIED_SPEED_KNOTS = 40

# Maximum AIS gap allowed inside one trajectory.
MAX_GAP_SECONDS = 3600

# Minimum route requirements
MIN_RECORDS = 100
MIN_MOVING_RECORDS = 50
MIN_DISTANCE_KM = 5
MIN_DURATION_HOURS = 0.25


print("=" * 70)
print("RL ROUTE EXTRACTION + MOVEMENT CLEANING")
print("=" * 70)

print("\nLoading RL V2 dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows    : {len(df):,}")
print(f"Vessels : {df['MMSI'].nunique():,}")


# ============================================================
# TIMESTAMP
# ============================================================

df["TIMESTAMP"] = pd.to_numeric(
    df["TIMESTAMP"],
    errors="coerce"
)

df["datetime"] = pd.to_datetime(
    df["TIMESTAMP"],
    unit="s",
    errors="coerce"
)


# ============================================================
# SORT
# ============================================================

print("\nSorting trajectories...")

df = df.sort_values(
    ["MMSI", "TIMESTAMP"]
).reset_index(drop=True)


# ============================================================
# PREVIOUS POSITION
# ============================================================

df["prev_lat"] = (
    df.groupby("MMSI")["LAT"].shift(1)
)

df["prev_lon"] = (
    df.groupby("MMSI")["LON"].shift(1)
)

df["gap_seconds"] = (
    df.groupby("MMSI")["TIMESTAMP"].diff()
)


# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(
        np.sqrt(a)
    )


df["step_distance_km"] = haversine(
    df["prev_lat"],
    df["prev_lon"],
    df["LAT"],
    df["LON"]
)

df["step_distance_km"] = (
    df["step_distance_km"].fillna(0)
)


# ============================================================
# IMPLIED SPEED
# ============================================================

# km/hour
df["implied_speed_kmh"] = (
    df["step_distance_km"]
    /
    (df["gap_seconds"] / 3600)
)

# knots = km/h / 1.852
df["implied_speed_knots"] = (
    df["implied_speed_kmh"] / 1.852
)

# First record has no previous point
df.loc[
    df["gap_seconds"].isna(),
    "implied_speed_knots"
] = 0


print("\n" + "=" * 70)
print("MOVEMENT QUALITY CHECK")
print("=" * 70)

print(
    f"Maximum implied speed: "
    f"{df['implied_speed_knots'].max():.2f} knots"
)

bad_movement = (
    df["implied_speed_knots"]
    > MAX_IMPLIED_SPEED_KNOTS
)

print(
    f"Implausible movement records: "
    f"{bad_movement.sum():,}"
)

print(
    f"Percentage: "
    f"{bad_movement.mean() * 100:.4f}%"
)


# ============================================================
# REMOVE IMPOSSIBLE MOVEMENT
# ============================================================

df = df[
    ~bad_movement
].copy()


# ============================================================
# MOVEMENT FLAG
# ============================================================

df["moving"] = (
    df["SOG"] > 1.0
)


# ============================================================
# CREATE TRAJECTORY SEGMENTS
# ============================================================

print("\nCreating trajectory segments...")

new_vessel = (
    df["MMSI"] != df["MMSI"].shift(1)
)

large_gap = (
    df["gap_seconds"] > MAX_GAP_SECONDS
)

df["new_segment"] = (
    new_vessel |
    large_gap
)

df["segment_id"] = (
    df["new_segment"].cumsum()
)


# ============================================================
# BUILD ROUTES
# ============================================================

print("\nBuilding cleaned route episodes...")

routes = []

for segment_id, group in df.groupby("segment_id"):

    if len(group) < MIN_RECORDS:
        continue

    moving = group[
        group["moving"]
    ]

    if len(moving) < MIN_MOVING_RECORDS:
        continue

    distance = (
        group["step_distance_km"].sum()
    )

    if distance < MIN_DISTANCE_KM:
        continue

    start = group.iloc[0]
    end = group.iloc[-1]

    duration_seconds = (
        end["TIMESTAMP"]
        -
        start["TIMESTAMP"]
    )

    if duration_seconds <= 0:
        continue

    duration_hours = (
        duration_seconds / 3600
    )

    if duration_hours < MIN_DURATION_HOURS:
        continue

    direct_distance = haversine(
        start["LAT"],
        start["LON"],
        end["LAT"],
        end["LON"]
    )

    routes.append({

        "MMSI": start["MMSI"],

        "segment_id": segment_id,

        "start_time": start["datetime"],
        "end_time": end["datetime"],

        "start_lat": start["LAT"],
        "start_lon": start["LON"],

        "end_lat": end["LAT"],
        "end_lon": end["LON"],

        "route_distance_km": distance,

        "direct_distance_km": direct_distance,

        "duration_hours": duration_hours,

        "average_speed_knots": (
            moving["SOG"].mean()
        ),

        "maximum_speed_knots": (
            moving["SOG"].max()
        ),

        "num_records": len(group),

        "moving_records": len(moving)
    })


routes_df = pd.DataFrame(routes)


# ============================================================
# SAVE + SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN ROUTE EPISODE SUMMARY")
print("=" * 70)

print(
    f"Route episodes : "
    f"{len(routes_df):,}"
)

if len(routes_df) > 0:

    print(
        f"Unique vessels : "
        f"{routes_df['MMSI'].nunique():,}"
    )

    print("\nRoute distance:")

    print(
        routes_df[
            "route_distance_km"
        ].describe()
    )

    print("\nDirect distance:")

    print(
        routes_df[
            "direct_distance_km"
        ].describe()
    )

    print("\nDuration:")

    print(
        routes_df[
            "duration_hours"
        ].describe()
    )

    # Route efficiency / tortuosity
    routes_df["route_efficiency"] = np.where(
        routes_df["route_distance_km"] > 0,
        routes_df["direct_distance_km"]
        /
        routes_df["route_distance_km"],
        0
    )

    print("\nRoute efficiency:")

    print(
        routes_df[
            "route_efficiency"
        ].describe()
    )

    print("\nTop 20 routes by distance:")

    print(
        routes_df
        .sort_values(
            "route_distance_km",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    routes_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(
        f"1. {OUTPUT_FILE}"
    )

else:

    print(
        "\nWARNING: No routes detected."
    )


print("\n" + "=" * 70)
print("CLEAN ROUTE EXTRACTION COMPLETED")
print("=" * 70)