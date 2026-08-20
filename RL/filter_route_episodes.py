import pandas as pd
import numpy as np

INPUT_FILE = "rl_route_episodes_clean.csv"
OUTPUT_FILE = "rl_route_episodes_final.csv"

print("=" * 70)
print("RL ROUTE QUALITY FILTERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput routes   : {len(df):,}")
print(f"Input vessels  : {df['MMSI'].nunique():,}")

# ------------------------------------------------------------
# Calculate route efficiency safely
# ------------------------------------------------------------

df["route_efficiency"] = np.where(
    df["route_distance_km"] > 0,
    df["direct_distance_km"] / df["route_distance_km"],
    np.nan
)

# Small numerical/data errors can produce >1.
df["route_efficiency"] = df["route_efficiency"].clip(0, 1)

# ------------------------------------------------------------
# Quality filters
# ------------------------------------------------------------

filtered = df[
    (df["route_distance_km"] >= 10) &
    (df["direct_distance_km"] >= 10) &
    (df["duration_hours"] >= 0.5) &
    (df["average_speed_knots"] >= 2) &
    (df["average_speed_knots"] <= 35) &
    (df["route_efficiency"] >= 0.40)
].copy()

print("\n" + "=" * 70)
print("FILTER RESULTS")
print("=" * 70)

print(f"Routes before filtering : {len(df):,}")
print(f"Routes after filtering  : {len(filtered):,}")
print(
    f"Routes removed          : "
    f"{len(df) - len(filtered):,}"
)

print(
    f"Vessels remaining       : "
    f"{filtered['MMSI'].nunique():,}"
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ROUTE STATISTICS")
print("=" * 70)

if len(filtered) > 0:

    print("\nDistance:")
    print(filtered["route_distance_km"].describe())

    print("\nDirect distance:")
    print(filtered["direct_distance_km"].describe())

    print("\nDuration:")
    print(filtered["duration_hours"].describe())

    print("\nAverage speed:")
    print(filtered["average_speed_knots"].describe())

    print("\nRoute efficiency:")
    print(filtered["route_efficiency"].describe())

    print("\nTop 20 routes:")

    print(
        filtered
        .sort_values(
            "direct_distance_km",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filtered.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(f"1. {OUTPUT_FILE}")

else:

    print("\nWARNING: No routes remain.")
    print("We will loosen the filters.")

print("\n" + "=" * 70)
print("ROUTE QUALITY FILTERING COMPLETED")
print("=" * 70)