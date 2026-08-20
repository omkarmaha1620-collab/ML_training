import pandas as pd

PPO_FILE = "rl_unseen_vessel_results.csv"
BASELINE_FILE = "rl_baseline_results.csv"

ppo = pd.read_csv(PPO_FILE)
baseline = pd.read_csv(BASELINE_FILE)

print("=" * 70)
print("NEW PPO VS DIRECT-NAVIGATION ANALYSIS")
print("=" * 70)

print("\nPPO columns:")
print(ppo.columns.tolist())

print("\nBaseline columns:")
print(baseline.columns.tolist())

# ---------------------------------------------------------
# Find route identity columns
# ---------------------------------------------------------

keys = ["MMSI"]

for col in ["route_start_time", "route_start_time_ppo"]:
    if col in ppo.columns:
        ppo_key = col
        break
else:
    ppo_key = None

for col in ["route_start_time", "route_start_time_baseline"]:
    if col in baseline.columns:
        baseline_key = col
        break
else:
    baseline_key = None


# ---------------------------------------------------------
# Rename columns
# ---------------------------------------------------------

if "ppo_distance_km" not in ppo.columns:
    raise ValueError("ppo_distance_km not found in PPO file.")

if "baseline_distance_km" not in baseline.columns:
    raise ValueError("baseline_distance_km not found in baseline file.")


# ---------------------------------------------------------
# Match routes
# ---------------------------------------------------------

if ppo_key and baseline_key:

    ppo_small = ppo[
        ["MMSI", ppo_key, "ppo_distance_km",
         "total_turning_degrees", "destination_reached"]
    ].copy()

    baseline_small = baseline[
        ["MMSI", baseline_key, "baseline_distance_km"]
    ].copy()

    ppo_small = ppo_small.rename(
        columns={ppo_key: "route_start_time"}
    )

    baseline_small = baseline_small.rename(
        columns={baseline_key: "route_start_time"}
    )

    merged = pd.merge(
        ppo_small,
        baseline_small,
        on=["MMSI", "route_start_time"],
        how="inner"
    )

else:

    print("\nCould not find route start-time columns.")
    print("Matching rows by index instead.")

    n = min(len(ppo), len(baseline))

    merged = pd.DataFrame({
        "MMSI": ppo["MMSI"].iloc[:n].values,
        "ppo_distance_km": ppo["ppo_distance_km"].iloc[:n].values,
        "baseline_distance_km":
            baseline["baseline_distance_km"].iloc[:n].values,
        "total_turning_degrees":
            ppo["total_turning_degrees"].iloc[:n].values,
        "destination_reached":
            ppo["destination_reached"].iloc[:n].values,
    })


# ---------------------------------------------------------
# Calculate comparison
# ---------------------------------------------------------

merged["difference_km"] = (
    merged["ppo_distance_km"]
    - merged["baseline_distance_km"]
)

merged["ppo_vs_baseline_percent"] = (
    merged["difference_km"]
    / merged["baseline_distance_km"]
    * 100
)

merged["ppo_better"] = (
    merged["difference_km"] < 0
)

merged["ppo_worse"] = (
    merged["difference_km"] > 0
)


# ---------------------------------------------------------
# Overall
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL")
print("=" * 70)

print(f"Routes matched       : {len(merged)}")
print(
    f"PPO mean distance    : "
    f"{merged['ppo_distance_km'].mean():.2f} km"
)
print(
    f"Baseline mean        : "
    f"{merged['baseline_distance_km'].mean():.2f} km"
)
print(
    f"Mean difference      : "
    f"{merged['difference_km'].mean():.2f} km"
)

print(
    f"PPO better           : "
    f"{merged['ppo_better'].sum()}"
)

print(
    f"PPO worse            : "
    f"{merged['ppo_worse'].sum()}"
)

print(
    f"Equal                : "
    f"{(merged['difference_km'] == 0).sum()}"
)

print(
    f"Mean turning         : "
    f"{merged['total_turning_degrees'].mean():.2f}°"
)


# ---------------------------------------------------------
# Worst PPO routes
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("10 WORST PPO ROUTES")
print("=" * 70)

worst = merged.sort_values(
    "difference_km",
    ascending=False
).head(10)

print(
    worst[
        [
            "MMSI",
            "baseline_distance_km",
            "ppo_distance_km",
            "difference_km",
            "ppo_vs_baseline_percent",
            "total_turning_degrees",
            "destination_reached",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Best PPO routes
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("10 BEST PPO ROUTES")
print("=" * 70)

best = merged.sort_values(
    "difference_km",
    ascending=True
).head(10)

print(
    best[
        [
            "MMSI",
            "baseline_distance_km",
            "ppo_distance_km",
            "difference_km",
            "ppo_vs_baseline_percent",
            "total_turning_degrees",
            "destination_reached",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

merged.to_csv(
    "rl_new_ppo_vs_baseline_analysis.csv",
    index=False
)

print("\nSaved:")
print("1. rl_new_ppo_vs_baseline_analysis.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)