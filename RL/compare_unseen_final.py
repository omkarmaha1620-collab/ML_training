import pandas as pd

print("=" * 70)
print("CORRECT UNSEEN-VESSEL PPO VS BASELINE COMPARISON")
print("=" * 70)

PPO_FILE = "rl_unseen_vessel_results.csv"
BASELINE_FILE = "rl_baseline_results.csv"

ppo = pd.read_csv(PPO_FILE)
baseline = pd.read_csv(BASELINE_FILE)

# ------------------------------------------------------------
# Normalize timestamps
# ------------------------------------------------------------

ppo["route_start_time"] = pd.to_datetime(
    ppo["route_start_time"]
)

baseline["route_start_time"] = pd.to_datetime(
    baseline["route_start_time"]
)

# ------------------------------------------------------------
# IMPORTANT:
# Match the exact same route:
# MMSI + route start time
# ------------------------------------------------------------

merged = pd.merge(
    ppo,
    baseline,
    left_on=[
        "MMSI",
        "route_start_time"
    ],
    right_on=[
        "MMSI",
        "route_start_time"
    ],
    suffixes=(
        "_ppo",
        "_baseline"
    )
)

print("\nROUTE MATCHING")
print("-" * 50)

print(
    f"PPO routes      : {len(ppo)}"
)

print(
    f"Baseline routes : {len(baseline)}"
)

print(
    f"Matched routes  : {len(merged)}"
)

if len(merged) != len(ppo):
    print(
        "\nWARNING: Not every PPO route matched."
    )

if len(merged) != len(baseline):
    print(
        "WARNING: Not every baseline route matched."
    )

# ------------------------------------------------------------
# Distance
# ------------------------------------------------------------

historical = merged[
    "historical_route_distance_km_ppo"
].mean()

baseline_mean = merged[
    "baseline_distance_km"
].mean()

ppo_mean = merged[
    "ppo_distance_km"
].mean()

print("\n" + "=" * 70)
print("MEAN ROUTE DISTANCE")
print("=" * 70)

print(
    f"Historical AIS : {historical:.2f} km"
)

print(
    f"Baseline       : {baseline_mean:.2f} km"
)

print(
    f"PPO            : {ppo_mean:.2f} km"
)

# ------------------------------------------------------------
# Improvement vs historical
# ------------------------------------------------------------

baseline_change = (
    (baseline_mean - historical)
    / historical
    * 100
)

ppo_change = (
    (ppo_mean - historical)
    / historical
    * 100
)

print("\n" + "=" * 70)
print("IMPROVEMENT VS HISTORICAL")
print("=" * 70)

print(
    f"Baseline : {baseline_change:+.2f}%"
)

print(
    f"PPO      : {ppo_change:+.2f}%"
)

# ------------------------------------------------------------
# PPO vs baseline
# ------------------------------------------------------------

ppo_vs_baseline = (
    (ppo_mean - baseline_mean)
    / baseline_mean
    * 100
)

ppo_shorter = (
    merged["ppo_distance_km"]
    < merged["baseline_distance_km"]
).sum()

ppo_longer = (
    merged["ppo_distance_km"]
    > merged["baseline_distance_km"]
).sum()

ppo_equal = (
    merged["ppo_distance_km"]
    == merged["baseline_distance_km"]
).sum()

print("\n" + "=" * 70)
print("PPO VS BASELINE")
print("=" * 70)

print(
    f"PPO change vs baseline : "
    f"{ppo_vs_baseline:+.2f}%"
)

print(
    f"PPO shorter            : "
    f"{ppo_shorter}"
)

print(
    f"PPO longer             : "
    f"{ppo_longer}"
)

print(
    f"Equal                  : "
    f"{ppo_equal}"
)

# ------------------------------------------------------------
# Success
# ------------------------------------------------------------

ppo_success = (
    merged["destination_reached"]
    .mean()
    * 100
)

baseline_success = (
    merged["baseline_destination_reached"]
    .mean()
    * 100
)

print("\n" + "=" * 70)
print("DESTINATION SUCCESS")
print("=" * 70)

print(
    f"PPO      : {ppo_success:.2f}%"
)

print(
    f"Baseline : {baseline_success:.2f}%"
)

# ------------------------------------------------------------
# Route-level comparison
# ------------------------------------------------------------

merged[
    "ppo_minus_baseline_km"
] = (
    merged["ppo_distance_km"]
    - merged["baseline_distance_km"]
)

merged[
    "ppo_vs_baseline_percent"
] = (
    merged["ppo_minus_baseline_km"]
    / merged["baseline_distance_km"]
    * 100
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT = "rl_unseen_final_comparison.csv"

merged.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(f"1. {OUTPUT}")

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

if ppo_mean < baseline_mean:

    print(
        "PPO BEATS THE DIRECT-NAVIGATION BASELINE."
    )

else:

    print(
        "PPO DOES NOT YET BEAT THE "
        "DIRECT-NAVIGATION BASELINE."
    )

print(
    f"\nHistorical : {historical:.2f} km"
)

print(
    f"Baseline   : {baseline_mean:.2f} km"
)

print(
    f"PPO        : {ppo_mean:.2f} km"
)

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)