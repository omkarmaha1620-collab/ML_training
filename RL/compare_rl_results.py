import pandas as pd

PPO_FILE = "rl_unseen_vessel_results.csv"
BASELINE_FILE = "rl_baseline_results.csv"

OUTPUT_FILE = "rl_final_comparison.csv"


print("=" * 70)
print("FINAL AIS-ONLY ROUTE COMPARISON")
print("=" * 70)


# ============================================================
# LOAD RESULTS
# ============================================================

ppo = pd.read_csv(PPO_FILE)

baseline = pd.read_csv(
    BASELINE_FILE
)


# ============================================================
# CONVERT TIMESTAMPS
# ============================================================

ppo["route_start_time"] = pd.to_datetime(
    ppo["route_start_time"]
)

baseline["route_start_time"] = pd.to_datetime(
    baseline["route_start_time"]
)


# ============================================================
# MERGE SAME TEST ROUTES
# ============================================================

merged = pd.merge(
    ppo,
    baseline,
    on=[
        "MMSI",
        "route_start_time"
    ],
    suffixes=(
        "_ppo",
        "_baseline"
    )
)


print(
    f"\nMatched routes: {len(merged)}"
)


# ============================================================
# MEAN DISTANCE
# ============================================================

historical_mean = (
    merged[
        "historical_route_distance_km_ppo"
    ].mean()
)

baseline_mean = (
    merged[
        "baseline_distance_km"
    ].mean()
)

ppo_mean = (
    merged[
        "ppo_distance_km"
    ].mean()
)


print("\nMEAN ROUTE DISTANCE")
print("-" * 50)

print(
    f"Historical AIS : "
    f"{historical_mean:.2f} km"
)

print(
    f"Baseline       : "
    f"{baseline_mean:.2f} km"
)

print(
    f"PPO            : "
    f"{ppo_mean:.2f} km"
)


# ============================================================
# IMPROVEMENT VS HISTORICAL
# ============================================================

baseline_improvement = (
    (baseline_mean - historical_mean)
    / historical_mean
    * 100.0
)

ppo_improvement = (
    (ppo_mean - historical_mean)
    / historical_mean
    * 100.0
)


print("\nIMPROVEMENT VS HISTORICAL")
print("-" * 50)

print(
    f"Baseline : "
    f"{baseline_improvement:.2f}%"
)

print(
    f"PPO      : "
    f"{ppo_improvement:.2f}%"
)


# ============================================================
# PPO VS BASELINE
# ============================================================

ppo_vs_baseline = (
    (ppo_mean - baseline_mean)
    / baseline_mean
    * 100.0
)


print("\nPPO VS BASELINE")
print("-" * 50)

print(
    f"PPO change vs baseline : "
    f"{ppo_vs_baseline:.2f}%"
)


ppo_better = (
    merged["ppo_distance_km"]
    < merged["baseline_distance_km"]
).sum()

ppo_worse = (
    merged["ppo_distance_km"]
    > merged["baseline_distance_km"]
).sum()

equal = (
    merged["ppo_distance_km"]
    == merged["baseline_distance_km"]
).sum()


print(
    f"PPO shorter : "
    f"{ppo_better}"
)

print(
    f"PPO longer  : "
    f"{ppo_worse}"
)

print(
    f"Equal       : "
    f"{equal}"
)


# ============================================================
# SUCCESS RATE
# ============================================================

ppo_success = (
    merged[
        "destination_reached"
    ].mean()
    * 100.0
)

baseline_success = (
    merged[
        "baseline_destination_reached"
    ].mean()
    * 100.0
)


print("\nDESTINATION SUCCESS")
print("-" * 50)

print(
    f"Baseline : "
    f"{baseline_success:.2f}%"
)

print(
    f"PPO      : "
    f"{ppo_success:.2f}%"
)


# ============================================================
# DISTANCE SAVINGS
# ============================================================

merged[
    "ppo_vs_baseline_difference_km"
] = (
    merged["ppo_distance_km"]
    - merged["baseline_distance_km"]
)

merged[
    "ppo_vs_historical_difference_km"
] = (
    merged["ppo_distance_km"]
    - merged[
        "historical_route_distance_km_ppo"
    ]
)

merged[
    "baseline_vs_historical_difference_km"
] = (
    merged["baseline_distance_km"]
    - merged[
        "historical_route_distance_km_ppo"
    ]
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(
    f"1. {OUTPUT_FILE}"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("CURRENT AIS-ONLY RL CONCLUSION")
print("=" * 70)

if ppo_mean < baseline_mean:

    print(
        "\nPPO currently outperforms "
        "the direct-navigation baseline."
    )

else:

    print(
        "\nPPO currently does NOT outperform "
        "the direct-navigation baseline."
    )

print(
    "\nHistorical : "
    f"{historical_mean:.2f} km"
)

print(
    "Baseline   : "
    f"{baseline_mean:.2f} km"
)

print(
    "PPO        : "
    f"{ppo_mean:.2f} km"
)

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)