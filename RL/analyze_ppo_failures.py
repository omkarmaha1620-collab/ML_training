import pandas as pd

FILE = "rl_unseen_final_comparison.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("PPO FAILURE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Route-level PPO vs baseline difference
# ------------------------------------------------------------

df["difference_km"] = (
    df["ppo_distance_km"]
    - df["baseline_distance_km"]
)

df["difference_percent"] = (
    df["difference_km"]
    / df["baseline_distance_km"]
    * 100
)

# ------------------------------------------------------------
# Overall
# ------------------------------------------------------------

print("\nOVERALL")
print("-" * 50)

print(f"Routes              : {len(df)}")
print(
    f"PPO shorter         : "
    f"{(df['difference_km'] < 0).sum()}"
)
print(
    f"PPO longer          : "
    f"{(df['difference_km'] > 0).sum()}"
)
print(
    f"Equal               : "
    f"{(df['difference_km'] == 0).sum()}"
)

print(
    f"Mean PPO distance   : "
    f"{df['ppo_distance_km'].mean():.2f} km"
)

print(
    f"Mean baseline       : "
    f"{df['baseline_distance_km'].mean():.2f} km"
)

print(
    f"Mean difference     : "
    f"{df['difference_km'].mean():.2f} km"
)

# ------------------------------------------------------------
# PPO turning
# ------------------------------------------------------------

print("\nTURNING")
print("-" * 50)

print(
    f"Mean PPO turning    : "
    f"{df['total_turning_degrees'].mean():.2f}°"
)

print(
    f"Median PPO turning  : "
    f"{df['total_turning_degrees'].median():.2f}°"
)

print(
    f"Maximum PPO turning : "
    f"{df['total_turning_degrees'].max():.2f}°"
)

# ------------------------------------------------------------
# Worst routes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10 WORST PPO ROUTES")
print("=" * 70)

worst = df.sort_values(
    "difference_km",
    ascending=False
)

columns = [
    "MMSI",
    "historical_route_distance_km_ppo",
    "baseline_distance_km",
    "ppo_distance_km",
    "difference_km",
    "difference_percent",
    "ppo_steps",
    "total_turning_degrees",
    "destination_reached"
]

print(
    worst[columns]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Best routes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10 BEST PPO ROUTES")
print("=" * 70)

best = df.sort_values(
    "difference_km",
    ascending=True
)

print(
    best[columns]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

worst.to_csv(
    "rl_ppo_failure_analysis.csv",
    index=False
)

print("\nSaved:")
print("1. rl_ppo_failure_analysis.csv")

print("\n" + "=" * 70)
print("FAILURE ANALYSIS COMPLETED")
print("=" * 70)