import pandas as pd

df = pd.read_csv("rl_unseen_final_comparison.csv")

historical = df["historical_route_distance_km_ppo"].mean()
baseline = df["baseline_distance_km"].mean()
ppo = df["ppo_distance_km"].mean()

diff = ppo - baseline

metrics = pd.DataFrame({
    "Metric": [
        "Unseen test routes",
        "Unseen test vessels",
        "Historical AIS mean distance (km)",
        "Direct baseline mean distance (km)",
        "PPO mean distance (km)",
        "PPO improvement vs historical (%)",
        "PPO change vs baseline (%)",
        "PPO shorter routes",
        "PPO longer routes",
        "Equal routes",
        "PPO destination success (%)",
        "Baseline destination success (%)",
        "PPO mean turning (degrees)",
        "PPO median turning (degrees)"
    ],
    "Value": [
        len(df),
        df["MMSI"].nunique(),
        historical,
        baseline,
        ppo,
        ((historical - ppo) / historical) * 100,
        (diff / baseline) * 100,
        (df["ppo_distance_km"] < df["baseline_distance_km"]).sum(),
        (df["ppo_distance_km"] > df["baseline_distance_km"]).sum(),
        (df["ppo_distance_km"] == df["baseline_distance_km"]).sum(),
        df["destination_reached"].mean() * 100,
        df["baseline_destination_reached"].mean() * 100,
        df["total_turning_degrees"].mean(),
        df["total_turning_degrees"].median()
    ]
})

metrics.to_csv("rl_final_metrics.csv", index=False)

print("=" * 70)
print("FINAL RL METRICS CREATED")
print("=" * 70)
print(metrics.to_string(index=False))
print()
print("Saved:")
print("rl_final_metrics.csv")
