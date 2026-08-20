import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv("rl_unseen_final_comparison.csv")

historical = df["historical_route_distance_km_ppo"]
baseline = df["baseline_distance_km"]
ppo = df["ppo_distance_km"]

# ============================================================
# 1. MEAN ROUTE DISTANCE
# ============================================================

means = [
    historical.mean(),
    baseline.mean(),
    ppo.mean()
]

labels = [
    "Historical AIS",
    "Direct Baseline",
    "PPO"
]

plt.figure(figsize=(8, 5))
plt.bar(labels, means)
plt.ylabel("Mean Route Distance (km)")
plt.title("Mean Route Distance Comparison")
plt.tight_layout()
plt.savefig("rl_mean_distance_comparison.png", dpi=300)
plt.close()

# ============================================================
# 2. PPO VS BASELINE — ALL UNSEEN ROUTES
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(df))

plt.plot(
    x,
    baseline,
    label="Direct Baseline"
)

plt.plot(
    x,
    ppo,
    label="PPO"
)

plt.xlabel("Unseen Test Route")
plt.ylabel("Route Distance (km)")
plt.title("PPO vs Direct Baseline on Unseen Vessels")
plt.legend()
plt.tight_layout()
plt.savefig("rl_ppo_vs_baseline.png", dpi=300)
plt.close()

# ============================================================
# 3. PPO IMPROVEMENT VS HISTORICAL
# ============================================================

improvement = (
    (historical - ppo)
    / historical
) * 100

plt.figure(figsize=(10, 6))

plt.hist(
    improvement,
    bins=15
)

plt.axvline(
    improvement.mean(),
    linestyle="--",
    label=f"Mean = {improvement.mean():.2f}%"
)

plt.xlabel("PPO Improvement vs Historical AIS (%)")
plt.ylabel("Number of Routes")
plt.title("PPO Improvement Over Historical AIS")
plt.legend()
plt.tight_layout()
plt.savefig("rl_ppo_improvement_distribution.png", dpi=300)
plt.close()

# ============================================================
# 4. TURNING DISTRIBUTION
# ============================================================

if "total_turning_degrees" in df.columns:

    turning = df["total_turning_degrees"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        turning,
        bins=15
    )

    plt.axvline(
        turning.mean(),
        linestyle="--",
        label=f"Mean = {turning.mean():.2f}°"
    )

    plt.axvline(
        turning.median(),
        linestyle=":",
        label=f"Median = {turning.median():.2f}°"
    )

    plt.xlabel("Total Turning (degrees)")
    plt.ylabel("Number of Routes")
    plt.title("PPO Route Turning Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl_turning_distribution.png", dpi=300)
    plt.close()

print("=" * 70)
print("FINAL RL PLOTS CREATED")
print("=" * 70)

print("\nSaved:")
print("1. rl_mean_distance_comparison.png")
print("2. rl_ppo_vs_baseline.png")
print("3. rl_ppo_improvement_distribution.png")
print("4. rl_turning_distribution.png")

print("\nPlot generation completed.")