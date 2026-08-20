import pandas as pd
import numpy as np

print("=" * 70)
print("FINAL RL ROUTE OPTIMIZATION REPORT")
print("=" * 70)

# ------------------------------------------------------------
# LOAD FINAL UNSEEN-VESSEL RESULTS
# ------------------------------------------------------------

df = pd.read_csv("rl_unseen_final_comparison.csv")

# ------------------------------------------------------------
# DISTANCE
# ------------------------------------------------------------

historical = df["historical_route_distance_km_ppo"]
ppo = df["ppo_distance_km"]
baseline = df["baseline_distance_km"]

ppo_vs_historical = (
    (ppo.mean() - historical.mean())
    / historical.mean()
) * 100

ppo_vs_baseline = (
    (ppo.mean() - baseline.mean())
    / baseline.mean()
) * 100

diff = ppo - baseline

print("\nDATASET")
print("-" * 50)
print(f"Routes evaluated       : {len(df)}")
print(
    f"Unique test vessels    : "
    f"{df['MMSI'].nunique()}"
)

print("\nMEAN ROUTE DISTANCE")
print("-" * 50)
print(f"Historical AIS         : {historical.mean():.4f} km")
print(f"Direct baseline        : {baseline.mean():.4f} km")
print(f"PPO                    : {ppo.mean():.4f} km")

print("\nPPO VS HISTORICAL")
print("-" * 50)
print(f"Difference             : {ppo.mean() - historical.mean():.4f} km")
print(f"Improvement            : {-ppo_vs_historical:.2f}%")

print("\nPPO VS BASELINE")
print("-" * 50)
print(f"Difference             : {diff.mean():.4f} km")
print(f"Change                 : {ppo_vs_baseline:+.2f}%")

print("\nROUTE COMPARISON")
print("-" * 50)
print(f"PPO shorter            : {(diff < 0).sum()}")
print(f"PPO longer             : {(diff > 0).sum()}")
print(f"Equal                  : {(diff == 0).sum()}")

print(
    f"PPO shorter (%)        : "
    f"{(diff < 0).mean() * 100:.2f}%"
)

print(
    f"PPO longer (%)         : "
    f"{(diff > 0).mean() * 100:.2f}%"
)

print(
    f"Equal (%)              : "
    f"{(diff == 0).mean() * 100:.2f}%"
)

print("\nDESTINATION SUCCESS")
print("-" * 50)

ppo_success = (
    df["destination_reached"].mean() * 100
)

baseline_success = (
    df["baseline_destination_reached"].mean() * 100
)

print(f"PPO                    : {ppo_success:.2f}%")
print(f"Baseline               : {baseline_success:.2f}%")

print("\nPPO TURNING")
print("-" * 50)

if "total_turning_degrees" in df.columns:
    print(
        f"Mean turning           : "
        f"{df['total_turning_degrees'].mean():.2f}°"
    )

    print(
        f"Median turning         : "
        f"{df['total_turning_degrees'].median():.2f}°"
    )

print("\nFINAL VERDICT")
print("-" * 50)

if ppo.mean() < baseline.mean():
    print("PPO OUTPERFORMS THE DIRECT-NAVIGATION BASELINE.")
elif np.isclose(ppo.mean(), baseline.mean(), atol=0.05):
    print(
        "PPO MATCHES THE DIRECT-NAVIGATION BASELINE "
        "WITHIN 0.05 KM."
    )
else:
    print(
        "PPO DOES NOT OUTPERFORM THE DIRECT-NAVIGATION "
        "BASELINE."
    )

print("\n" + "=" * 70)
print("FINAL RL REPORT COMPLETED")
print("=" * 70)