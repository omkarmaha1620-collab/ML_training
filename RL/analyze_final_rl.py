import pandas as pd
import numpy as np

df = pd.read_csv("rl_unseen_final_comparison.csv")

ppo = df["ppo_distance_km"]
baseline = df["baseline_distance_km"]

diff = ppo - baseline

print("=" * 70)
print("FINAL PPO VS BASELINE STATISTICAL ANALYSIS")
print("=" * 70)

print(f"\nRoutes: {len(df)}")

print("\nDISTANCE")
print("-" * 50)
print(f"Baseline mean : {baseline.mean():.4f} km")
print(f"PPO mean      : {ppo.mean():.4f} km")
print(f"Mean diff     : {diff.mean():.4f} km")
print(f"Median diff   : {diff.median():.4f} km")
print(f"Std diff      : {diff.std():.4f} km")

print("\nROUTE COUNTS")
print("-" * 50)
print(f"PPO shorter   : {(diff < 0).sum()}")
print(f"PPO longer    : {(diff > 0).sum()}")
print(f"Equal         : {(diff == 0).sum()}")

print("\nPERCENTAGES")
print("-" * 50)
print(f"PPO better    : {(diff < 0).mean() * 100:.2f}%")
print(f"PPO worse     : {(diff > 0).mean() * 100:.2f}%")
print(f"Equal         : {(diff == 0).mean() * 100:.2f}%")

print("\nRANGE")
print("-" * 50)
print(f"Best PPO diff : {diff.min():.4f} km")
print(f"Worst PPO diff: {diff.max():.4f} km")

print("\nSUCCESS")
print("-" * 50)
print(
    f"PPO success      : "
    f"{df['destination_reached'].mean() * 100:.2f}%"
)

print(
    f"Baseline success : "
    f"{df['baseline_destination_reached'].mean() * 100:.2f}%"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)