import pandas as pd
import numpy as np

INPUT_FILE = "rl_route_episodes_final.csv"

TRAIN_FILE = "rl_routes_train.csv"
TEST_FILE = "rl_routes_test.csv"

TEST_VESSEL_FRACTION = 0.20
SEED = 42


print("=" * 70)
print("CREATING VESSEL-BASED TRAIN / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal routes  : {len(df):,}")
print(f"Total vessels : {df['MMSI'].nunique():,}")

# ------------------------------------------------------------
# Get unique vessels
# ------------------------------------------------------------

vessels = df["MMSI"].unique()

rng = np.random.default_rng(SEED)

rng.shuffle(vessels)

n_test = max(
    1,
    int(len(vessels) * TEST_VESSEL_FRACTION)
)

test_vessels = set(
    vessels[:n_test]
)

train_vessels = set(
    vessels[n_test:]
)

# ------------------------------------------------------------
# Split by vessel
# ------------------------------------------------------------

train_df = df[
    df["MMSI"].isin(train_vessels)
].copy()

test_df = df[
    df["MMSI"].isin(test_vessels)
].copy()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(
    f"\nTraining vessels : "
    f"{train_df['MMSI'].nunique():,}"
)

print(
    f"Training routes  : "
    f"{len(train_df):,}"
)

print(
    f"Test vessels     : "
    f"{test_df['MMSI'].nunique():,}"
)

print(
    f"Test routes      : "
    f"{len(test_df):,}"
)

# Verify no vessel overlap

overlap = (
    set(train_df["MMSI"])
    & set(test_df["MMSI"])
)

print(
    f"\nVessel overlap   : "
    f"{len(overlap)}"
)

if len(overlap) == 0:
    print("PASS: No vessel appears in both sets.")
else:
    print("WARNING: Vessel overlap detected.")

print("\nSaved:")
print(f"1. {TRAIN_FILE}")
print(f"2. {TEST_FILE}")

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 70)