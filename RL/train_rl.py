import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from route_environment import RouteOptimizationEnv


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = "rl_model"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_route_optimizer")

TOTAL_TIMESTEPS = 100_000

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PPO RL ROUTE OPTIMIZATION TRAINING")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nCreating route optimization environment...")

    env = RouteOptimizationEnv()
    env = Monitor(env)

    print("Environment created.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    print("\n" + "=" * 70)
    print("CREATING PPO AGENT")
    print("=" * 70)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=SEED,
    )

    print("\n" + "=" * 70)
    print("STARTING PPO TRAINING")
    print("=" * 70)

    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print("Please wait...\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    model.save(MODEL_PATH)

    print(f"\nModel saved to:")
    print(f"{MODEL_PATH}.zip")

    env.close()

    print("\nPPO ROUTE OPTIMIZATION TRAINING COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()