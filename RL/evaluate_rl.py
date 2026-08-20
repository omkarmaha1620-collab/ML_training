import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from route_environment import RouteOptimizationEnv


MODEL_PATH = "rl_model/ppo_route_optimizer.zip"
OUTPUT_FILE = "rl_evaluation_results.csv"

N_TEST_ROUTES = 100
SEED = 123


def run_episode(model, env, seed):

    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    done = False

    while not done:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            step_info
        ) = env.step(action)

        total_reward += reward

        done = terminated or truncated

    return {
        "MMSI": info["MMSI"],
        "route_start_time": info["route_start_time"],
        "route_end_time": info["route_end_time"],

        "initial_distance_km":
            info["initial_distance_km"],

        "historical_route_distance_km":
            info["historical_route_distance_km"],

        "historical_duration_hours":
            info["historical_duration_hours"],

        "ppo_distance_km":
            step_info["total_distance_km"],

        "ppo_steps":
            step_info["steps"],

        "ppo_reward":
            total_reward,

        "destination_distance_km":
            step_info["distance_to_destination_km"],

        "destination_reached":
            step_info["route_completed"],

        "total_turning_degrees":
            step_info["total_turning_degrees"]
    }


def main():

    print("=" * 70)
    print("PPO RL ROUTE OPTIMIZATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLoading PPO model...")

    model = PPO.load(
        MODEL_PATH
    )

    print("PPO model loaded successfully.")

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    print("\nCreating environment...")

    env = RouteOptimizationEnv()

    print("Environment ready.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING TEST ROUTES")
    print("=" * 70)

    results = []

    for i in range(N_TEST_ROUTES):

        result = run_episode(
            model,
            env,
            SEED + i
        )

        results.append(result)

        if (i + 1) % 10 == 0:

            print(
                f"Evaluated "
                f"{i + 1}/{N_TEST_ROUTES} routes"
            )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    results_df["distance_difference_km"] = (
        results_df["ppo_distance_km"]
        - results_df["historical_route_distance_km"]
    )

    results_df["distance_change_percent"] = (
        results_df["distance_difference_km"]
        / results_df["historical_route_distance_km"]
        * 100.0
    )

    results_df["ppo_efficiency"] = np.where(
        results_df["ppo_distance_km"] > 0,
        results_df["initial_distance_km"]
        / results_df["ppo_distance_km"],
        0.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PPO EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nRoutes evaluated      : "
        f"{len(results_df):,}"
    )

    print(
        f"Successful routes     : "
        f"{results_df['destination_reached'].sum():,}"
    )

    success_rate = (
        results_df["destination_reached"].mean()
        * 100.0
    )

    print(
        f"Success rate          : "
        f"{success_rate:.2f}%"
    )

    print("\nDistance comparison:")

    print(
        f"Historical mean       : "
        f"{results_df['historical_route_distance_km'].mean():.2f} km"
    )

    print(
        f"PPO mean              : "
        f"{results_df['ppo_distance_km'].mean():.2f} km"
    )

    print(
        f"Mean difference       : "
        f"{results_df['distance_difference_km'].mean():.2f} km"
    )

    print(
        f"Mean change           : "
        f"{results_df['distance_change_percent'].mean():.2f}%"
    )

    print("\nEpisode statistics:")

    print(
        f"Mean PPO reward       : "
        f"{results_df['ppo_reward'].mean():.2f}"
    )

    print(
        f"Mean PPO steps        : "
        f"{results_df['ppo_steps'].mean():.2f}"
    )

    print(
        f"Mean final distance   : "
        f"{results_df['destination_distance_km'].mean():.2f} km"
    )

    print(
        f"Mean total turning    : "
        f"{results_df['total_turning_degrees'].mean():.2f}°"
    )

    # --------------------------------------------------------
    # Better / worse routes
    # --------------------------------------------------------

    better = (
        results_df["distance_difference_km"] < 0
    ).sum()

    worse = (
        results_df["distance_difference_km"] > 0
    ).sum()

    equal = (
        results_df["distance_difference_km"] == 0
    ).sum()

    print("\nRoute comparison:")

    print(
        f"PPO shorter than historical : "
        f"{better:,}"
    )

    print(
        f"PPO longer than historical  : "
        f"{worse:,}"
    )

    print(
        f"Equal                        : "
        f"{equal:,}"
    )

    # --------------------------------------------------------
    # Best routes
    # --------------------------------------------------------

    print("\nBest PPO improvements:")

    print(
        results_df[
            [
                "MMSI",
                "historical_route_distance_km",
                "ppo_distance_km",
                "distance_change_percent",
                "destination_reached"
            ]
        ]
        .sort_values(
            "distance_change_percent"
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(f"1. {OUTPUT_FILE}")

    env.close()

    print("\n" + "=" * 70)
    print("PPO EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()