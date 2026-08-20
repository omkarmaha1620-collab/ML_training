import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from route_environment import RouteOptimizationEnv


MODEL_PATH = "rl_model/ppo_route_optimizer.zip"
TEST_ROUTES = "rl_routes_test.csv"
AIS_FILE = "rl_training_dataset_v2.csv"
OUTPUT_FILE = "rl_unseen_vessel_results.csv"

SEED = 1000


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

        total_reward += float(reward)

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

        "final_distance_km":
            step_info["distance_to_destination_km"],

        "destination_reached":
            step_info["route_completed"],

        "total_turning_degrees":
            step_info["total_turning_degrees"]
    }


def main():

    print("=" * 70)
    print("UNSEEN-VESSEL PPO ROUTE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(TEST_ROUTES):
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_ROUTES}"
        )

    # --------------------------------------------------------
    # Load test routes
    # --------------------------------------------------------

    test_routes = pd.read_csv(
        TEST_ROUTES
    )

    test_routes["start_time"] = pd.to_datetime(
        test_routes["start_time"]
    )

    test_routes["end_time"] = pd.to_datetime(
        test_routes["end_time"]
    )

    test_vessels = set(
        test_routes["MMSI"].unique()
    )

    print("\nTEST DATASET")
    print(
        f"Test routes   : {len(test_routes):,}"
    )
    print(
        f"Test vessels  : {len(test_vessels):,}"
    )

    # --------------------------------------------------------
    # Load training vessels
    # --------------------------------------------------------

    train_routes = pd.read_csv(
        "rl_routes_train.csv",
        usecols=["MMSI"]
    )

    train_vessels = set(
        train_routes["MMSI"].unique()
    )

    overlap = (
        train_vessels
        & test_vessels
    )

    print(
        f"Train/test overlap : {len(overlap)}"
    )

    if len(overlap) != 0:
        raise ValueError(
            "ERROR: Training and test vessels overlap."
        )

    print(
        "PASS: Test vessels are completely unseen."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading PPO model...")

    model = PPO.load(
        MODEL_PATH
    )

    print(
        "PPO model loaded successfully."
    )

    # --------------------------------------------------------
    # Create environment using TEST routes
    # --------------------------------------------------------

    print(
        "\nCreating test environment..."
    )

    env = RouteOptimizationEnv(
        route_file=TEST_ROUTES,
        ais_file=AIS_FILE
    )

    print(
        "Test environment created."
    )

    # --------------------------------------------------------
    # Evaluate every test route
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATING ALL UNSEEN TEST ROUTES")
    print("=" * 70)

    results = []

    # We explicitly reset the environment to each
    # test route instead of randomly sampling only 100.

    for i in range(len(test_routes)):

        # Select exact test route
        env.current_route = test_routes.iloc[i]

        # Reset normally to initialize environment
        #
        # The seed is changed for each route.
        observation, info = env.reset(
            seed=SEED + i
        )

        # Because reset randomly selects a route, repeat until
        # the selected route corresponds to the intended test
        # route. Instead, we will use the environment's internal
        # route selection with a deterministic seed below.

        # ----------------------------------------------------
        # Direct deterministic setup
        # ----------------------------------------------------

        route = test_routes.iloc[i]

        env.current_route = route

        env.current_lat = float(
            route["start_lat"]
        )

        env.current_lon = float(
            route["start_lon"]
        )

        env.destination_lat = float(
            route["end_lat"]
        )

        env.destination_lon = float(
            route["end_lon"]
        )

        # Historical starting heading
        mmsi = str(
            int(route["MMSI"])
        )

        start_time = pd.Timestamp(
            route["start_time"]
        )

        key = (
            mmsi
            + "_"
            + str(start_time)
        )

        historical_heading = (
            env.heading_lookup.get(key)
        )

        if historical_heading is not None:

            env.initial_heading = float(
                historical_heading
            )

            if not (
                0 <= env.initial_heading <= 360
            ):
                historical_heading = None

        if historical_heading is None:

            from route_environment import bearing_degrees

            env.initial_heading = bearing_degrees(
                env.current_lat,
                env.current_lon,
                env.destination_lat,
                env.destination_lon
            )

        env.current_heading = (
            env.initial_heading
        )

        env.current_speed = float(
            route["average_speed_knots"]
        )

        if env.current_speed <= 0:
            env.current_speed = 10.0

        env.previous_distance = (
            env._get_distance_to_destination()
            if hasattr(
                env,
                "_get_distance_to_destination"
            )
            else np.nan
        )

        # Calculate initial distance directly
        from route_environment import haversine_km

        env.previous_distance = haversine_km(
            env.current_lat,
            env.current_lon,
            env.destination_lat,
            env.destination_lon
        )

        env.steps = 0
        env.total_distance = 0.0
        env.total_turning = 0.0

        observation = env._get_observation()

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

            total_reward += float(reward)

            done = (
                terminated
                or truncated
            )

        results.append({
            "MMSI": int(
                route["MMSI"]
            ),

            "route_start_time":
                route["start_time"],

            "route_end_time":
                route["end_time"],

            "initial_distance_km":
                haversine_km(
                    route["start_lat"],
                    route["start_lon"],
                    route["end_lat"],
                    route["end_lon"]
                ),

            "historical_route_distance_km":
                float(
                    route["route_distance_km"]
                ),

            "historical_duration_hours":
                float(
                    route["duration_hours"]
                ),

            "ppo_distance_km":
                float(
                    step_info["total_distance_km"]
                ),

            "ppo_steps":
                int(
                    step_info["steps"]
                ),

            "ppo_reward":
                total_reward,

            "final_distance_km":
                float(
                    step_info[
                        "distance_to_destination_km"
                    ]
                ),

            "destination_reached":
                bool(
                    step_info[
                        "route_completed"
                    ]
                ),

            "total_turning_degrees":
                float(
                    step_info[
                        "total_turning_degrees"
                    ]
                )
        })

        if (i + 1) % 10 == 0:

            print(
                f"Evaluated "
                f"{i + 1}/{len(test_routes)} routes"
            )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    results_df[
        "distance_difference_km"
    ] = (
        results_df["ppo_distance_km"]
        - results_df[
            "historical_route_distance_km"
        ]
    )

    results_df[
        "distance_change_percent"
    ] = (
        results_df[
            "distance_difference_km"
        ]
        / results_df[
            "historical_route_distance_km"
        ]
        * 100.0
    )

    results_df[
        "absolute_distance_savings_km"
    ] = (
        results_df[
            "historical_route_distance_km"
        ]
        - results_df[
            "ppo_distance_km"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    successful = (
        results_df[
            "destination_reached"
        ].sum()
    )

    success_rate = (
        successful
        / len(results_df)
        * 100.0
    )

    mean_historical = (
        results_df[
            "historical_route_distance_km"
        ].mean()
    )

    mean_ppo = (
        results_df[
            "ppo_distance_km"
        ].mean()
    )

    mean_difference = (
        results_df[
            "distance_difference_km"
        ].mean()
    )

    percentage_using_means = (
        (
            mean_ppo
            - mean_historical
        )
        / mean_historical
        * 100.0
    )

    mean_individual_change = (
        results_df[
            "distance_change_percent"
        ].mean()
    )

    better = (
        results_df[
            "distance_difference_km"
        ] < 0
    ).sum()

    worse = (
        results_df[
            "distance_difference_km"
        ] > 0
    ).sum()

    equal = (
        results_df[
            "distance_difference_km"
        ] == 0
    ).sum()

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNSEEN-VESSEL PPO RESULTS")
    print("=" * 70)

    print(
        f"\nTest routes                 : "
        f"{len(results_df):,}"
    )

    print(
        f"Unseen vessels              : "
        f"{results_df['MMSI'].nunique():,}"
    )

    print(
        f"Successful destinations      : "
        f"{successful:,}"
    )

    print(
        f"Success rate                : "
        f"{success_rate:.2f}%"
    )

    print("\nDISTANCE COMPARISON")

    print(
        f"Historical mean distance    : "
        f"{mean_historical:.2f} km"
    )

    print(
        f"PPO mean distance           : "
        f"{mean_ppo:.2f} km"
    )

    print(
        f"Mean difference             : "
        f"{mean_difference:.2f} km"
    )

    print(
        f"Improvement using means     : "
        f"{percentage_using_means:.2f}%"
    )

    print(
        f"Mean individual change      : "
        f"{mean_individual_change:.2f}%"
    )

    print("\nROUTE COMPARISON")

    print(
        f"PPO shorter                : "
        f"{better}"
    )

    print(
        f"PPO longer                 : "
        f"{worse}"
    )

    print(
        f"Equal                      : "
        f"{equal}"
    )

    print("\nPPO PERFORMANCE")

    print(
        f"Mean reward                : "
        f"{results_df['ppo_reward'].mean():.2f}"
    )

    print(
        f"Mean steps                 : "
        f"{results_df['ppo_steps'].mean():.2f}"
    )

    print(
        f"Mean final distance        : "
        f"{results_df['final_distance_km'].mean():.2f} km"
    )

    print(
        f"Mean total turning         : "
        f"{results_df['total_turning_degrees'].mean():.2f}°"
    )

    print("\nBEST PPO IMPROVEMENTS")

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

    print("\nWORST PPO RESULTS")

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
            "distance_change_percent",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(
        f"1. {OUTPUT_FILE}"
    )

    env.close()

    print("\n" + "=" * 70)
    print("UNSEEN-VESSEL EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()