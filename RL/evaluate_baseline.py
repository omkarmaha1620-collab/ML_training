import os
import numpy as np
import pandas as pd

from route_environment import haversine_km, bearing_degrees


TEST_FILE = "rl_routes_test.csv"
OUTPUT_FILE = "rl_baseline_results.csv"

STEP_DISTANCE_KM = 1.0
DESTINATION_THRESHOLD_KM = 2.0
MAX_STEPS = 300


def normalize_angle(angle):
    return (angle + 360.0) % 360.0


def angle_difference(a, b):
    return abs((b - a + 180.0) % 360.0 - 180.0)


def move_point(lat, lon, heading, distance_km):

    heading_rad = np.radians(heading)

    delta_lat = (
        distance_km
        * np.cos(heading_rad)
        / 111.0
    )

    longitude_scale = (
        111.0 * np.cos(np.radians(lat))
    )

    if abs(longitude_scale) < 1e-6:
        longitude_scale = 111.0

    delta_lon = (
        distance_km
        * np.sin(heading_rad)
        / longitude_scale
    )

    lat += delta_lat
    lon += delta_lon

    lat = np.clip(lat, -90.0, 90.0)
    lon = ((lon + 180.0) % 360.0) - 180.0

    return lat, lon


def run_baseline(route):

    lat = float(route["start_lat"])
    lon = float(route["start_lon"])

    destination_lat = float(route["end_lat"])
    destination_lon = float(route["end_lon"])

    initial_distance = haversine_km(
        lat,
        lon,
        destination_lat,
        destination_lon
    )

    total_distance = 0.0
    total_turning = 0.0
    steps = 0

    # Simple baseline:
    # At every step, point directly toward
    # the destination.

    while steps < MAX_STEPS:

        distance = haversine_km(
            lat,
            lon,
            destination_lat,
            destination_lon
        )

        if distance <= DESTINATION_THRESHOLD_KM:
            break

        heading = bearing_degrees(
            lat,
            lon,
            destination_lat,
            destination_lon
        )

        lat, lon = move_point(
            lat,
            lon,
            heading,
            STEP_DISTANCE_KM
        )

        total_distance += STEP_DISTANCE_KM
        steps += 1

    final_distance = haversine_km(
        lat,
        lon,
        destination_lat,
        destination_lon
    )

    reached = (
        final_distance
        <= DESTINATION_THRESHOLD_KM
    )

    return {
        "MMSI": int(route["MMSI"]),

        "route_start_time":
            route["start_time"],

        "route_end_time":
            route["end_time"],

        "historical_route_distance_km":
            float(route["route_distance_km"]),

        "historical_duration_hours":
            float(route["duration_hours"]),

        "baseline_distance_km":
            total_distance,

        "baseline_steps":
            steps,

        "baseline_final_distance_km":
            final_distance,

        "baseline_destination_reached":
            reached,

        "baseline_turning_degrees":
            total_turning
    }


def main():

    print("=" * 70)
    print("BASELINE ROUTE OPTIMIZATION EVALUATION")
    print("=" * 70)

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Missing: {TEST_FILE}"
        )

    routes = pd.read_csv(TEST_FILE)

    print(
        f"\nTest routes   : {len(routes):,}"
    )

    print(
        f"Test vessels  : "
        f"{routes['MMSI'].nunique():,}"
    )

    results = []

    for i, (_, route) in enumerate(
        routes.iterrows()
    ):

        results.append(
            run_baseline(route)
        )

        if (i + 1) % 10 == 0:
            print(
                f"Evaluated "
                f"{i + 1}/{len(routes)} routes"
            )

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Compare baseline with historical route
    # --------------------------------------------------------

    df["baseline_difference_km"] = (
        df["baseline_distance_km"]
        - df["historical_route_distance_km"]
    )

    df["baseline_change_percent"] = (
        df["baseline_difference_km"]
        / df["historical_route_distance_km"]
        * 100.0
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    historical_mean = (
        df["historical_route_distance_km"]
        .mean()
    )

    baseline_mean = (
        df["baseline_distance_km"]
        .mean()
    )

    improvement = (
        (baseline_mean - historical_mean)
        / historical_mean
        * 100.0
    )

    success_rate = (
        df["baseline_destination_reached"]
        .mean()
        * 100.0
    )

    shorter = (
        df["baseline_difference_km"] < 0
    ).sum()

    longer = (
        df["baseline_difference_km"] > 0
    ).sum()

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"\nRoutes evaluated        : "
        f"{len(df):,}"
    )

    print(
        f"Vessels evaluated       : "
        f"{df['MMSI'].nunique():,}"
    )

    print(
        f"Success rate             : "
        f"{success_rate:.2f}%"
    )

    print("\nDISTANCE")

    print(
        f"Historical mean         : "
        f"{historical_mean:.2f} km"
    )

    print(
        f"Baseline mean           : "
        f"{baseline_mean:.2f} km"
    )

    print(
        f"Mean difference         : "
        f"{baseline_mean - historical_mean:.2f} km"
    )

    print(
        f"Mean change             : "
        f"{improvement:.2f}%"
    )

    print("\nROUTE COMPARISON")

    print(
        f"Baseline shorter        : "
        f"{shorter}"
    )

    print(
        f"Baseline longer         : "
        f"{longer}"
    )

    print("\nSaved:")
    print(
        f"1. {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()