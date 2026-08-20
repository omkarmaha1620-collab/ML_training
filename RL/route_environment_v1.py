import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ============================================================
# FILES
# ============================================================

ROUTE_FILE = "rl_routes_train.csv"
AIS_FILE = "rl_training_dataset_v2.csv"


# ============================================================
# NAVIGATION HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates."""

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return R * 2.0 * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2."""

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(lon2 - lon1)

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return (bearing + 360.0) % 360.0


def signed_angle_difference(a, b):
    """
    Signed smallest difference from heading a to heading b.
    Result is in [-180, 180].
    """

    return (b - a + 180.0) % 360.0 - 180.0


def angle_difference(a, b):
    """Absolute smallest difference between two headings."""

    return abs(signed_angle_difference(a, b))


# ============================================================
# RL ENVIRONMENT
# ============================================================

class RouteOptimizationEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(
        self,
        route_file=ROUTE_FILE,
        ais_file=AIS_FILE
    ):

        super().__init__()

        # ----------------------------------------------------
        # LOAD ROUTE EPISODES
        # ----------------------------------------------------

        print("Loading route episodes...")

        self.routes = pd.read_csv(route_file)

        if len(self.routes) == 0:
            raise ValueError(
                "Route dataset is empty."
            )

        required_route_columns = [
            "MMSI",
            "start_time",
            "end_time",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "route_distance_km",
            "direct_distance_km",
            "duration_hours",
            "average_speed_knots"
        ]

        missing = [
            c for c in required_route_columns
            if c not in self.routes.columns
        ]

        if missing:
            raise ValueError(
                f"Missing route columns: {missing}"
            )

        # Convert route timestamps
        self.routes["start_time"] = pd.to_datetime(
            self.routes["start_time"]
        )

        self.routes["end_time"] = pd.to_datetime(
            self.routes["end_time"]
        )

        # ----------------------------------------------------
        # LOAD HISTORICAL AIS START HEADINGS
        # ----------------------------------------------------
        #
        # We use the actual COG from rl_training_dataset_v2
        # at each route's starting timestamp.
        #
        # This is loaded ONCE, not during every reset.
        #

        print("Loading historical AIS start headings...")

        ais = pd.read_csv(
            ais_file,
            usecols=[
                "MMSI",
                "TIMESTAMP",
                "COG"
            ]
        )

        ais["datetime"] = pd.to_datetime(
            ais["TIMESTAMP"],
            unit="s"
        )

        # Create lookup key
        ais["key"] = (
            ais["MMSI"].astype(str)
            + "_"
            + ais["datetime"].astype(str)
        )

        # Keep only valid COG
        ais = ais[
            ais["COG"].notna()
        ].copy()

        self.heading_lookup = dict(
            zip(
                ais["key"],
                ais["COG"]
            )
        )

        print(
            f"Historical heading records loaded: "
            f"{len(self.heading_lookup):,}"
        )

        # ----------------------------------------------------
        # ACTION SPACE
        # ----------------------------------------------------
        #
        # 0 = maintain heading
        # 1 = turn left 5°
        # 2 = turn right 5°
        # 3 = turn left 10°
        # 4 = turn right 10°
        #

        self.action_space = spaces.Discrete(5)

        self.turn_amounts = {
            0: 0.0,
            1: -5.0,
            2: 5.0,
            3: -10.0,
            4: 10.0
        }

        # ----------------------------------------------------
        # OBSERVATION SPACE
        # ----------------------------------------------------
        #
        # 0  current latitude
        # 1  current longitude
        # 2  destination latitude
        # 3  destination longitude
        # 4  speed
        # 5  current heading
        # 6  distance to destination
        # 7  bearing to destination
        # 8  heading error
        #

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # SIMULATION SETTINGS
        # ----------------------------------------------------

        self.step_distance_km = 1.0

        self.destination_threshold_km = 2.0

        self.max_steps = 300

        # ----------------------------------------------------
        # EPISODE VARIABLES
        # ----------------------------------------------------

        self.current_route = None

        self.current_lat = 0.0
        self.current_lon = 0.0

        self.destination_lat = 0.0
        self.destination_lon = 0.0

        self.current_heading = 0.0
        self.initial_heading = 0.0

        self.current_speed = 0.0

        self.previous_distance = 0.0

        self.steps = 0
        self.total_distance = 0.0
        self.total_turning = 0.0

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # ----------------------------------------------------
        # Select historical route
        # ----------------------------------------------------

        route_index = self.np_random.integers(
            0,
            len(self.routes)
        )

        self.current_route = (
            self.routes.iloc[route_index]
        )

        # ----------------------------------------------------
        # Historical starting position
        # ----------------------------------------------------

        self.current_lat = float(
            self.current_route["start_lat"]
        )

        self.current_lon = float(
            self.current_route["start_lon"]
        )

        # ----------------------------------------------------
        # Historical destination
        # ----------------------------------------------------

        self.destination_lat = float(
            self.current_route["end_lat"]
        )

        self.destination_lon = float(
            self.current_route["end_lon"]
        )

        # ----------------------------------------------------
        # HISTORICAL INITIAL HEADING
        # ----------------------------------------------------

        mmsi = str(
            int(self.current_route["MMSI"])
        )

        start_time = pd.Timestamp(
            self.current_route["start_time"]
        )

        key = (
            mmsi
            + "_"
            + str(start_time)
        )

        historical_heading = (
            self.heading_lookup.get(key)
        )

        # If exact timestamp exists, use historical COG.
        if historical_heading is not None:

            self.initial_heading = float(
                historical_heading
            )

            # Normalize invalid COG
            if (
                self.initial_heading < 0
                or self.initial_heading > 360
            ):
                historical_heading = None

        # Fallback if exact historical COG wasn't found
        if historical_heading is None:

            self.initial_heading = bearing_degrees(
                self.current_lat,
                self.current_lon,
                self.destination_lat,
                self.destination_lon
            )

        self.current_heading = (
            self.initial_heading
        )

        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------

        self.current_speed = float(
            self.current_route[
                "average_speed_knots"
            ]
        )

        if self.current_speed <= 0:
            self.current_speed = 10.0

        # ----------------------------------------------------
        # Initial distance
        # ----------------------------------------------------

        self.previous_distance = haversine_km(
            self.current_lat,
            self.current_lon,
            self.destination_lat,
            self.destination_lon
        )

        self.steps = 0
        self.total_distance = 0.0
        self.total_turning = 0.0

        observation = self._get_observation()

        info = {
            "MMSI": int(
                self.current_route["MMSI"]
            ),
            "route_start_time": str(
                self.current_route["start_time"]
            ),
            "route_end_time": str(
                self.current_route["end_time"]
            ),
            "initial_heading": self.initial_heading,
            "destination_lat": self.destination_lat,
            "destination_lon": self.destination_lon,
            "initial_distance_km": self.previous_distance,
            "historical_route_distance_km": float(
                self.current_route[
                    "route_distance_km"
                ]
            ),
            "historical_duration_hours": float(
                self.current_route[
                    "duration_hours"
                ]
            )
        }

        return observation, info

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = int(action)

        # ----------------------------------------------------
        # ACTION → HEADING CHANGE
        # ----------------------------------------------------

        turn = self.turn_amounts[action]

        self.current_heading = (
            self.current_heading + turn
        ) % 360.0

        self.total_turning += abs(turn)

        # ----------------------------------------------------
        # MOVE VESSEL
        # ----------------------------------------------------

        distance = self.step_distance_km

        heading_rad = np.radians(
            self.current_heading
        )

        # Latitude movement
        delta_lat = (
            distance
            * np.cos(heading_rad)
            / 111.0
        )

        # Longitude movement
        longitude_scale = (
            111.0
            * np.cos(
                np.radians(
                    self.current_lat
                )
            )
        )

        if abs(longitude_scale) < 1e-6:
            longitude_scale = 111.0

        delta_lon = (
            distance
            * np.sin(heading_rad)
            / longitude_scale
        )

        self.current_lat += delta_lat
        self.current_lon += delta_lon

        # Keep geographic coordinates valid
        self.current_lat = np.clip(
            self.current_lat,
            -90.0,
            90.0
        )

        self.current_lon = (
            (self.current_lon + 180.0)
            % 360.0
        ) - 180.0

        self.total_distance += distance

        self.steps += 1

        # ----------------------------------------------------
        # NEW DISTANCE
        # ----------------------------------------------------

        new_distance = haversine_km(
            self.current_lat,
            self.current_lon,
            self.destination_lat,
            self.destination_lon
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        progress = (
            self.previous_distance
            - new_distance
        )

        # ----------------------------------------------------
        # DESIRED BEARING
        # ----------------------------------------------------

        desired_bearing = bearing_degrees(
            self.current_lat,
            self.current_lon,
            self.destination_lat,
            self.destination_lon
        )

        heading_error = angle_difference(
            self.current_heading,
            desired_bearing
        )

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------
        #
        # Positive:
        #   - progress toward destination
        #   - reaching destination
        #
        # Negative:
        #   - distance travelled
        #   - excessive turning
        #   - heading away from destination
        #

        reward = 0.0

        # Progress reward
        reward += progress * 5.0

        # Movement cost
        reward -= 0.05

        # Turning penalty
        reward -= abs(turn) * 0.01

        # Heading error penalty
        reward -= heading_error * 0.003

        # ----------------------------------------------------
        # DESTINATION
        # ----------------------------------------------------

        terminated = False
        truncated = False

        if new_distance <= self.destination_threshold_km:

            # Strong completion reward
            reward += 100.0

            terminated = True

        # ----------------------------------------------------
        # MAXIMUM STEPS
        # ----------------------------------------------------

        if self.steps >= self.max_steps:

            truncated = True

            # Failure penalty
            reward -= 20.0

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        self.previous_distance = new_distance

        observation = self._get_observation()

        info = {
            "distance_to_destination_km": float(
                new_distance
            ),
            "total_distance_km": float(
                self.total_distance
            ),
            "heading": float(
                self.current_heading
            ),
            "heading_error": float(
                heading_error
            ),
            "steps": self.steps,
            "total_turning_degrees": float(
                self.total_turning
            ),
            "route_completed": terminated
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(self):

        distance = haversine_km(
            self.current_lat,
            self.current_lon,
            self.destination_lat,
            self.destination_lon
        )

        desired_bearing = bearing_degrees(
            self.current_lat,
            self.current_lon,
            self.destination_lat,
            self.destination_lon
        )

        heading_error = angle_difference(
            self.current_heading,
            desired_bearing
        )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        lat_norm = (
            self.current_lat / 90.0
        )

        lon_norm = (
            self.current_lon / 180.0
        )

        destination_lat_norm = (
            self.destination_lat / 90.0
        )

        destination_lon_norm = (
            self.destination_lon / 180.0
        )

        speed_norm = np.clip(
            self.current_speed / 40.0,
            0.0,
            1.0
        )

        heading_norm = (
            self.current_heading / 180.0
        ) - 1.0

        distance_norm = np.clip(
            distance / 500.0,
            0.0,
            1.0
        )

        bearing_norm = (
            desired_bearing / 180.0
        ) - 1.0

        heading_error_norm = (
            heading_error / 180.0
        )

        observation = np.array(
            [
                lat_norm,
                lon_norm,
                destination_lat_norm,
                destination_lon_norm,
                speed_norm,
                heading_norm,
                distance_norm,
                bearing_norm,
                heading_error_norm
            ],
            dtype=np.float32
        )

        return observation

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        print(
            f"Step {self.steps} | "
            f"Position: "
            f"({self.current_lat:.4f}, "
            f"{self.current_lon:.4f}) | "
            f"Destination: "
            f"({self.destination_lat:.4f}, "
            f"{self.destination_lon:.4f}) | "
            f"Distance: "
            f"{self.previous_distance:.2f} km | "
            f"Heading: "
            f"{self.current_heading:.1f}°"
        )


# ============================================================
# ENVIRONMENT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING IMPROVED ROUTE OPTIMIZATION ENVIRONMENT")
    print("=" * 70)

    env = RouteOptimizationEnv()

    print("\nEnvironment created.")

    print(
        "Observation space:",
        env.observation_space
    )

    print(
        "Action space:",
        env.action_space
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=42
    )

    print("\nInitial observation:")
    print(observation)

    print("\nInitial route information:")

    for key, value in info.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nHistorical initial heading:",
        f"{info['initial_heading']:.2f}°"
    )

    print(
        "Initial distance:",
        f"{info['initial_distance_km']:.2f} km"
    )

    print(
        "Historical route distance:",
        f"{info['historical_route_distance_km']:.2f} km"
    )

    # --------------------------------------------------------
    # Test actions
    # --------------------------------------------------------

    print("\nRunning 10 test actions...")

    for step in range(10):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        print(
            f"Step {step + 1}: "
            f"Action={action}, "
            f"Reward={reward:.3f}, "
            f"Distance="
            f"{info['distance_to_destination_km']:.2f} km, "
            f"Heading="
            f"{info['heading']:.1f}°"
        )

        if terminated or truncated:
            break

    print("\nEnvironment test completed successfully.")