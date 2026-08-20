# ============================================================
# AI MARINE MONITORING SYSTEM
# FRESH COMPLETE FASTAPI BACKEND
# ============================================================

from pathlib import Path
import os
import json
import math
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import websockets

from dotenv import load_dotenv

import geopandas as gpd
from shapely.geometry import Point, LineString

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from stable_baselines3 import PPO


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent

ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# ENVIRONMENT
# ============================================================

AISSTREAM_API_KEY = os.getenv(
    "AISSTREAM_API_KEY"
)

SHIPFINDER_API_KEY = os.getenv(
    "SHIPFINDER_API_KEY"
)

WEATHER_API_KEY = os.getenv(
    "WEATHER_API_KEY"
)

SHIPFINDER_SEED_MMSI = 477232800


# ============================================================
# MODEL PATHS
# ============================================================

RF_PATH = (
    BASE_DIR /
    "random_forest_vessel_risk_v2.pkl"
)

XGB_PATH = (
    BASE_DIR /
    "xgboost_model.pkl"
)

XGB_HIGH_WAVE_PATH = (
    BASE_DIR /
    "XGBoost" /
    "xgboost_high_wave_ndbc_model.pkl"
)

LSTM_MODEL_PATH = (
    BASE_DIR /
    "LSTM" /
    "lstm_model.keras"
)

LSTM_SCALER_PATH = (
    BASE_DIR /
    "LSTM" /
    "lstm_scaler.pkl"
)

PPO_PATH = (
    BASE_DIR /
    "RL" /
    "rl_model" /
    "ppo_route_optimizer.zip"
)


# ============================================================
# GEOGRAPHIC DATA
# ============================================================

LAND_FILE = (
    BASE_DIR /
    "GeoData" /
    "land" /
    "ne_10m_land.shp"
)

COASTLINE_FILE = (
    BASE_DIR /
    "GeoData" /
    "coastline" /
    "ne_10m_coastline.shp"
)

PORTS_FILE = (
    BASE_DIR /
    "GeoData" /
    "ports" /
    "ne_10m_ports.shp"
)


LAND_GDF = None
COASTLINE_GDF = None
PORTS_GDF = None

COASTLINE_METRIC_GDF = None
PORTS_METRIC_GDF = None

GEO_DATA_AVAILABLE = False


# ============================================================
# LOAD LAND
# ============================================================

def load_land_polygons():

    global LAND_GDF

    print()
    print("=" * 70)
    print("Loading land polygons...")
    print("=" * 70)

    print(
        "Land shapefile:",
        LAND_FILE
    )

    if not LAND_FILE.exists():

        print(
            "WARNING: Land shapefile not found."
        )

        LAND_GDF = None
        return

    try:

        LAND_GDF = gpd.read_file(
            LAND_FILE
        )

        if LAND_GDF.empty:

            LAND_GDF = None

            print(
                "WARNING: Land shapefile is empty."
            )

            return

        if LAND_GDF.crs is None:

            LAND_GDF = LAND_GDF.set_crs(
                "EPSG:4326"
            )

        else:

            LAND_GDF = LAND_GDF.to_crs(
                "EPSG:4326"
            )

        LAND_GDF = LAND_GDF[
            LAND_GDF.geometry.notnull()
        ]

        LAND_GDF = LAND_GDF[
            ~LAND_GDF.geometry.is_empty
        ]

        print(
            "Land polygons loaded:",
            len(LAND_GDF)
        )

    except Exception as e:

        LAND_GDF = None

        print(
            "Land loading error:",
            e
        )


load_land_polygons()


# ============================================================
# LOAD COASTLINE AND PORTS
# ============================================================

def load_geographic_data():

    global COASTLINE_GDF
    global PORTS_GDF
    global COASTLINE_METRIC_GDF
    global PORTS_METRIC_GDF
    global GEO_DATA_AVAILABLE

    print()
    print("=" * 70)
    print("Loading geographic data...")
    print("=" * 70)

    try:

        if COASTLINE_FILE.exists():

            COASTLINE_GDF = gpd.read_file(
                COASTLINE_FILE
            )

            if COASTLINE_GDF.crs is None:

                COASTLINE_GDF = (
                    COASTLINE_GDF.set_crs(
                        "EPSG:4326"
                    )
                )

            else:

                COASTLINE_GDF = (
                    COASTLINE_GDF.to_crs(
                        "EPSG:4326"
                    )
                )

            COASTLINE_METRIC_GDF = (
                COASTLINE_GDF.to_crs(
                    "EPSG:3857"
                )
            )

        if PORTS_FILE.exists():

            PORTS_GDF = gpd.read_file(
                PORTS_FILE
            )

            if PORTS_GDF.crs is None:

                PORTS_GDF = (
                    PORTS_GDF.set_crs(
                        "EPSG:4326"
                    )
                )

            else:

                PORTS_GDF = (
                    PORTS_GDF.to_crs(
                        "EPSG:4326"
                    )
                )

            PORTS_METRIC_GDF = (
                PORTS_GDF.to_crs(
                    "EPSG:3857"
                )
            )

        GEO_DATA_AVAILABLE = True

        print(
            "Geographic data loaded."
        )

    except Exception as e:

        GEO_DATA_AVAILABLE = False

        print(
            "Geographic data error:",
            e
        )


load_geographic_data()


# ============================================================
# AIS STATE
# ============================================================

AIS_VESSELS = {}

AIS_HISTORY = {}

AIS_CONNECTED = False

AIS_LAST_MESSAGE_TIME = None

AIS_LAST_MESSAGE_TYPE = None

AIS_LAST_ERROR = None

AIS_TASK = None


# ============================================================
# AIS BOUNDING BOX
# ============================================================
#
# This box covers the current demo / UK / North Sea area.
#
# Format:
# [
#   [south, west],
#   [north, east]
# ]
# ============================================================

AIS_BBOX = [
    [
        [45.0, -15.0],
        [65.0, 15.0]
    ]
]


# ============================================================
# AIS MESSAGE TYPES
# ============================================================

POSITION_MESSAGE_TYPES = {
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport"
}


# ============================================================
# LOAD ML MODELS
# ============================================================

print()
print("=" * 70)
print("Loading ML models...")
print("=" * 70)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

try:

    rf_model = joblib.load(
        RF_PATH
    )

    print(
        "Random Forest loaded."
    )

except Exception as e:

    rf_model = None

    print(
        "Random Forest load failed:",
        e
    )


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

try:

    xgb_model = joblib.load(
        XGB_PATH
    )

    print(
        "XGBoost loaded."
    )

except Exception as e:

    xgb_model = None

    print(
        "XGBoost load failed:",
        e
    )


# ------------------------------------------------------------
# HIGH WAVE XGBOOST
# ------------------------------------------------------------

try:

    xgb_high_wave_model = joblib.load(
        XGB_HIGH_WAVE_PATH
    )

    print(
        "High-wave XGBoost loaded."
    )

except Exception as e:

    xgb_high_wave_model = None

    print(
        "High-wave XGBoost load failed:",
        e
    )


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

try:

    lstm_model = load_model(
        LSTM_MODEL_PATH
    )

    print(
        "LSTM loaded."
    )

except Exception as e:

    lstm_model = None

    print(
        "LSTM load failed:",
        e
    )


# ------------------------------------------------------------
# LSTM SCALER
# ------------------------------------------------------------

try:

    lstm_scaler = joblib.load(
        LSTM_SCALER_PATH
    )

    print(
        "LSTM scaler loaded."
    )

except Exception as e:

    lstm_scaler = None

    print(
        "LSTM scaler load failed:",
        e
    )


# ------------------------------------------------------------
# PPO
# ------------------------------------------------------------

try:

    ppo_model = PPO.load(
        PPO_PATH
    )

    print(
        "PPO loaded."
    )

except Exception as e:

    ppo_model = None

    print(
        "PPO load failed:",
        e
    )


# ============================================================
# LSTM WAVE HISTORY
# ============================================================

LSTM_WAVE_HISTORY = []

LSTM_WAVE_HISTORY_MAX = 100


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1_rad = math.radians(
        float(lat1)
    )

    lat2_rad = math.radians(
        float(lat2)
    )

    dlat = math.radians(
        float(lat2) -
        float(lat1)
    )

    dlon = math.radians(
        float(lon2) -
        float(lon1)
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.sin(dlon / 2) ** 2
    )

    a = min(
        1.0,
        max(
            0.0,
            a
        )
    )

    return (
        R *
        2 *
        math.asin(
            math.sqrt(a)
        )
    )


def bearing_degrees(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1_rad = math.radians(
        float(lat1)
    )

    lat2_rad = math.radians(
        float(lat2)
    )

    dlon = math.radians(
        float(lon2) -
        float(lon1)
    )

    x = (
        math.sin(dlon)
        *
        math.cos(lat2_rad)
    )

    y = (
        math.cos(lat1_rad)
        *
        math.sin(lat2_rad)
        -
        math.sin(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.cos(dlon)
    )

    bearing = math.degrees(
        math.atan2(
            x,
            y
        )
    )

    return (
        bearing + 360
    ) % 360


def angle_difference(
    a,
    b
):

    return abs(
        (
            float(b)
            -
            float(a)
            +
            180
        )
        % 360
        -
        180
    )


# ============================================================
# LAND FUNCTIONS
# ============================================================

def point_is_land(
    latitude,
    longitude
):

    if (
        LAND_GDF is None
        or LAND_GDF.empty
    ):

        return False

    try:

        point = Point(
            float(longitude),
            float(latitude)
        )

        return bool(
            LAND_GDF.geometry
            .covers(point)
            .any()
        )

    except Exception:

        return False


def segment_crosses_land(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    if (
        LAND_GDF is None
        or LAND_GDF.empty
    ):

        return False

    try:

        line = LineString([
            (
                float(start_lon),
                float(start_lat)
            ),
            (
                float(end_lon),
                float(end_lat)
            )
        ])

        return bool(
            LAND_GDF.geometry
            .intersects(line)
            .any()
        )

    except Exception:

        return False


def route_crosses_land(
    points
):

    if len(points) < 2:

        return False

    for i in range(
        len(points) - 1
    ):

        if segment_crosses_land(
            points[i]["latitude"],
            points[i]["longitude"],
            points[i + 1]["latitude"],
            points[i + 1]["longitude"]
        ):

            return True

    return False


# ============================================================
# DISTANCE TO SHORE
# ============================================================

def distance_to_shore_km(
    latitude,
    longitude
):

    if (
        COASTLINE_METRIC_GDF is None
    ):

        return 0.0

    try:

        point = gpd.GeoSeries(
            [
                Point(
                    float(longitude),
                    float(latitude)
                )
            ],
            crs="EPSG:4326"
        ).to_crs(
            "EPSG:3857"
        ).iloc[0]

        distance_m = (
            COASTLINE_METRIC_GDF
            .geometry
            .distance(point)
            .min()
        )

        return float(
            distance_m / 1000
        )

    except Exception:

        return 0.0


# ============================================================
# DISTANCE TO PORT
# ============================================================

def distance_to_nearest_port_km(
    latitude,
    longitude
):

    if (
        PORTS_METRIC_GDF is None
    ):

        return 0.0

    try:

        point = gpd.GeoSeries(
            [
                Point(
                    float(longitude),
                    float(latitude)
                )
            ],
            crs="EPSG:4326"
        ).to_crs(
            "EPSG:3857"
        ).iloc[0]

        distance_m = (
            PORTS_METRIC_GDF
            .geometry
            .distance(point)
            .min()
        )

        return float(
            distance_m / 1000
        )

    except Exception:

        return 0.0


# ============================================================
# AIS MESSAGE PROCESSING
# ============================================================

def process_ais_message(
    data
):

    global AIS_LAST_MESSAGE_TIME
    global AIS_LAST_MESSAGE_TYPE

    message_type = data.get(
        "MessageType"
    )

    AIS_LAST_MESSAGE_TYPE = (
        message_type
    )

    metadata = data.get(
        "MetaData",
        {}
    )

    AIS_LAST_MESSAGE_TIME = (
        metadata.get(
            "time_utc"
        )
    )

    if (
        message_type
        not in POSITION_MESSAGE_TYPES
    ):

        return

    message_body = (
        data
        .get("Message", {})
        .get(
            message_type,
            {}
        )
    )

    mmsi = (
        message_body.get(
            "UserID"
        )
        or
        metadata.get(
            "MMSI"
        )
    )

    if mmsi is None:

        return

    latitude = message_body.get(
        "Latitude"
    )

    longitude = message_body.get(
        "Longitude"
    )

    if (
        latitude is None
        or longitude is None
    ):

        return

    try:

        mmsi = int(mmsi)

        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )

    except Exception:

        return

    speed = message_body.get(
        "Sog",
        0.0
    )

    course = message_body.get(
        "Cog",
        0.0
    )

    heading = message_body.get(
        "TrueHeading"
    )

    try:

        speed = float(
            speed
            if speed is not None
            else 0
        )

    except Exception:

        speed = 0.0

    try:

        course = float(
            course
            if course is not None
            else 0
        )

    except Exception:

        course = 0.0

    try:

        heading = float(
            heading
        )

        if heading == 511:

            heading = course

    except Exception:

        heading = course

    key = str(
        mmsi
    )

    timestamp = (
        metadata.get(
            "time_utc"
        )
        or
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    if key not in AIS_HISTORY:

        AIS_HISTORY[key] = []

    AIS_HISTORY[key].append({

        "latitude":
            latitude,

        "longitude":
            longitude,

        "speed":
            speed,

        "course":
            course,

        "heading":
            heading,

        "timestamp":
            timestamp

    })

    AIS_HISTORY[key] = (
        AIS_HISTORY[key][-100:]
    )

    # --------------------------------------------------------
    # LIVE VESSEL
    # --------------------------------------------------------

    AIS_VESSELS[key] = {

        "mmsi":
            mmsi,

        "ship_name":
            metadata.get(
                "ShipName"
            )
            or
            "UNKNOWN VESSEL",

        "latitude":
            latitude,

        "longitude":
            longitude,

        "speed":
            speed,

        "course":
            course,

        "heading":
            heading,

        "timestamp":
            timestamp,

        "message_type":
            message_type,

        "source":
            "AISStream",

        "imo":
            metadata.get(
                "IMO"
            ),

        "call_sign":
            metadata.get(
                "CallSign"
            ),

        "ship_type":
            metadata.get(
                "ShipType"
            )
    }

    print(
        f"AIS VESSEL | "
        f"MMSI={mmsi} | "
        f"NAME={AIS_VESSELS[key]['ship_name']} | "
        f"LAT={latitude:.5f} | "
        f"LON={longitude:.5f} | "
        f"SOG={speed:.2f} | "
        f"COG={course:.2f}"
    )


# ============================================================
# FALLBACK VESSELS
# ============================================================

def load_fallback_vessels():

    now = datetime.now(
        timezone.utc
    ).isoformat()

    return {

        "228361900": {

            "mmsi":
                228361900,

            "ship_name":
                "F/V MARIE CATHERINE2",

            "latitude":
                51.95997,

            "longitude":
                4.00994,

            "speed":
                7.5,

            "course":
                58.1,

            "heading":
                58.1,

            "ship_type":
                "Fishing",

            "message_type":
                "FALLBACK",

            "source":
                "Fallback",

            "timestamp":
                now
        },

        "244710359": {

            "mmsi":
                244710359,

            "ship_name":
                "AIS DEMO VESSEL 1",

            "latitude":
                52.37623,

            "longitude":
                4.96966,

            "speed":
                7.3,

            "course":
                133.0,

            "heading":
                133.0,

            "ship_type":
                "Cargo",

            "message_type":
                "FALLBACK",

            "source":
                "Fallback",

            "timestamp":
                now
        },

        "244110242": {

            "mmsi":
                244110242,

            "ship_name":
                "AIS DEMO VESSEL 2",

            "latitude":
                52.76186,

            "longitude":
                4.68774,

            "speed":
                5.0,

            "course":
                26.1,

            "heading":
                26.1,

            "ship_type":
                "Cargo",

            "message_type":
                "FALLBACK",

            "source":
                "Fallback",

            "timestamp":
                now
        },

        "244860069": {

            "mmsi":
                244860069,

            "ship_name":
                "AIS DEMO VESSEL 3",

            "latitude":
                52.36952,

            "longitude":
                4.90007,

            "speed":
                2.6,

            "course":
                310.5,

            "heading":
                310.5,

            "ship_type":
                "Tanker",

            "message_type":
                "FALLBACK",

            "source":
                "Fallback",

            "timestamp":
                now
        },

        "235090959": {

            "mmsi":
                235090959,

            "ship_name":
                "AIS DEMO VESSEL 4",

            "latitude":
                50.70678,

            "longitude":
                -1.98339,

            "speed":
                6.9,

            "course":
                133.7,

            "heading":
                133.7,

            "ship_type":
                "Cargo",

            "message_type":
                "FALLBACK",

            "source":
                "Fallback",

            "timestamp":
                now
        }
    }


# ============================================================
# AIS MOVEMENT FEATURES
# ============================================================

def get_ais_movement_features(
    mmsi
):

    history = AIS_HISTORY.get(
        str(mmsi),
        []
    )

    if not history:

        return {

            "total_distance_km":
                0.0,

            "average_speed_knots":
                0.0
        }

    total_distance = 0.0

    speeds = []

    for item in history:

        try:

            speeds.append(
                float(
                    item.get(
                        "speed",
                        0
                    )
                )
            )

        except Exception:

            pass

    for i in range(
        1,
        len(history)
    ):

        a = history[i - 1]

        b = history[i]

        try:

            total_distance += (
                haversine_km(
                    a["latitude"],
                    a["longitude"],
                    b["latitude"],
                    b["longitude"]
                )
            )

        except Exception:

            pass

    return {

        "total_distance_km":
            float(
                total_distance
            ),

        "average_speed_knots":
            float(
                np.mean(speeds)
            )
            if speeds
            else
            0.0
    }


# ============================================================
# RANDOM FOREST LIVE RISK
# ============================================================

def predict_live_vessel_risk(
    mmsi,
    latitude,
    longitude,
    speed
):

    if rf_model is None:

        raise RuntimeError(
            "Random Forest model is not loaded."
        )

    movement = (
        get_ais_movement_features(
            mmsi
        )
    )

    shore_distance = (
        distance_to_shore_km(
            latitude,
            longitude
        )
    )

    port_distance = (
        distance_to_nearest_port_km(
            latitude,
            longitude
        )
    )

    history = AIS_HISTORY.get(
        str(mmsi),
        []
    )

    if history:

        first = history[0]

        start_shore = (
            distance_to_shore_km(
                first["latitude"],
                first["longitude"]
            )
        )

        start_port = (
            distance_to_nearest_port_km(
                first["latitude"],
                first["longitude"]
            )
        )

    else:

        start_shore = (
            shore_distance
        )

        start_port = (
            port_distance
        )

    features = pd.DataFrame(

        [[

            float(latitude),

            float(longitude),

            float(
                movement[
                    "total_distance_km"
                ]
            ),

            float(
                movement[
                    "average_speed_knots"
                ]
            ),

            float(start_shore),

            float(shore_distance),

            float(start_port),

            float(port_distance)

        ]],

        columns=[

            "position.lat",

            "position.lon",

            "fishing.totalDistanceKm",

            "fishing.averageSpeedKnots",

            "distances.startDistanceFromShoreKm",

            "distances.endDistanceFromShoreKm",

            "distances.startDistanceFromPortKm",

            "distances.endDistanceFromPortKm"
        ]
    )

    prediction = int(
        rf_model.predict(
            features
        )[0]
    )

    probability = float(
        rf_model.predict_proba(
            features
        )[0][1]
    )

    if probability >= 0.70:

        risk_level = "HIGH"

    elif probability >= 0.30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {

        "prediction":
            prediction,

        "probability":
            probability,

        "probability_percent":
            round(
                probability * 100,
                2
            ),

        "risk_level":
            risk_level,

        "features": {

            "latitude":
                float(latitude),

            "longitude":
                float(longitude),

            "total_distance_km":
                round(
                    movement[
                        "total_distance_km"
                    ],
                    3
                ),

            "average_speed_knots":
                round(
                    movement[
                        "average_speed_knots"
                    ],
                    3
                ),

            "start_distance_from_shore_km":
                round(
                    start_shore,
                    3
                ),

            "end_distance_from_shore_km":
                round(
                    shore_distance,
                    3
                ),

            "start_distance_from_port_km":
                round(
                    start_port,
                    3
                ),

            "end_distance_from_port_km":
                round(
                    port_distance,
                    3
                )
        }
    }


# ============================================================
# XGBOOST LIVE RISK
# ============================================================

def predict_live_vessel_xgboost_risk(
    mmsi,
    latitude,
    longitude,
    speed
):

    if xgb_model is None:

        raise RuntimeError(
            "XGBoost model is not loaded."
        )

    movement = (
        get_ais_movement_features(
            mmsi
        )
    )

    shore_distance = (
        distance_to_shore_km(
            latitude,
            longitude
        )
    )

    port_distance = (
        distance_to_nearest_port_km(
            latitude,
            longitude
        )
    )

    history = AIS_HISTORY.get(
        str(mmsi),
        []
    )

    if history:

        first = history[0]

        start_shore = (
            distance_to_shore_km(
                first["latitude"],
                first["longitude"]
            )
        )

        start_port = (
            distance_to_nearest_port_km(
                first["latitude"],
                first["longitude"]
            )
        )

    else:

        start_shore = (
            shore_distance
        )

        start_port = (
            port_distance
        )

    features = pd.DataFrame(

        [[

            float(latitude),

            float(longitude),

            float(
                movement[
                    "total_distance_km"
                ]
            ),

            float(
                movement[
                    "average_speed_knots"
                ]
            ),

            np.nan,

            float(start_shore),

            float(shore_distance),

            float(start_port),

            float(port_distance),

            1

        ]],

        columns=[

            "position.lat",

            "position.lon",

            "fishing.totalDistanceKm",

            "fishing.averageSpeedKnots",

            "fishing.averageDurationHours",

            "distances.startDistanceFromShoreKm",

            "distances.endDistanceFromShoreKm",

            "distances.startDistanceFromPortKm",

            "distances.endDistanceFromPortKm",

            "vessel.type_fishing"
        ]
    )

    prediction = int(
        xgb_model.predict(
            features
        )[0]
    )

    probability = float(
        xgb_model.predict_proba(
            features
        )[0][1]
    )

    if probability >= 0.70:

        risk_level = "HIGH"

    elif probability >= 0.30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {

        "prediction":
            prediction,

        "probability":
            probability,

        "probability_percent":
            round(
                probability * 100,
                2
            ),

        "risk_level":
            risk_level
    }


# ============================================================
# AIS STREAM WORKER
# ============================================================

async def ais_stream_worker():

    global AIS_CONNECTED
    global AIS_LAST_ERROR
    global AIS_LAST_MESSAGE_TYPE
    global AIS_LAST_MESSAGE_TIME

    print()
    print("=" * 70)
    print("AIS BACKGROUND WORKER STARTED")
    print("=" * 70)

    while True:

        try:

            if not AISSTREAM_API_KEY:

                raise RuntimeError(
                    "AISSTREAM_API_KEY is missing."
                )

            uri = (
                "wss://stream.aisstream.io/v0/stream"
            )

            subscription = {

                "APIKey":
                    AISSTREAM_API_KEY,

                "BoundingBoxes":
                    AIS_BBOX,

                "FilterMessageTypes": [
                    "PositionReport",
                    "StandardClassBPositionReport",
                    "ExtendedClassBPositionReport"
                ]
            }

            print(
                "Connecting to AISStream..."
            )

            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            ) as websocket:

                await websocket.send(
                    json.dumps(
                        subscription
                    )
                )

                AIS_CONNECTED = True

                AIS_LAST_ERROR = None

                AIS_LAST_MESSAGE_TYPE = (
                    "AISSTREAM_CONNECTED"
                )

                AIS_LAST_MESSAGE_TIME = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                print(
                    "AISSTREAM CONNECTED"
                )

                while True:

                    raw_message = (
                        await websocket.recv()
                    )

                    if isinstance(
                        raw_message,
                        bytes
                    ):

                        raw_message = (
                            raw_message.decode(
                                "utf-8"
                            )
                        )

                    data = json.loads(
                        raw_message
                    )

                    process_ais_message(
                        data
                    )

        except asyncio.CancelledError:

            AIS_CONNECTED = False

            print(
                "AIS worker stopped."
            )

            raise

        except Exception as e:

            AIS_CONNECTED = False

            AIS_LAST_ERROR = (
                f"{type(e).__name__}: {e}"
            )

            print(
                "AISStream error:",
                AIS_LAST_ERROR
            )

            print(
                "Retrying AISStream in 10 seconds..."
            )

            await asyncio.sleep(
                10
            )


# ============================================================
# PYDANTIC MODELS
# ============================================================

class PredictionRequest(
    BaseModel
):

    features: list[float]


class LSTMRequest(
    BaseModel
):

    sequence: list[list[float]]


class NDBCObservation(
    BaseModel
):

    WVHT: float
    WSPD: float
    GST: float
    DPD: float
    APD: float
    PRES: float
    ATMP: float
    WTMP: float


class HighWaveRequest(
    BaseModel
):

    observations: list[
        NDBCObservation
    ]


class PPORequest(
    BaseModel
):

    observation: list[float]


class AISRouteRequest(
    BaseModel
):

    mmsi: int

    destination_lat: float

    destination_lon: float

    route_hazard: str | None = None

    ndbc_observations: (
        list[NDBCObservation] |
        None
    ) = None


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app
):

    global AIS_TASK

    print()
    print("=" * 70)
    print(
        "AI MARINE MONITORING SYSTEM"
    )
    print("=" * 70)

    AIS_TASK = asyncio.create_task(
        ais_stream_worker()
    )

    try:

        yield

    finally:

        if AIS_TASK is not None:

            AIS_TASK.cancel()

            try:

                await AIS_TASK

            except asyncio.CancelledError:

                pass

        print(
            "Backend shutdown complete."
        )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=
        "AI Marine Monitoring System",

    description=
        "AI based marine vessel monitoring, "
        "risk prediction, wave forecasting "
        "and PPO route optimization",

    version="2.0.0",

    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://127.0.0.1:5500",

        "http://localhost:5500",

        "*"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "system":
            "AI Marine Monitoring System",

        "status":
            "running",

        "backend":
            "FastAPI",

        "version":
            "2.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "models": {

            "random_forest":
                rf_model is not None,

            "xgboost":
                xgb_model is not None,

            "xgboost_high_wave":
                xgb_high_wave_model is not None,

            "lstm":
                lstm_model is not None,

            "lstm_scaler":
                lstm_scaler is not None,

            "ppo":
                ppo_model is not None
        },

        "ais": {

            "connected":
                AIS_CONNECTED,

            "vessel_count":
                len(AIS_VESSELS),

            "last_message_type":
                AIS_LAST_MESSAGE_TYPE,

            "last_message_time":
                AIS_LAST_MESSAGE_TIME,

            "last_error":
                AIS_LAST_ERROR
        }
    }


# ============================================================
# AIS RAW
# ============================================================

@app.get("/ais/raw")
async def get_ais_raw():

    return {

        "status":
            "success",

        "count":
            len(AIS_VESSELS),

        "vessels":
            list(
                AIS_VESSELS.values()
            ),

        "source":
            "AISStream"
    }


# ============================================================
# CREATE FALLBACK COPY
# ============================================================

def get_fallback_copy():

    return [
        dict(vessel)
        for vessel in
        load_fallback_vessels().values()
    ]


# ============================================================
# BUILD COMPLETE VESSEL RECORD
# ============================================================

def build_vessel_record(
    vessel
):

    data = dict(
        vessel
    )

    data.setdefault(
        "mmsi",
        None
    )

    data.setdefault(
        "ship_name",
        "UNKNOWN VESSEL"
    )

    data.setdefault(
        "latitude",
        None
    )

    data.setdefault(
        "longitude",
        None
    )

    data.setdefault(
        "speed",
        0.0
    )

    data.setdefault(
        "course",
        0.0
    )

    data.setdefault(
        "heading",
        data.get(
            "course",
            0.0
        )
    )

    data.setdefault(
        "ship_type",
        "UNKNOWN"
    )

    data.setdefault(
        "imo",
        None
    )

    data.setdefault(
        "call_sign",
        None
    )

    data.setdefault(
        "destination",
        None
    )

    data.setdefault(
        "source",
        "AIS"
    )

    data.setdefault(
        "message_type",
        "AIS"
    )

    return data


# ============================================================
# ALL AIS VESSELS
# ============================================================

@app.get("/ais/vessels")
async def get_ais_vessels():

    # --------------------------------------------------------
    # LIVE AIS
    # --------------------------------------------------------

    if AIS_VESSELS:

        vessels = [

            build_vessel_record(
                vessel
            )

            for vessel
            in AIS_VESSELS.values()
        ]

        return {

            "status":
                "success",

            "count":
                len(vessels),

            "vessels":
                vessels,

            "source":
                "live",

            "message":
                "All currently received AIS vessels."
        }


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    vessels = get_fallback_copy()

    for vessel in vessels:

        try:

            vessel["vessel_risk"] = (
                predict_live_vessel_risk(
                    vessel["mmsi"],
                    vessel["latitude"],
                    vessel["longitude"],
                    vessel.get(
                        "speed",
                        0.0
                    )
                )
            )

        except Exception:

            vessel["vessel_risk"] = {
                "risk_level":
                    "PENDING"
            }

    return {

        "status":
            "success",

        "count":
            len(vessels),

        "vessels":
            vessels,

        "source":
            "fallback",

        "message":
            "Live AIS unavailable. "
            "Fallback vessels are shown."
    }


# ============================================================
# FIND VESSEL
# ============================================================

def find_vessel_by_mmsi(
    mmsi
):

    target = int(
        mmsi
    )

    vessel = AIS_VESSELS.get(
        str(target)
    )

    if vessel is not None:

        return vessel

    fallback = (
        load_fallback_vessels()
    )

    return fallback.get(
        str(target)
    )


# ============================================================
# AIS RISK FOR ONE VESSEL
# ============================================================

@app.get(
    "/ais/risk/{mmsi}"
)
async def get_live_vessel_risk(
    mmsi: int
):

    vessel = (
        find_vessel_by_mmsi(
            mmsi
        )
    )

    if vessel is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Vessel MMSI {mmsi} "
                "not found."
            )
        )

    latitude = float(
        vessel["latitude"]
    )

    longitude = float(
        vessel["longitude"]
    )

    speed = float(
        vessel.get(
            "speed",
            0.0
        )
    )

    try:

        rf_result = (
            predict_live_vessel_risk(
                mmsi,
                latitude,
                longitude,
                speed
            )
        )

    except Exception as e:

        rf_result = {
            "error":
                str(e)
        }

    try:

        xgb_result = (
            predict_live_vessel_xgboost_risk(
                mmsi,
                latitude,
                longitude,
                speed
            )
        )

    except Exception as e:

        xgb_result = {
            "error":
                str(e)
        }

    return {

        "status":
            "success",

        "vessel":
            build_vessel_record(
                vessel
            ),

        "random_forest":
            rf_result,

        "xgboost":
            xgb_result
    }


# ============================================================
# ALL VESSEL RISKS AT ONCE
# ============================================================

@app.get(
    "/ais/risk/all"
)
async def get_all_vessel_risks():

    vessels = list(
        AIS_VESSELS.values()
    )

    source = "live"

    if not vessels:

        vessels = (
            get_fallback_copy()
        )

        source = "fallback"

    results = []

    for vessel in vessels:

        data = build_vessel_record(
            vessel
        )

        mmsi = data.get(
            "mmsi"
        )

        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )

        speed = data.get(
            "speed",
            0.0
        )

        try:

            rf_result = (
                predict_live_vessel_risk(
                    mmsi,
                    latitude,
                    longitude,
                    speed
                )
            )

        except Exception as e:

            rf_result = {
                "error":
                    str(e)
            }

        try:

            xgb_result = (
                predict_live_vessel_xgboost_risk(
                    mmsi,
                    latitude,
                    longitude,
                    speed
                )
            )

        except Exception as e:

            xgb_result = {
                "error":
                    str(e)
            }

        data["random_forest"] = (
            rf_result
        )

        data["xgboost"] = (
            xgb_result
        )

        results.append(
            data
        )

    return {

        "status":
            "success",

        "count":
            len(results),

        "source":
            source,

        "vessels":
            results
    }


# ============================================================
# XGBOOST BASIC
# ============================================================

@app.post(
    "/predict/xgboost"
)
def predict_xgboost(
    request: PredictionRequest
):

    if xgb_model is None:

        raise HTTPException(
            status_code=500,
            detail=
                "XGBoost model is not loaded."
        )

    if len(
        request.features
    ) != 10:

        raise HTTPException(
            status_code=400,
            detail=
                "XGBoost expects 10 features."
        )

    X = np.array(
        [request.features],
        dtype=np.float32
    )

    prediction = (
        xgb_model.predict(
            X
        )
    )

    return {

        "model":
            "XGBoost",

        "prediction":
            prediction.tolist()
    }


# ============================================================
# HIGH WAVE XGBOOST
# ============================================================

@app.post(
    "/predict/high-wave"
)
def predict_high_wave(
    request: HighWaveRequest
):

    if xgb_high_wave_model is None:

        raise HTTPException(
            status_code=500,
            detail=
                "High-wave XGBoost model "
                "is not loaded."
        )

    if len(
        request.observations
    ) != 3:

        raise HTTPException(
            status_code=400,
            detail=
                "Exactly 3 observations required."
        )

    feature_names = [

        "WVHT",
        "WSPD",
        "GST",
        "DPD",
        "APD",
        "PRES",
        "ATMP",
        "WTMP"
    ]

    features = []

    for observation in (
        request.observations
    ):

        values = (
            observation.model_dump()
        )

        for name in feature_names:

            features.append(
                float(
                    values[name]
                )
            )

    X = np.array(
        [features],
        dtype=np.float32
    )

    probability = float(
        xgb_high_wave_model
        .predict_proba(
            X
        )[0][1]
    )

    prediction = int(
        probability >= 0.50
    )

    return {

        "model":
            "XGBoost NDBC",

        "prediction":
            prediction,

        "hazard":
            (
                "HIGH_WAVE"
                if prediction
                else
                "NORMAL"
            ),

        "probability":
            probability,

        "probability_percent":
            round(
                probability * 100,
                2
            ),

        "observations_used":
            3,

        "features_used":
            24
    }


# ============================================================
# ADD WAVE OBSERVATION
# ============================================================

def add_wave_observation(
    vhm0,
    vtpk,
    vped,
    timestamp=None
):

    try:

        vhm0 = float(
            vhm0
        )

        vtpk = float(
            vtpk
        )

        vped = float(
            vped
        )

    except Exception:

        return False

    if not all(
        math.isfinite(value)
        for value in [
            vhm0,
            vtpk,
            vped
        ]
    ):

        return False

    observation = {

        "VHM0":
            vhm0,

        "VTPK":
            vtpk,

        "VPED":
            vped,

        "timestamp":
            timestamp
            or
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    LSTM_WAVE_HISTORY.append(
        observation
    )

    if len(
        LSTM_WAVE_HISTORY
    ) > LSTM_WAVE_HISTORY_MAX:

        del LSTM_WAVE_HISTORY[
            :-LSTM_WAVE_HISTORY_MAX
        ]

    return True


# ============================================================
# WAVE OBSERVATION API
# ============================================================

@app.post(
    "/wave/observation"
)
def add_wave_observation_api(
    observation: dict
):

    for field in [
        "VHM0",
        "VTPK",
        "VPED"
    ]:

        if field not in observation:

            raise HTTPException(
                status_code=400,
                detail=
                    f"Missing field: {field}"
            )

    success = (
        add_wave_observation(
            observation["VHM0"],
            observation["VTPK"],
            observation["VPED"],
            observation.get(
                "timestamp"
            )
        )
    )

    if not success:

        raise HTTPException(
            status_code=400,
            detail=
                "Invalid wave observation."
        )

    return {

        "status":
            "success",

        "count":
            len(
                LSTM_WAVE_HISTORY
            ),

        "observation":
            LSTM_WAVE_HISTORY[-1]
    }


# ============================================================
# WAVE HISTORY
# ============================================================

@app.get(
    "/wave/history"
)
def get_wave_history():

    return {

        "status":
            "success",

        "count":
            len(
                LSTM_WAVE_HISTORY
            ),

        "history":
            LSTM_WAVE_HISTORY
    }


@app.get(
    "/wave/history/lstm"
)
def get_lstm_wave_history():

    history = (
        LSTM_WAVE_HISTORY[-8:]
    )

    return {

        "status":
            "success",

        "count":
            len(history),

        "required":
            8,

        "history":
            history
    }


# ============================================================
# LSTM PREDICTION
# ============================================================

def run_lstm_prediction(
    history
):

    if lstm_model is None:

        raise HTTPException(
            status_code=500,
            detail=
                "LSTM model is not loaded."
        )

    if lstm_scaler is None:

        raise HTTPException(
            status_code=500,
            detail=
                "LSTM scaler is not loaded."
        )

    sequence = np.array(

        [

            [
                float(
                    item["VHM0"]
                ),

                float(
                    item["VTPK"]
                ),

                float(
                    item["VPED"]
                )
            ]

            for item in history

        ],

        dtype=np.float32
    )

    if sequence.shape != (
        8,
        3
    ):

        raise HTTPException(
            status_code=500,
            detail=
                "LSTM sequence must be 8 x 3."
        )

    scaled = (
        lstm_scaler.transform(
            sequence
        )
    )

    X = np.expand_dims(
        scaled,
        axis=0
    )

    prediction = (
        lstm_model.predict(
            X,
            verbose=0
        )
    )

    predicted_vhm0 = float(
        prediction.reshape(-1)[0]
    )

    if predicted_vhm0 >= 3.0:

        wave_status = (
            "HIGH_WAVE"
        )

    elif predicted_vhm0 >= 2.0:

        wave_status = (
            "MODERATE"
        )

    else:

        wave_status = (
            "LOW"
        )

    return {

        "status":
            "success",

        "model":
            "LSTM",

        "prediction":
            predicted_vhm0,

        "predicted_vhm0_m":
            round(
                predicted_vhm0,
                3
            ),

        "target":
            "VHM0",

        "unit":
            "m",

        "wave_status":
            wave_status,

        "features": [
            "VHM0",
            "VTPK",
            "VPED"
        ],

        "timesteps":
            8,

        "observations_used":
            8,

        "history":
            history
    }


# ============================================================
# MANUAL LSTM
# ============================================================

@app.post(
    "/predict/lstm"
)
def predict_lstm(
    request: LSTMRequest
):

    if len(
        request.sequence
    ) != 8:

        raise HTTPException(
            status_code=400,
            detail=
                "LSTM requires 8 observations."
        )

    history = [

        {

            "VHM0":
                row[0],

            "VTPK":
                row[1],

            "VPED":
                row[2]

        }

        for row in
        request.sequence

    ]

    return run_lstm_prediction(
        history
    )


# ============================================================
# LIVE LSTM
# ============================================================

@app.get(
    "/predict/lstm/live"
)
def predict_lstm_live():

    if len(
        LSTM_WAVE_HISTORY
    ) < 8:

        return {

            "status":
                "collecting",

            "model":
                "LSTM",

            "message":
                "Waiting for 8 real wave observations.",

            "observations_available":
                len(
                    LSTM_WAVE_HISTORY
                ),

            "observations_required":
                8
        }

    history = (
        LSTM_WAVE_HISTORY[-8:]
    )

    return run_lstm_prediction(
        history
    )


# ============================================================
# BASIC PPO
# ============================================================

@app.post(
    "/optimize-route"
)
def optimize_route(
    request: PPORequest
):

    if ppo_model is None:

        raise HTTPException(
            status_code=500,
            detail=
                "PPO model is not loaded."
        )

    if len(
        request.observation
    ) != 9:

        raise HTTPException(
            status_code=400,
            detail=
                "PPO expects 9 observations."
        )

    observation = np.array(
        request.observation,
        dtype=np.float32
    )

    action, _ = (
        ppo_model.predict(
            observation,
            deterministic=True
        )
    )

    return {

        "model":
            "PPO",

        "action":
            int(action)
    }


# ============================================================
# PPO ROUTE FROM AIS
# ============================================================

@app.post(
    "/optimize-route-from-ais"
)
def optimize_route_from_ais(
    request: AISRouteRequest
):



    # --------------------------------------------------------
    # CHECK PPO
    # --------------------------------------------------------

    if ppo_model is None:

        raise HTTPException(
            status_code=500,
            detail=
                "PPO model is not loaded."
        )

    # --------------------------------------------------------
    # FIND VESSEL
    # --------------------------------------------------------

    vessel = (
        find_vessel_by_mmsi(
            request.mmsi
        )
    )

    if vessel is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Vessel MMSI "
                f"{request.mmsi} "
                "not found."
            )
        )

    # --------------------------------------------------------
    # CURRENT POSITION
    # --------------------------------------------------------

    current_lat = float(
        vessel["latitude"]
    )

    current_lon = float(
        vessel["longitude"]
    )

    current_speed = float(
        vessel.get(
            "speed",
            0.0
        )
    )

    current_heading = float(
        vessel.get(
            "heading",
            vessel.get(
                "course",
                0.0
            )
        )
    )

    destination_lat = float(
        request.destination_lat
    )

    destination_lon = float(
        request.destination_lon
    )

    # --------------------------------------------------------
    # OCEAN VALIDATION
    # --------------------------------------------------------

    if point_is_land(
        current_lat,
        current_lon
    ):

        raise HTTPException(
            status_code=400,
            detail=
                "Current vessel position is on land."
        )

    if point_is_land(
        destination_lat,
        destination_lon
    ):

        raise HTTPException(
            status_code=400,
            detail=
                "Destination is on land."
        )

    # --------------------------------------------------------
    # DEFAULT ROUTE STATUS
    # --------------------------------------------------------

    route_mode = (
        "OPTIMIZED"
    )

    hazard = (
        "NO_WAVE_DATA"
    )

    wave_status = (
        "NO_WAVE_DATA"
    )

    wave_prediction = None

    xgb_probability = None

    # --------------------------------------------------------
    # MONITORING PAGE HAZARD DECISION
    # --------------------------------------------------------

    if request.route_hazard:

        monitoring_hazard = (
            str(
                request.route_hazard
            )
            .upper()
            .replace(
                " ",
                "_"
            )
        )

        if monitoring_hazard == "HIGH_WAVE":

            hazard = "HIGH_WAVE"

            wave_status = "HIGH_WAVE"

            route_mode = "SAFEST"

        elif monitoring_hazard in (
            "LOW",
            "NORMAL"
        ):

            hazard = "NORMAL"

            wave_status = "LOW"

            route_mode = "OPTIMIZED"




    # --------------------------------------------------------
    # NDBC HIGH-WAVE XGBOOST
    # --------------------------------------------------------

    if (
        request.ndbc_observations
        is not None
        and
        len(
            request.ndbc_observations
        ) == 3
        and
        xgb_high_wave_model
        is not None
    ):

        feature_names = [

            "WVHT",
            "WSPD",
            "GST",
            "DPD",
            "APD",
            "PRES",
            "ATMP",
            "WTMP"
        ]

        features = []

        for observation in (
            request.ndbc_observations
        ):

            values = (
                observation.model_dump()
            )

            for name in feature_names:

                features.append(
                    float(
                        values[name]
                    )
                )

        X = np.array(
            [features],
            dtype=np.float32
        )

        xgb_probability = float(

            xgb_high_wave_model
            .predict_proba(
                X
            )[0][1]

        )

        if xgb_probability >= 0.50:

            hazard = (
                "HIGH_WAVE"
            )

            wave_status = (
                "HIGH_WAVE"
            )

            route_mode = (
                "SAFEST"
            )

        else:

            hazard = (
                "NORMAL"
            )

            wave_status = (
                "LOW"
            )

            route_mode = (
                "OPTIMIZED"
            )

    # --------------------------------------------------------
    # OTHERWISE USE LIVE LSTM RESULT
    # --------------------------------------------------------

    elif len(
        LSTM_WAVE_HISTORY
    ) >= 8:

        try:

            lstm_result = (
                run_lstm_prediction(
                    LSTM_WAVE_HISTORY[-8:]
                )
            )

            wave_prediction = (
                lstm_result[
                    "predicted_vhm0_m"
                ]
            )

            wave_status = (
                lstm_result[
                    "wave_status"
                ]
            )

            if wave_prediction >= 3.0:

                hazard = (
                    "HIGH_WAVE"
                )

                route_mode = (
                    "SAFEST"
                )

            elif wave_prediction >= 2.0:

                hazard = (
                    "MODERATE"
                )

                route_mode = (
                    "OPTIMIZED"
                )

            else:

                hazard = (
                    "NORMAL"
                )

                route_mode = (
                    "OPTIMIZED"
                )

        except Exception as e:

            print(
                "LSTM route prediction error:",
                e
            )

    # --------------------------------------------------------
    # INITIAL ROUTE VALUES
    # --------------------------------------------------------

    initial_distance = (
        haversine_km(
            current_lat,
            current_lon,
            destination_lat,
            destination_lon
        )
    )

    desired_bearing = (
        bearing_degrees(
            current_lat,
            current_lon,
            destination_lat,
            destination_lon
        )
    )

    heading_error = (
        angle_difference(
            current_heading,
            desired_bearing
        )
    )

    # --------------------------------------------------------
    # PPO ACTIONS
    # --------------------------------------------------------

    turn_amounts = {

        0: -10.0,

        1: -5.0,

        2: 0.0,

        3: 5.0,

        4: 10.0
    }

    # --------------------------------------------------------
    # START ROUTE
    # --------------------------------------------------------

    route_points = [

        {

            "latitude":
                current_lat,

            "longitude":
                current_lon,

            "action":
                None,

            "heading":
                current_heading,

            "step":
                0
        }
    ]

    simulation_lat = (
        current_lat
    )

    simulation_lon = (
        current_lon
    )

    simulation_heading = (
        current_heading
    )

    # --------------------------------------------------------
    # ROUTE SIMULATION
    # --------------------------------------------------------

    max_steps = 120

    for step in range(
        max_steps
    ):

        distance = (
            haversine_km(
                simulation_lat,
                simulation_lon,
                destination_lat,
                destination_lon
            )
        )

        if distance <= 2.0:

            break

        desired_bearing = (
            bearing_degrees(
                simulation_lat,
                simulation_lon,
                destination_lat,
                destination_lon
            )
        )

        heading_error = (
            angle_difference(
                simulation_heading,
                desired_bearing
            )
        )

        observation = np.array(

            [

                simulation_lat / 90.0,

                simulation_lon / 180.0,

                destination_lat / 90.0,

                destination_lon / 180.0,

                np.clip(
                    current_speed / 40.0,
                    0.0,
                    1.0
                ),

                (
                    simulation_heading /
                    180.0
                ) - 1.0,

                np.clip(
                    distance / 500.0,
                    0.0,
                    1.0
                ),

                (
                    desired_bearing /
                    180.0
                ) - 1.0,

                heading_error /
                180.0
            ],

            dtype=np.float32
        )

        action, _ = (
            ppo_model.predict(
                observation,
                deterministic=True
            )
        )

        action = int(
            action
        )

        if action not in turn_amounts:

            action = 2

        ppo_turn = (
            turn_amounts[action]
        )

        # ----------------------------------------------------
        # SAFETY CANDIDATES
        # ----------------------------------------------------

        candidate_turns = [

            ppo_turn,

            ppo_turn + 5,

            ppo_turn - 5,

            ppo_turn + 10,

            ppo_turn - 10,

            ppo_turn + 20,

            ppo_turn - 20,

            ppo_turn + 45,

            ppo_turn - 45,

            ppo_turn + 90,

            ppo_turn - 90
        ]

        step_distance = min(

            5.0,

            max(
                0.5,
                distance / 20.0
            )
        )

        selected = None

        earth_radius = 6371.0

        # ----------------------------------------------------
        # TEST EACH TURN
        # ----------------------------------------------------

        for turn in (
            candidate_turns
        ):

            candidate_heading = (

                simulation_heading
                +
                turn

            ) % 360.0

            heading_rad = math.radians(
                candidate_heading
            )

            lat_rad = math.radians(
                simulation_lat
            )

            new_lat_rad = (

                lat_rad

                +

                (
                    step_distance /
                    earth_radius
                )
                *
                math.cos(
                    heading_rad
                )
            )

            cos_lat = max(

                math.cos(
                    lat_rad
                ),

                0.01
            )

            new_lon_rad = (

                math.radians(
                    simulation_lon
                )

                +

                (
                    step_distance /
                    (
                        earth_radius *
                        cos_lat
                    )
                )
                *
                math.sin(
                    heading_rad
                )
            )

            candidate_lat = math.degrees(
                new_lat_rad
            )

            candidate_lon = math.degrees(
                new_lon_rad
            )

            if candidate_lon > 180:

                candidate_lon -= 360

            if candidate_lon < -180:

                candidate_lon += 360

            if point_is_land(
                candidate_lat,
                candidate_lon
            ):

                continue

            if segment_crosses_land(

                simulation_lat,
                simulation_lon,

                candidate_lat,
                candidate_lon

            ):

                continue

            selected = (

                candidate_lat,

                candidate_lon,

                candidate_heading,

                turn
            )

            break

        # ----------------------------------------------------
        # NO SAFE STEP
        # ----------------------------------------------------

        if selected is None:

            raise HTTPException(

                status_code=400,

                detail=
                    "PPO could not find "
                    "an ocean-safe route step."
            )

        (

            simulation_lat,

            simulation_lon,

            simulation_heading,

            selected_turn

        ) = selected

        route_points.append({

            "latitude":
                simulation_lat,

            "longitude":
                simulation_lon,

            "action":
                action,

            "turn_angle":
                float(
                    selected_turn
                ),

            "ppo_turn_angle":
                float(
                    ppo_turn
                ),

            "heading":
                simulation_heading,

            "step":
                step + 1
        })

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    route_points.append({

        "latitude":
            destination_lat,

        "longitude":
            destination_lon,

        "action":
            None,

        "heading":
            None,

        "step":
            len(route_points)
    })

    # --------------------------------------------------------
    # FINAL LAND CHECK
    # --------------------------------------------------------

    if route_crosses_land(
        route_points
    ):

        raise HTTPException(

            status_code=400,

            detail=
                "Generated route intersects land."
        )

    # --------------------------------------------------------
    # ROUTE DISTANCE
    # --------------------------------------------------------

    route_distance = 0.0

    for i in range(
        len(route_points) - 1
    ):

        route_distance += (
            haversine_km(

                route_points[i][
                    "latitude"
                ],

                route_points[i][
                    "longitude"
                ],

                route_points[i + 1][
                    "latitude"
                ],

                route_points[i + 1][
                    "longitude"
                ]
            )
        )

    # --------------------------------------------------------
    # ESTIMATED TIME
    # --------------------------------------------------------

    speed_kmh = (
        max(
            current_speed,
            0.1
        )
        *
        1.852
    )

    estimated_time_hours = (
        route_distance /
        speed_kmh
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "status":
            "success",

        "model":
            "PPO Ocean Route",

        "route_mode":
            route_mode,

        "hazard":
            hazard,

        "wave_status":
            wave_status,

        "wave_prediction_m":
            wave_prediction,

        "xgb_wave_probability":
            xgb_probability,

        "mmsi":
            request.mmsi,

        "vessel":
            build_vessel_record(
                vessel
            ),

        "current_position": {

            "latitude":
                current_lat,

            "longitude":
                current_lon
        },

        "destination": {

            "latitude":
                destination_lat,

            "longitude":
                destination_lon
        },

        "initial_distance_km":
            float(
                initial_distance
            ),

        "route_distance_km":
            float(
                route_distance
            ),

        "estimated_time_hours":
            float(
                estimated_time_hours
            ),

        "speed_knots":
            float(
                current_speed
            ),

        "speed_kmh":
            float(
                speed_kmh
            ),

        "ocean_only":
            True,

        "land_check":
            True,

        "waypoints":
            route_points,

        "waypoint_count":
            len(
                route_points
            )
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host=
            "127.0.0.1",

        port=
            8000,

        reload=
            True
    )