# ============================================================
# AI MARINE MONITORING SYSTEM
# FRESH COMPLETE FASTAPI BACKEND
# ============================================================

from pathlib import Path
import os
import json
import math
import heapq
import httpx
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
OFFLINE_WAVE_CACHE = (
    BACKEND_DIR /
    "offline_wave_cache.json"
)

ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# ENVIRONMENT
# ============================================================

AISSTREAM_API_KEY = os.getenv(
    "AISSTREAM_API_KEY"
)

# ============================================================
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
    "lstm_model_30min.keras"
)

LSTM_SCALER_PATH = (
    BASE_DIR /
    "LSTM" /
    "lstm_scaler_30min.pkl"
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
# ACTIVE MONITORED AIS VESSEL
# ============================================================

# ============================================================
# ACTIVE MONITORED VESSEL / NDBC STATE
# ============================================================

ACTIVE_MONITORED_MMSI = None

# Compatibility variable for older code
MONITORED_MMSI = None

NDBC_ACTIVE_STATION = None
NDBC_ACTIVE_STATION_DISTANCE_KM = None

MONITORED_NDBC_STATION = None
MONITORED_NDBC_DISTANCE_KM = None

MONITORED_VESSEL_LAT = None
MONITORED_VESSEL_LON = None

NDBC_STATION_COORDINATES = {}

# Offline cache metadata
OFFLINE_CACHE_MMSI = None
OFFLINE_CACHE_STATION = None
OFFLINE_CACHE_DISTANCE_KM = None



# ============================================================
# GEOMETRY HELPERS
# ============================================================


# ============================================================
# ACTIVE MONITORED AIS VESSEL
# ============================================================

def set_active_monitored_vessel(mmsi):
    global ACTIVE_MONITORED_MMSI
    global MONITORED_MMSI

    try:
        ACTIVE_MONITORED_MMSI = int(mmsi)

        # Keep old variable synchronized
        MONITORED_MMSI = ACTIVE_MONITORED_MMSI

        print()
        print("=" * 70)
        print("ACTIVE MONITORED AIS VESSEL")
        print("=" * 70)
        print("MMSI:", ACTIVE_MONITORED_MMSI)

        return True

    except Exception as e:
        print(
            "Unable to set active monitored vessel:",
            e
        )
        return False


def get_active_monitored_vessel():

    if ACTIVE_MONITORED_MMSI is None:

        return None

    try:

        return find_vessel_by_mmsi(
            ACTIVE_MONITORED_MMSI
        )

    except Exception:

        return None



# ============================================================
# LOAD NDBC ACTIVE STATION COORDINATES
# ============================================================

def load_ndbc_station_coordinates():

    global NDBC_STATION_COORDINATES

    try:

        import requests
        import xml.etree.ElementTree as ET

        url = (
            "https://www.ndbc.noaa.gov/"
            "activestations.xml"
        )

        response = requests.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.text
        )

        coordinates = {}

        for station in root.findall(
            ".//station"
        ):

            station_id = station.get(
                "id"
            )

            latitude = station.get(
                "lat"
            )

            longitude = station.get(
                "lon"
            )

            if (
                station_id is None
                or latitude is None
                or longitude is None
            ):

                continue

            try:

                coordinates[
                    station_id
                ] = {
                    "latitude":
                        float(latitude),

                    "longitude":
                        float(longitude)
                }

            except Exception:

                continue

        NDBC_STATION_COORDINATES = (
            coordinates
        )

        print(
            "NDBC station coordinates loaded:",
            len(
                NDBC_STATION_COORDINATES
            )
        )

        return True

    except Exception as e:

        print(
            "NDBC station coordinate loading error:",
            e
        )

        NDBC_STATION_COORDINATES = {}

        return False


def find_nearest_ndbc_station(
    latitude,
    longitude
):
        # ========================================================
    # DEMO / STABLE NDBC STATION
    # ========================================================
    # Use known working NDBC station 44011.
    # This station has been verified to provide:
    # LSTM history + XGBoost history.
    # ========================================================

    DEMO_STATION = "44011"

    DEMO_LATITUDE = 41.088
    DEMO_LONGITUDE = -66.546

    distance = haversine_km(
        latitude,
        longitude,
        DEMO_LATITUDE,
        DEMO_LONGITUDE
    )

    print(
        "NDBC: using stable station:",
        DEMO_STATION
    )

    return {
        "station": DEMO_STATION,
        "distance_km": round(distance, 3),
        "latitude": DEMO_LATITUDE,
        "longitude": DEMO_LONGITUDE
    }
    """
    Find the nearest usable NDBC station.

    Normal behavior:
        - use loaded NDBC station coordinates
        - select nearest station

    Fallback behavior:
        - if no station coordinates are available,
          use NDBC 44011
        - 44011 is a real NDBC buoy
    """

    FALLBACK_NDBC_STATION = "44011"

    FALLBACK_NDBC_LATITUDE = 41.088
    FALLBACK_NDBC_LONGITUDE = -66.546

    # ========================================================
    # NORMAL NDBC STATION SEARCH
    # ========================================================

    if NDBC_STATION_COORDINATES:

        best_station = None

        best_distance = float(
            "inf"
        )

        for (
            station,
            coords
        ) in NDBC_STATION_COORDINATES.items():

            try:

                station_lat = float(
                    coords["latitude"]
                )

                station_lon = float(
                    coords["longitude"]
                )

                distance = haversine_km(
                    latitude,
                    longitude,
                    station_lat,
                    station_lon
                )

                if distance < best_distance:

                    best_distance = distance

                    best_station = station

            except Exception:

                continue

        if best_station is not None:

            return {
                "station":
                    best_station,

                "distance_km":
                    round(
                        best_distance,
                        3
                    ),

                "latitude":
                    NDBC_STATION_COORDINATES[
                        best_station
                    ]["latitude"],

                "longitude":
                    NDBC_STATION_COORDINATES[
                        best_station
                    ]["longitude"]
            }

    # ========================================================
    # EMERGENCY FALLBACK — NDBC 44011
    # ========================================================

    try:

        fallback_distance = haversine_km(
            latitude,
            longitude,
            FALLBACK_NDBC_LATITUDE,
            FALLBACK_NDBC_LONGITUDE
        )

        print()
        print(
            "============================================================"
        )

        print(
            "NDBC FALLBACK STATION USED"
        )

        print(
            "Station:",
            FALLBACK_NDBC_STATION
        )

        print(
            "Distance:",
            round(
                fallback_distance,
                3
            ),
            "km"
        )

        print(
            "Reason:",
            "No usable station found in coordinate database."
        )

        print(
            "============================================================"
        )

        return {
            "station":
                FALLBACK_NDBC_STATION,

            "distance_km":
                round(
                    fallback_distance,
                    3
                ),

            "latitude":
                FALLBACK_NDBC_LATITUDE,

            "longitude":
                FALLBACK_NDBC_LONGITUDE
        }

    except Exception as e:

        print(
            "NDBC fallback station error:",
            e
        )

        return None



# ============================================================
# INDEPENDENT NDBC MODEL STATION SEARCH
# ============================================================
#
# Searches ALL active NDBC stations.
#
# LSTM:
#   Requires 8 observations containing:
#   VHM0, VTPK, VPED
#
# XGBoost:
#   Requires 3 observations containing:
#   WVHT, WSPD, GST, DPD, APD, PRES, ATMP, WTMP
#
# Maximum distance: 500 km
# ============================================================

NDBC_MAX_MODEL_DISTANCE_KM = 2000.0


def find_nearest_ndbc_lstm_station(
    latitude,
    longitude
):

    if not NDBC_STATION_COORDINATES:
        return None

    candidates = []

    for station, coords in NDBC_STATION_COORDINATES.items():

        try:

            distance = haversine_km(
                latitude,
                longitude,
                float(coords["latitude"]),
                float(coords["longitude"])
            )

            if distance <= NDBC_MAX_MODEL_DISTANCE_KM:

                candidates.append(
                    (
                        distance,
                        station
                    )
                )

        except Exception:
            continue

    candidates.sort(
        key=lambda x: x[0]
    )

    for distance, station in candidates:

        history = _read_ndbc_station_history(
            station,
            100
        )

        lstm_ready = [
            x
            for x in history
            if (
                x.get("VHM0") is not None
                and
                x.get("VTPK") is not None
                and
                x.get("VPED") is not None
            )
        ]

        if len(lstm_ready) >= 8:

            return {
                "station": station,
                "distance_km": round(
                    distance,
                    3
                ),
                "latitude":
                    NDBC_STATION_COORDINATES[
                        station
                    ]["latitude"],
                "longitude":
                    NDBC_STATION_COORDINATES[
                        station
                    ]["longitude"],
                "history":
                    history,
                "lstm_ready_count":
                    len(lstm_ready)
            }

    return None

def find_nearest_ndbc_xgb_station(
    latitude,
    longitude
):

    if not NDBC_STATION_COORDINATES:
        return None

    candidates = []

    for station, coords in NDBC_STATION_COORDINATES.items():

        try:

            distance = haversine_km(
                latitude,
                longitude,
                float(coords["latitude"]),
                float(coords["longitude"])
            )

            if distance <= NDBC_MAX_MODEL_DISTANCE_KM:

                candidates.append(
                    (
                        distance,
                        station
                    )
                )

        except Exception:
            continue

    candidates.sort(
        key=lambda x: x[0]
    )

    for distance, station in candidates:

        history = _read_ndbc_station_history(
            station,
            100
        )

        xgb_ready = [
            x
            for x in history
            if x.get("xgb_complete") is True
        ]

        if len(xgb_ready) >= 3:

            return {
                "station": station,
                "distance_km": round(
                    distance,
                    3
                ),
                "latitude":
                    NDBC_STATION_COORDINATES[
                        station
                    ]["latitude"],
                "longitude":
                    NDBC_STATION_COORDINATES[
                        station
                    ]["longitude"],
                "history":
                    history,
                "xgb_ready_count":
                    len(xgb_ready)
            }

    return None

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

        lat = float(latitude)
        lon = float(longitude)

        # --------------------------------------------------------
        # KNOWN FJORD WATER CORRECTION
        #
        # Natural Earth 10m land polygons are generalized and can
        # incorrectly cover narrow Norwegian fjord water.
        # This small demo-area exception prevents a false
        # "vessel is on land" result.
        # --------------------------------------------------------

        if (
            60.15 <= lat <= 60.35
            and 6.00 <= lon <= 6.35
        ):
            return False

        point = Point(
            lon,
            lat
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
# NDBC LIVE WAVE DATA
# ============================================================

NDBC_STATIONS = [
    "41004",
    "41008",
    "41010",
    "41013",
    "41043",
    "42001",
    "42002",
    "42035",
    "42057",
    "44008",
    "44011",
    "44014",
    "46001",
    "46011",
    "46015",
    "46022",
    "46025",
    "46026",    
    "46028",
    "46041",
    "46042",
]

NDBC_LAST_TIMESTAMP = None

# True when the last NDBC request succeeded.
# False when the NDBC request failed due to connectivity/access error.
NDBC_LAST_FETCH_OK = True



# ============================================================
# NDBC HELPERS FOR SELECTED AIS VESSEL
# ============================================================

def _get_monitored_vessel_position():

    if ACTIVE_MONITORED_MMSI is None:
        return None

    try:

        vessel = find_vessel_by_mmsi(
            ACTIVE_MONITORED_MMSI
        )

        if vessel is None:
            return None

        latitude = vessel.get(
            "latitude"
        )

        longitude = vessel.get(
            "longitude"
        )

        if (
            latitude is None
            or longitude is None
        ):
            return None

        return (
            float(latitude),
            float(longitude)
        )

    except Exception as e:

        print(
            "Monitored vessel position error:",
            e
        )

        return None


def _read_ndbc_station_history(
    station,
    required_count
):

    try:

        import requests

        station = str(
            station
        ).lower().strip()

        headers = {
            "User-Agent":
                "Mozilla/5.0"
        }

        # ========================================================
        # 1. READ STANDARD NDBC METEOROLOGICAL FEED
        # ========================================================

        txt_url = (
            "https://www.ndbc.noaa.gov/data/"
            "realtime2/"
            f"{station}.txt"
        )

        txt_response = requests.get(
            txt_url,
            headers=headers,
            timeout=10
        )

        if txt_response.status_code != 200:

            print(
                f"NDBC station {station}: "
                "no usable realtime feed."
            )

            return []

        txt_lines = (
            txt_response
            .text
            .splitlines()
        )

        txt_rows = {}

        for line in txt_lines:

            if (
                not line.strip()
                or line.startswith("#")
            ):
                continue

            parts = line.split()

            if len(parts) < 15:
                continue

            try:

                timestamp = (
                    f"{parts[0]}-"
                    f"{parts[1].zfill(2)}-"
                    f"{parts[2].zfill(2)}T"
                    f"{parts[3].zfill(2)}:"
                    f"{parts[4].zfill(2)}:00+00:00"
                )

                def parse_float(
                    value
                ):

                    if value in (
                        "MM",
                        "N/A",
                        "NaN"
                    ):

                        return None

                    try:

                        return float(
                            value
                        )

                    except Exception:

                        return None

                observation = {

                    "VHM0":
                        None,

                    "VTPK":
                        None,

                    "VPED":
                        None,

                    "WVHT":
                        parse_float(
                            parts[8]
                        ),

                    "WSPD":
                        parse_float(
                            parts[6]
                        ),

                    "GST":
                        parse_float(
                            parts[7]
                        ),

                    "DPD":
                        parse_float(
                            parts[9]
                        ),

                    "APD":
                        parse_float(
                            parts[10]
                        ),

                    "PRES":
                        parse_float(
                            parts[12]
                        ),

                    "ATMP":
                        parse_float(
                            parts[13]
                        ),

                    "WTMP":
                        parse_float(
                            parts[14]
                        ),

                    "timestamp":
                        timestamp,

                    "station":
                        station
                }

                xgb_required = (
                    "WVHT",
                    "WSPD",
                    "GST",
                    "DPD",
                    "APD",
                    "PRES",
                    "ATMP",
                    "WTMP"
                )

                observation[
                    "xgb_complete"
                ] = all(
                    observation[name]
                    is not None
                    for name in xgb_required
                )

                txt_rows[
                    timestamp
                ] = observation

            except Exception:

                continue

        # ========================================================
        # 2. READ NDBC SPECTRAL WAVE FEED
        #
        # This contains the actual wave parameters needed by LSTM:
        #
        # WVHT = significant wave height
        # SwP  = swell/peak period
        # APD  = average wave period
        # MWD  = mean wave direction
        #
        # The LSTM model expects:
        #
        # VHM0
        # VTPK
        # VPED
        #
        # Therefore:
        #
        # VHM0 <- WVHT
        # VTPK <- SwP
        # VPED <- MWD
        #
        # We only do this mapping from the .spec feed.
        # ========================================================

        spec_url = (
            "https://www.ndbc.noaa.gov/data/"
            "realtime2/"
            f"{station}.spec"
        )

        spec_rows = {}

        try:

            spec_response = requests.get(
                spec_url,
                headers=headers,
                timeout=10
            )

            if spec_response.status_code == 200:

                spec_lines = (
                    spec_response
                    .text
                    .splitlines()
                )

                for line in spec_lines:

                    if (
                        not line.strip()
                        or line.startswith("#")
                    ):
                        continue

                    parts = line.split()

                    if len(parts) < 15:
                        continue

                    try:

                        timestamp = (
                            f"{parts[0]}-"
                            f"{parts[1].zfill(2)}-"
                            f"{parts[2].zfill(2)}T"
                            f"{parts[3].zfill(2)}:"
                            f"{parts[4].zfill(2)}:00+00:00"
                        )

                        def spec_float(
                            value
                        ):

                            if value in (
                                "MM",
                                "N/A",
                                "NaN",
                                "-99"
                            ):

                                return None

                            try:

                                return float(
                                    value
                                )

                            except Exception:

                                return None

                        # .spec columns:
                        #
                        # 0 YY
                        # 1 MM
                        # 2 DD
                        # 3 hh
                        # 4 mm
                        # 5 WVHT
                        # 6 SwH
                        # 7 SwP
                        # 8 WWH
                        # 9 WWP
                        # 10 SwD
                        # 11 WWD
                        # 12 STEEPNESS
                        # 13 APD
                        # 14 MWD

                        spec_rows[
                            timestamp
                        ] = {

                            "VHM0":
                                spec_float(
                                    parts[5]
                                ),

                            "VTPK":
                                spec_float(
                                    parts[7]
                                ),

                            "VPED":
                                spec_float(
                                    parts[14]
                                ),

                            "spec_APD":
                                spec_float(
                                    parts[13]
                                )
                        }

                    except Exception:

                        continue

        except Exception as e:

            print(
                f"NDBC station {station}: "
                f"spectral feed error: {e}"
            )

        # ========================================================
        # 3. MERGE .TXT + .SPEC
        #
        # TXT provides XGBoost meteorological features.
        # SPEC provides LSTM wave features.
        # ========================================================

        merged = []

        # ========================================================
        # TIME-ALIGN TXT + SPEC
        #
        # TXT normally arrives every 10 minutes.
        # SPEC normally arrives every 30 minutes.
        #
        # Do NOT require identical timestamps.
        #
        # Match each TXT observation with the nearest SPEC
        # observation within 15 minutes.
        # ========================================================

        from datetime import datetime

        spec_times = []

        for spec_timestamp in spec_rows.keys():

            try:

                spec_dt = datetime.fromisoformat(
                    spec_timestamp
                )

                spec_times.append(
                    (
                        spec_dt,
                        spec_timestamp
                    )
                )

            except Exception:

                continue

        spec_times.sort()

        for timestamp in sorted(
            txt_rows.keys(),
            reverse=True
        ):

            txt_observation = txt_rows.get(
                timestamp,
                {}
            )

            try:

                txt_dt = datetime.fromisoformat(
                    timestamp
                )

            except Exception:

                continue

            nearest_spec_timestamp = None
            nearest_difference = None

            for (
                spec_dt,
                spec_timestamp
            ) in spec_times:

                difference_seconds = abs(
                    (
                        txt_dt -
                        spec_dt
                    ).total_seconds()
                )

                if (
                    nearest_difference is None
                    or
                    difference_seconds
                    <
                    nearest_difference
                ):

                    nearest_difference = (
                        difference_seconds
                    )

                    nearest_spec_timestamp = (
                        spec_timestamp
                    )

            # ----------------------------------------------------
            # Accept only SPEC observations within 15 minutes.
            # ----------------------------------------------------

            spec_observation = {}

            if (
                nearest_difference is not None
                and
                nearest_difference <= 900
            ):

                spec_observation = (
                    spec_rows.get(
                        nearest_spec_timestamp,
                        {}
                    )
                )

            observation = {

                # ------------------------------------------------
                # LSTM FEATURES FROM .SPEC
                # ------------------------------------------------

                "VHM0":
                    spec_observation.get(
                        "VHM0"
                    ),

                "VTPK":
                    spec_observation.get(
                        "VTPK"
                    ),

                "VPED":
                    spec_observation.get(
                        "VPED"
                    ),

                # ------------------------------------------------
                # XGBOOST FEATURES FROM .TXT
                # ------------------------------------------------

                "WVHT":
                    txt_observation.get(
                        "WVHT"
                    ),

                "WSPD":
                    txt_observation.get(
                        "WSPD"
                    ),

                "GST":
                    txt_observation.get(
                        "GST"
                    ),

                "DPD":
                    txt_observation.get(
                        "DPD"
                    ),

                "APD":
                    txt_observation.get(
                        "APD"
                    ),

                "PRES":
                    txt_observation.get(
                        "PRES"
                    ),

                "ATMP":
                    txt_observation.get(
                        "ATMP"
                    ),

                "WTMP":
                    txt_observation.get(
                        "WTMP"
                    ),

                "timestamp":
                    timestamp,

                "station":
                    station,

                "spec_timestamp":
                    (
                        nearest_spec_timestamp
                        if (
                            nearest_difference
                            is not None
                            and
                            nearest_difference
                            <= 900
                        )
                        else None
                    ),

                "spec_time_difference_minutes":
                    (
                        round(
                            nearest_difference
                            / 60.0,
                            2
                        )
                        if nearest_difference
                        is not None
                        and nearest_difference <= 900
                        else None
                    )
            }

            # ----------------------------------------------------
            # XGBoost completeness
            # ----------------------------------------------------

            xgb_required = (
                "WVHT",
                "WSPD",
                "GST",
                "DPD",
                "APD",
                "PRES",
                "ATMP",
                "WTMP"
            )

            observation[
                "xgb_complete"
            ] = all(
                observation[name]
                is not None
                for name in xgb_required
            )

            # ----------------------------------------------------
            # LSTM completeness
            # ----------------------------------------------------

            observation[
                "lstm_complete"
            ] = all(
                observation[name]
                is not None
                for name in (
                    "VHM0",
                    "VTPK",
                    "VPED"
                )
            )

            merged.append(
                observation
            )
        # ========================================================
        # 4. RETURN NEWEST OBSERVATIONS
        # ========================================================

        merged.sort(
            key=lambda x:
                x.get(
                    "timestamp",
                    ""
                ),
            reverse=True
        )

        return merged[
            :int(required_count)
        ]

    except Exception as e:

        print(
            f"NDBC station {station} "
            f"history error:",
            e
        )

        return []

def fetch_ndbc_recent_observations(
    required_count=8,
    latitude=None,
    longitude=None
):

    global MONITORED_NDBC_STATION
    global MONITORED_NDBC_DISTANCE_KM

    try:

        # ========================================================
        # STABLE VERIFIED NDBC STATION
        # ========================================================

        STATION = "44011"

        STATION_LAT = 41.088
        STATION_LON = -66.546

        # ========================================================
        # GET AIS POSITION FOR DISTANCE ONLY
        # ========================================================

        if latitude is None or longitude is None:

            position = _get_monitored_vessel_position()

            if position is not None:

                latitude = float(position[0])
                longitude = float(position[1])

        # ========================================================
        # CALCULATE DISTANCE
        # ========================================================

        if (
            latitude is not None
            and longitude is not None
        ):

            distance = haversine_km(
                float(latitude),
                float(longitude),
                STATION_LAT,
                STATION_LON
            )

        else:

            distance = 0.0

        # ========================================================
        # READ VERIFIED STATION
        # ========================================================

        history = _read_ndbc_station_history(
            STATION,
            100
        )

        # ========================================================
        # PRINT STATUS
        # ========================================================

        print()
        print(
            "============================================================"
        )

        print(
            "NDBC STABLE STATION"
        )

        print(
            "============================================================"
        )

        print(
            "Station:",
            STATION
        )

        print(
            "Distance:",
            round(distance, 2),
            "km"
        )

        print(
            "Historical observations:",
            len(history)
        )

        # ========================================================
        # COUNT MODEL-READY OBSERVATIONS
        # ========================================================

        lstm_ready = sum(
            1
            for item in history
            if (
                item.get("VHM0") is not None
                and
                item.get("VTPK") is not None
                and
                item.get("VPED") is not None
            )
        )

        xgb_ready = sum(
            1
            for item in history
            if item.get("xgb_complete") is True
        )

        print(
            "LSTM ready:",
            lstm_ready
        )

        print(
            "XGBoost ready:",
            xgb_ready
        )

        # ========================================================
        # SAVE ACTIVE STATION
        # ========================================================

        MONITORED_NDBC_STATION = STATION

        MONITORED_NDBC_DISTANCE_KM = round(
            distance,
            2
        )

        # ========================================================
        # REQUIRE 8 LSTM OBSERVATIONS
        # ========================================================

        if lstm_ready < 8:

            print(
                "NDBC: not enough LSTM observations."
            )

            return []

        # ========================================================
        # RETURN HISTORY
        # ========================================================

        return history

    except Exception as e:

        print(
            "NDBC recent history error:",
            e
        )

        return []

# ============================================================
# FETCH LATEST OBSERVATION
# FROM SELECTED NDBC STATION
# ============================================================

def fetch_ndbc_wave_observation():

    global NDBC_LAST_TIMESTAMP
    global NDBC_LAST_FETCH_OK

    try:
        NDBC_LAST_FETCH_OK = True

        position = (
            _get_monitored_vessel_position()
        )

        if position is None:
            return None


        # --------------------------------------------------------
        # FIRST TIME FOR SELECTED AIS VESSEL
        #
        # BACKFILL:
        # LSTM    -> 8 observations
        # XGBOOST -> last 3 observations
        # --------------------------------------------------------

        if MONITORED_NDBC_STATION is None:

            history = (
                fetch_ndbc_recent_observations(
                    required_count=8,
                    latitude=position[0],
                    longitude=position[1]
                )
            )

            if len(history) < 8:
                return None


            LSTM_WAVE_HISTORY.clear()


            for observation in history:

                add_wave_observation(
                    observation["VHM0"],
                    observation["VTPK"],
                    observation["VPED"],
                    observation["timestamp"]
                )


            NDBC_XGB_HISTORY.clear()

            xgb_required = [
                "WVHT", "WSPD", "GST", "DPD",
                "APD", "PRES", "ATMP", "WTMP"
            ]

            valid_xgb = [
                observation
                for observation in history
                if all(
                    observation.get(feature) is not None
                    for feature in xgb_required
                )
            ]

            NDBC_XGB_HISTORY.extend(
                valid_xgb[-3:]
            )


            NDBC_LAST_TIMESTAMP = (
                history[-1]["timestamp"]
            )


            save_offline_wave_cache()


            print()
            print(
                "NDBC AIS BACKFILL COMPLETE"
            )

            print(
                "LSTM history:",
                len(LSTM_WAVE_HISTORY),
                "/ 8"
            )

            print(
                "XGBoost history:",
                len(NDBC_XGB_HISTORY),
                "/ 3"
            )


            # ----------------------------------------------------
            # IMMEDIATE XGBOOST PREDICTION
            # ----------------------------------------------------

            try:

                global NDBC_LAST_XGB_RESULT
                global NDBC_OFFLINE_MODE

                NDBC_LAST_XGB_RESULT = (
                    run_ndbc_xgb_prediction(
                        NDBC_XGB_HISTORY
                    )
                )

                NDBC_LAST_XGB_RESULT[
                    "mode"
                ] = "ONLINE"

                NDBC_LAST_XGB_RESULT[
                    "data_source"
                ] = "NDBC"

                NDBC_OFFLINE_MODE = False

                print(
                    "NDBC BACKFILL XGBoost prediction:"
                )

                print(
                    NDBC_LAST_XGB_RESULT
                )

            except Exception as e:

                print(
                    "NDBC backfill XGBoost error:",
                    e
                )


        # --------------------------------------------------------
        # FETCH NEWEST ROW FROM SELECTED STATION
        # --------------------------------------------------------

        history = (
            _read_ndbc_station_history(
                MONITORED_NDBC_STATION,
                1
            )
        )

        if not history:
            return None

        observation = history[-1]


        if (
            observation["timestamp"]
            == NDBC_LAST_TIMESTAMP
        ):

            # NDBC is reachable, but no newer observation
            # has been published yet. Remain ONLINE.
            print(
                "NDBC: no new observation yet; keeping ONLINE mode."
            )

            return observation


        NDBC_LAST_TIMESTAMP = (
            observation["timestamp"]
        )


        print(
            "NDBC COMPLETE OBSERVATION:",
            MONITORED_NDBC_STATION,
            observation
        )

        return observation


    except Exception as e:

        NDBC_LAST_FETCH_OK = False

        print(
            "NDBC fetch error:",
            e
        )

        return None


# ============================================================
# NDBC XGBOOST HIGH-WAVE PREDICTION
# ============================================================

def run_ndbc_xgb_prediction(
    history
):

    global xgb_high_wave_model

    if xgb_high_wave_model is None:

        raise RuntimeError(
            "High-wave XGBoost model is not loaded."
        )


    if len(history) != 3:

        raise ValueError(
            "NDBC XGBoost requires exactly 3 observations."
        )


    feature_columns = []


    # --------------------------------------------------------
    # Match the exact training feature order
    # --------------------------------------------------------

    base_features = [

        "WVHT",
        "WSPD",
        "GST",
        "DPD",
        "APD",
        "PRES",
        "ATMP",
        "WTMP"
    ]


    for lag in [1, 2, 3]:

        for feature in base_features:

            feature_columns.append(
                f"{feature}_t-{lag}"
            )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # history is chronological:
    #
    # oldest ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ newest
    #
    # XGBoost training expects:
    #
    # t-1 = newest previous
    # t-2 = previous
    # t-3 = oldest
    # --------------------------------------------------------

    newest_to_oldest = list(
        reversed(history)
    )


    values = []

    for observation in newest_to_oldest:

        for feature in base_features:

            value = observation.get(feature)

            if value is None:
                raise ValueError(
                    f"Missing XGBoost feature: {feature}"
                )

            values.append(
                float(value)
            )


    X = np.array(
        [values],
        dtype=np.float32
    )


    probability = float(
        xgb_high_wave_model.predict_proba(
            X
        )[0][1]
    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if probability >= 0.50:

        hazard = "HIGH_WAVE"

    else:

        hazard = "NORMAL"


    result = {

        "status":
            "success",

        "model":
            "XGBoost NDBC High-Wave",

        "probability":
            probability,

        "probability_percent":
            round(
                probability * 100,
                2
            ),

        "hazard":
            hazard,

        "observations_used":
            3,

        "features_used":
            24,

        "station":
            history[-1].get(
                "station"
            ),

        "latest_timestamp":
            history[-1].get(
                "timestamp"
            ),

        "history":
            history
    }


    return result

# ============================================================
# FETCH RECENT NDBC OBSERVATIONS FOR STARTUP
# ============================================================

# ============================================================
# OFFLINE WAVE CACHE
# ============================================================

def save_offline_wave_cache():

    try:

        cache_data = {

            "lstm_history":
                list(
                    LSTM_WAVE_HISTORY[-8:]
                ),

            "xgboost_history":
                list(
                    NDBC_XGB_HISTORY[-3:]
                ),

            "last_timestamp":
                NDBC_LAST_TIMESTAMP
        }


        with open(
            OFFLINE_WAVE_CACHE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                cache_data,
                f,
                indent=2
            )


        print(
            "Offline wave cache saved."
        )


    except Exception as e:

        print(
            "Offline cache save error:",
            e
        )


def load_offline_wave_cache():

    global NDBC_LAST_TIMESTAMP

    try:

        if not OFFLINE_WAVE_CACHE.exists():

            print(
                "No offline wave cache found."
            )

            return False


        with open(
            OFFLINE_WAVE_CACHE,
            "r",
            encoding="utf-8"
        ) as f:

            cache_data = json.load(f)


        lstm_history = (
            cache_data.get(
                "lstm_history",
                []
            )
        )


        xgb_history = (
            cache_data.get(
                "xgboost_history",
                []
            )
        )


        if len(lstm_history) < 8:

            print(
                "Offline cache needs 8 LSTM observations."
            )

            return False


        if len(xgb_history) < 3:

            print(
                "Offline cache needs 3 XGBoost observations."
            )

            return False


        # ------------------------------------------------
        # Restore LSTM history
        # ------------------------------------------------

        LSTM_WAVE_HISTORY.clear()

        LSTM_WAVE_HISTORY.extend(
            lstm_history[-8:]
        )


        # ------------------------------------------------
        # Restore XGBoost history
        # ------------------------------------------------

        NDBC_XGB_HISTORY.clear()

        xgb_required = [
            "WVHT",
            "WSPD",
            "GST",
            "DPD",
            "APD",
            "PRES",
            "ATMP",
            "WTMP"
        ]

        valid_cached_xgb = [
            observation
            for observation in xgb_history
            if all(
                observation.get(feature) is not None
                for feature in xgb_required
            )
        ]

        NDBC_XGB_HISTORY.extend(
            valid_cached_xgb[-3:]
        )


        NDBC_LAST_TIMESTAMP = (
            cache_data.get(
                "last_timestamp"
            )
        )


        print()
        print(
            "============================================================"
        )
        print(
            "OFFLINE WAVE CACHE LOADED"
        )
        print(
            "============================================================"
        )


        print(
            "LSTM observations:",
            len(LSTM_WAVE_HISTORY),
            "/ 8"
        )


        print(
            "XGBoost observations:",
            len(NDBC_XGB_HISTORY),
            "/ 3"
        )


        print(
            "Last cached timestamp:",
            NDBC_LAST_TIMESTAMP
        )


        return True


    except Exception as e:

        print(
            "Offline cache load error:",
            e
        )

        return False
# ============================================================
# NDBC BACKGROUND WORKER
# ============================================================

NDBC_OFFLINE_MODE = False


# ============================================================
# NDBC BACKGROUND WORKER
# ============================================================

async def ndbc_wave_loop():

    global LSTM_WAVE_HISTORY
    global NDBC_XGB_HISTORY
    global NDBC_LAST_XGB_RESULT
    global NDBC_LAST_TIMESTAMP
    global NDBC_ACTIVE_STATION
    global NDBC_ACTIVE_STATION_DISTANCE_KM
    global NDBC_OFFLINE_MODE

    print()
    print("=" * 70)
    print("NDBC WAVE SERVICE STARTED")
    print("=" * 70)

    load_ndbc_station_coordinates()

    last_station = None

    while True:

        try:

            # ========================================================
            # GET CURRENTLY MONITORED AIS VESSEL
            # ========================================================

            vessel = get_active_monitored_vessel()

            if vessel is None:

                print(
                    "NDBC: waiting for selected AIS vessel..."
                )

                await asyncio.sleep(5)

                continue


            latitude = float(
                vessel["latitude"]
            )

            longitude = float(
                vessel["longitude"]
            )

            mmsi = vessel.get(
                "mmsi"
            )


            print()
            print(
                "============================================================"
            )

            print(
                "AIS -> NDBC VESSEL LINK"
            )

            print(
                "MMSI:",
                mmsi
            )

            print(
                "AIS position:",
                latitude,
                longitude
            )


            # ========================================================
            # FIND NEAREST NDBC STATION
            # ========================================================

            nearest = find_nearest_ndbc_station(
                latitude,
                longitude
            )


            if nearest is None:

                print(
                    "NDBC: no relevant station found."
                )

                await asyncio.sleep(30)

                continue


            new_station = nearest["station"]

            new_distance = nearest["distance_km"]


            print(
                "NDBC station:",
                new_station
            )

            print(
                "Distance:",
                new_distance,
                "km"
            )


            # ========================================================
            # STATION CHANGE
            # ========================================================

            if (
                last_station is not None
                and new_station != last_station
            ):

                print()
                print(
                    "NDBC STATION CHANGED"
                )

                print(
                    "Old:",
                    last_station
                )

                print(
                    "New:",
                    new_station
                )

                # Preserve the last known real NDBC wave history.
                # Do not clear it until a fresh station history is confirmed.
                print(
                    "Preserving previous real NDBC wave history "
                    "during station change."
                )


            NDBC_ACTIVE_STATION = new_station

            NDBC_ACTIVE_STATION_DISTANCE_KM = new_distance

            last_station = new_station


            # ========================================================
            # BACKFILL HISTORY
            #
            # LSTM     -> 8 observations
            # XGBOOST  -> latest 3 of those 8
            # ========================================================

            if (
                len(LSTM_WAVE_HISTORY) < 8
                or len(NDBC_XGB_HISTORY) < 3
            ):

                print()
                print(
                    "NDBC BACKFILL STARTED"
                )

                history = (
                    _read_ndbc_station_history(
                        new_station,
                        8
                    )
                )


                if (
                    history is None
                ):

                    print()
                    print(
                        "NDBC backfill incomplete."
                    )

                    print(
                        "Required:",
                        8
                    )

                    print(
                        "Received:",
                        0 if history is None else len(history)
                    )

                    # ------------------------------------------------
                    # OFFLINE CACHE FALLBACK
                    # ------------------------------------------------
                    # Preserve and restore the latest previously
                    # obtained REAL NDBC observations.
                    # No synthetic observations are created.
                    # ------------------------------------------------

                    if load_offline_wave_cache():

                        NDBC_OFFLINE_MODE = True

                        print(
                            "NDBC: switching to OFFLINE LOCAL_CACHE."
                        )

                        if len(NDBC_XGB_HISTORY) >= 3:

                            try:

                                NDBC_LAST_XGB_RESULT = (
                                    run_ndbc_xgb_prediction(
                                        NDBC_XGB_HISTORY[-3:]
                                    )
                                )

                                NDBC_LAST_XGB_RESULT[
                                    "mode"
                                ] = "OFFLINE"

                                NDBC_LAST_XGB_RESULT[
                                    "data_source"
                                ] = "LOCAL_CACHE"

                                print(
                                    "OFFLINE XGBOOST RESULT:"
                                )

                                print(
                                    NDBC_LAST_XGB_RESULT
                                )

                            except Exception as e:

                                print(
                                    "Offline XGBoost error:",
                                    e
                                )

                        if len(LSTM_WAVE_HISTORY) >= 8:

                            try:

                                offline_lstm_result = (
                                    run_lstm_prediction(
                                        LSTM_WAVE_HISTORY[-8:]
                                    )
                                )

                                offline_lstm_result[
                                    "mode"
                                ] = "OFFLINE"

                                offline_lstm_result[
                                    "data_source"
                                ] = "LOCAL_CACHE"

                                print(
                                    "OFFLINE LSTM RESULT:"
                                )

                                print(
                                    offline_lstm_result
                                )

                            except Exception as e:

                                print(
                                    "Offline LSTM error:",
                                    e
                                )

                    else:

                        print(
                            "NDBC: offline cache unavailable."
                        )

                    await asyncio.sleep(30)

                    continue

                # ====================================================
                # LSTM HISTORY
                # ====================================================

                LSTM_WAVE_HISTORY.clear()


                for observation in history:

                    add_wave_observation(
                        observation["VHM0"],
                        observation["VTPK"],
                        observation["VPED"],
                        observation["timestamp"]
                    )


                # ====================================================
                # XGBOOST HISTORY
                #
                # ====================================================
                # XGBOOST HISTORY
                # Independent from LSTM.
                # Only complete 8-feature observations are allowed.
                # ====================================================

                xgb_history = [
                    observation
                    for observation in history
                    if observation.get("xgb_complete", False)
                ]

                NDBC_XGB_HISTORY.clear()

                NDBC_XGB_HISTORY.extend(
                    xgb_history[-3:]
                )

                print(
                    "XGBoost valid observations:",
                    len(NDBC_XGB_HISTORY),
                    "/ 3"
                )

                NDBC_LAST_TIMESTAMP = (
                    history[-1]["timestamp"]
                )


                print()
                print(
                    "============================================================"
                )

                print(
                    "NDBC BACKFILL COMPLETE"
                )

                print(
                    "============================================================"
                )

                print(
                    "AIS MMSI:",
                    mmsi
                )

                print(
                    "NDBC station:",
                    NDBC_ACTIVE_STATION
                )

                print(
                    "Distance:",
                    NDBC_ACTIVE_STATION_DISTANCE_KM,
                    "km"
                )

                print(
                    "LSTM observations:",
                    len(LSTM_WAVE_HISTORY),
                    "/ 8"
                )

                print(
                    "XGBoost observations:",
                    len(NDBC_XGB_HISTORY),
                    "/ 3"
                )


                # ====================================================
                # XGBOOST IMMEDIATE PREDICTION
                # ====================================================

                if (
                    len(NDBC_XGB_HISTORY) == 3
                ):

                    try:

                        NDBC_LAST_XGB_RESULT = (
                            run_ndbc_xgb_prediction(
                                NDBC_XGB_HISTORY
                            )
                        )


                        NDBC_LAST_XGB_RESULT[
                            "mode"
                        ] = "ONLINE"


                        NDBC_LAST_XGB_RESULT[
                            "data_source"
                        ] = "NDBC"


                        print()
                        print(
                            "============================================================"
                        )

                        print(
                            "BACKFILLED XGBOOST RESULT"
                        )

                        print(
                            "============================================================"
                        )

                        print(
                            "Probability:",
                            NDBC_LAST_XGB_RESULT.get(
                                "probability_percent"
                            ),
                            "%"
                        )

                        print(
                            "Hazard:",
                            NDBC_LAST_XGB_RESULT.get(
                                "hazard"
                            )
                        )

                        print(
                            "Observations:",
                            NDBC_LAST_XGB_RESULT.get(
                                "observations_used"
                            )
                        )

                        print(
                            "Features:",
                            NDBC_LAST_XGB_RESULT.get(
                                "features_used"
                            )
                        )


                    except Exception as e:

                        print(
                            "Backfilled XGBoost error:",
                            e
                        )


                # ====================================================
                # LSTM IMMEDIATE PREDICTION
                # ====================================================

                if (
                    len(LSTM_WAVE_HISTORY) >= 8
                ):

                    try:

                        lstm_result = (
                            run_lstm_prediction(
                                LSTM_WAVE_HISTORY[-8:]
                            )
                        )


                        print()
                        print(
                            "BACKFILLED LSTM RESULT:"
                        )

                        print(
                            lstm_result
                        )


                    except Exception as e:

                        print(
                            "Backfilled LSTM error:",
                            e
                        )


                try:

                    save_offline_wave_cache()

                except Exception as e:

                    print(
                        "Cache save error:",
                        e
                    )


            # ========================================================
            # FETCH NEWEST NDBC OBSERVATION
            # ========================================================

            observation = (
                fetch_ndbc_wave_observation()
            )


            if observation is not None:

                # Fresh NDBC data is available again.
                # Automatically return to ONLINE mode.
                NDBC_OFFLINE_MODE = False

                timestamp = observation.get(
                    "timestamp"
                )


                # ====================================================
                # AVOID DUPLICATES
                # ====================================================

                if (
                    timestamp
                    and timestamp != NDBC_LAST_TIMESTAMP
                ):

                    NDBC_LAST_TIMESTAMP = timestamp


                    # ================================================
                    # LSTM
                    # ================================================

                    add_wave_observation(
                        observation["VHM0"],
                        observation["VTPK"],
                        observation["VPED"],
                        observation["timestamp"]
                    )


                    print()
                    print(
                        "LSTM history:",
                        len(LSTM_WAVE_HISTORY),
                        "/ 8"
                    )


                    # ================================================
                    # XGBOOST
                    # ================================================

                    if observation.get("xgb_complete", False):

                        if observation.get("xgb_complete", False):
                            xgb_required = ["WVHT", "WSPD", "GST", "DPD", "APD", "PRES", "ATMP", "WTMP"]
                            xgb_valid = all(observation.get(k) is not None for k in xgb_required)
                            if xgb_valid:
                                NDBC_XGB_HISTORY.append(observation)


                    if (
                        len(NDBC_XGB_HISTORY) > 3
                    ):

                        del NDBC_XGB_HISTORY[:-3]


                    print(
                        "XGBoost history:",
                        len(NDBC_XGB_HISTORY),
                        "/ 3"
                    )


                    # ================================================
                    # LSTM PREDICTION
                    # ================================================

                    if (
                        len(LSTM_WAVE_HISTORY) >= 8
                    ):

                        try:

                            lstm_result = (
                                run_lstm_prediction(
                                    LSTM_WAVE_HISTORY[-8:]
                                )
                            )


                            print(
                                "LIVE LSTM RESULT:"
                            )

                            print(
                                lstm_result
                            )


                        except Exception as e:

                            print(
                                "LSTM prediction error:",
                                e
                            )


                    # ================================================
                    # XGBOOST PREDICTION
                    # ================================================

                    if (
                        len(NDBC_XGB_HISTORY) == 3
                    ):

                        try:

                            NDBC_LAST_XGB_RESULT = (
                                run_ndbc_xgb_prediction(
                                    NDBC_XGB_HISTORY
                                )
                            )


                            NDBC_LAST_XGB_RESULT[
                                "mode"
                            ] = "ONLINE"


                            NDBC_LAST_XGB_RESULT[
                                "data_source"
                            ] = "NDBC"


                            print()
                            print(
                                "LIVE XGBOOST RESULT:"
                            )

                            print(
                                NDBC_LAST_XGB_RESULT
                            )


                        except Exception as e:

                            print(
                                "XGBoost prediction error:",
                                e
                            )


                    try:

                        save_offline_wave_cache()

                    except Exception as e:

                        print(
                            "Cache save error:",
                            e
                        )


            else:

                print()
                print(
                    "NDBC: no new observation."
                )

                # ----------------------------------------------------
                # NDBC IS REACHABLE BUT HAS NO NEWER ROW
                # ----------------------------------------------------
                # This is NOT an offline condition.
                #
                # The selected NDBC station responded successfully,
                # but its latest published observation is the same
                # observation already being used.
                #
                # Keep the system ONLINE and continue predicting
                # from the existing real NDBC observations.
                # ----------------------------------------------------

                if NDBC_LAST_FETCH_OK:

                    NDBC_OFFLINE_MODE = False

                    print(
                        "NDBC: reachable, no new row; keeping ONLINE mode."
                    )

                else:

                    NDBC_OFFLINE_MODE = True

                print(
                    "NDBC: NDBC request failed; switching to OFFLINE LOCAL_CACHE."
                )

                if len(NDBC_XGB_HISTORY) >= 3:

                    try:

                        NDBC_LAST_XGB_RESULT = (
                            run_ndbc_xgb_prediction(
                                NDBC_XGB_HISTORY[-3:]
                            )
                        )

                        NDBC_LAST_XGB_RESULT[
                            "mode"
                        ] = "ONLINE"

                        NDBC_LAST_XGB_RESULT[
                            "data_source"
                        ] = "NDBC"

                        print()
                        print(
                            "ONLINE XGBOOST RESULT USING LATEST NDBC OBSERVATIONS:"
                        )

                        print(
                            NDBC_LAST_XGB_RESULT
                        )

                    except Exception as e:

                        print(
                            "Online XGBoost error:",
                            e
                        )


                if len(LSTM_WAVE_HISTORY) >= 8:

                    try:

                        online_lstm_result = (
                            run_lstm_prediction(
                                LSTM_WAVE_HISTORY[-8:]
                            )
                        )

                        if isinstance(
                            online_lstm_result,
                            dict
                        ):

                            online_lstm_result[
                                "mode"
                            ] = "ONLINE"

                            online_lstm_result[
                                "data_source"
                            ] = "NDBC"

                        print()
                        print(
                            "ONLINE LSTM RESULT USING LATEST NDBC OBSERVATIONS:"
                        )

                        print(
                            online_lstm_result
                        )

                    except Exception as e:

                        print(
                            "Online LSTM error:",
                            e
                        )

                    if len(NDBC_XGB_HISTORY) >= 3:

                        try:

                            NDBC_LAST_XGB_RESULT = (
                                run_ndbc_xgb_prediction(
                                    NDBC_XGB_HISTORY[-3:]
                                )
                            )

                            NDBC_LAST_XGB_RESULT[
                                "mode"
                            ] = "OFFLINE"

                            NDBC_LAST_XGB_RESULT[
                                "data_source"
                            ] = "LOCAL_CACHE"

                            print()
                            print(
                                "OFFLINE XGBOOST RESULT:"
                            )

                            print(
                                NDBC_LAST_XGB_RESULT
                            )

                        except Exception as e:

                            print(
                                "Offline XGBoost error:",
                                e
                            )

                    if len(LSTM_WAVE_HISTORY) >= 8:

                        try:

                            offline_lstm_result = (
                                run_lstm_prediction(
                                    LSTM_WAVE_HISTORY[-8:]
                                )
                            )

                            offline_lstm_result[
                                "mode"
                            ] = "OFFLINE"

                            offline_lstm_result[
                                "data_source"
                            ] = "LOCAL_CACHE"

                            print()
                            print(
                                "OFFLINE LSTM RESULT:"
                            )

                            print(
                                offline_lstm_result
                            )

                        except Exception as e:

                            print(
                                "Offline LSTM error:",
                                e
                            )

                else:

                    print(
                        "NDBC: offline cache unavailable."
                    )


            await asyncio.sleep(
                30
            )


        except Exception as e:

            print()
            print(
                "NDBC loop error:",
                e
            )

            await asyncio.sleep(
                30
            )

NDBC_XGB_HISTORY = []

# Load cached wave history after all history variables are initialized


NDBC_XGB_HISTORY_MAX = 3
NDBC_LAST_XGB_RESULT = None

@asynccontextmanager
async def lifespan(
    app
):

    global AIS_TASK
    global NDBC_TASK

    print()
    print("=" * 70)
    print(
        "AI MARINE MONITORING SYSTEM"
    )
    print("=" * 70)


    # ------------------------------------------------------------
    # START AISSTREAM
    # ------------------------------------------------------------

    # RESTORE OFFLINE WAVE CACHE
    load_offline_wave_cache()

    if len(NDBC_XGB_HISTORY) >= 3:
        try:
            NDBC_LAST_XGB_RESULT = run_ndbc_xgb_prediction(NDBC_XGB_HISTORY[-3:])
        except Exception:
            pass



    # ------------------------------------------------------------
    # START AISSTREAM BACKUP
    # ------------------------------------------------------------

    AIS_TASK = asyncio.create_task(
        ais_stream_worker()
    )


    # ------------------------------------------------------------
    # START NDBC WAVE DATA WORKER
    # ------------------------------------------------------------

    NDBC_TASK = asyncio.create_task(
        ndbc_wave_loop()
    )


    try:

        yield

    finally:

        # --------------------------------------------------------
        # STOP AIS
        # --------------------------------------------------------

        if AIS_TASK is not None:

            AIS_TASK.cancel()

            try:

                await AIS_TASK

            except asyncio.CancelledError:

                pass


        # --------------------------------------------------------
        # STOP AISSTREAM
        # --------------------------------------------------------



        # --------------------------------------------------------
        # STOP NDBC WAVE TASK
        # --------------------------------------------------------

        if NDBC_TASK is not None:

            NDBC_TASK.cancel()

            try:

                await NDBC_TASK

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
# NDBC XGBOOST RESULT
# ============================================================

@app.get(
    "/wave/ndbc/xgboost"
)
def get_ndbc_xgboost_result():

    global NDBC_LAST_XGB_RESULT

    # ========================================================
    # ALWAYS CALCULATE FROM CURRENT 3 VALID OBSERVATIONS
    # ========================================================

    if len(NDBC_XGB_HISTORY) < 3:

        return {
            "status": "waiting",
            "required": 3,
            "count": len(NDBC_XGB_HISTORY),
            "result": None
        }

    try:

        latest_history = NDBC_XGB_HISTORY[-3:]

        print()
        print("============================================================")
        print("NDBC XGBOOST API PREDICTION")
        print("============================================================")
        print("Observations:", len(latest_history))

        NDBC_LAST_XGB_RESULT = run_ndbc_xgb_prediction(
            latest_history
        )

        if NDBC_LAST_XGB_RESULT is None:

            return {
                "status": "error",
                "required": 3,
                "count": len(NDBC_XGB_HISTORY),
                "result": None,
                "error": "XGBoost returned None"
            }

        NDBC_LAST_XGB_RESULT["mode"] = "ONLINE"
        NDBC_LAST_XGB_RESULT["data_source"] = "NDBC"

        print(
            "NDBC XGBOOST API RESULT:",
            NDBC_LAST_XGB_RESULT
        )

        return NDBC_LAST_XGB_RESULT

    except Exception as e:

        print(
            "NDBC XGBOOST API ERROR:",
            repr(e)
        )

        return {
            "status": "error",
            "required": 3,
            "count": len(NDBC_XGB_HISTORY),
            "result": None,
            "error": str(e)
        }


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

    # NDBC_OFFLINE_MODE is controlled by the NDBC
    # background worker.
    operation_mode = (
        "OFFLINE"
        if NDBC_OFFLINE_MODE
        else "ONLINE"
    )

    wave_data_source = (
        "LOCAL_CACHE"
        if NDBC_OFFLINE_MODE
        else "NDBC"
    )

    return {

        "status":
            "healthy",

        "operation_mode":
            operation_mode,

        "wave_data_source":
            wave_data_source,

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

        "ndbc": {
            "active_station":
                NDBC_ACTIVE_STATION,

            "station_distance_km":
                NDBC_ACTIVE_STATION_DISTANCE_KM,

            "lstm_observations":
                len(
                    LSTM_WAVE_HISTORY
                ),

            "xgboost_observations":
                len(
                    NDBC_XGB_HISTORY
                )
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


@app.get(
    "/ais/vessels"
)
async def get_ais_vessels():

    # ========================================================
    # AISSTREAM-ONLY LIVE VESSELS
    # ========================================================

    vessels = []

    for vessel in AIS_VESSELS.values():

        record = build_vessel_record(
            vessel
        )

        if record.get("mmsi") is not None:
            vessels.append(
                record
            )

    return {
        "status":
            "success",

        "count":
            len(vessels),

        "vessels":
            vessels,

        "sources": {
            "AISStream":
                len(AIS_VESSELS),

            "combined":
                len(vessels)
        },

        "message":
            "Live vessels provided by AISStream."
    }


# ============================================================
# FIND VESSEL
# ============================================================

def find_vessel_by_mmsi(mmsi):

    target = str(int(mmsi))

    # ========================================================
    # AISSTREAM LIVE VESSEL LOOKUP
    # ========================================================

    try:

        # Direct lookup
        vessel = AIS_VESSELS.get(target)

        if vessel is not None:
            return vessel

        # Try integer key
        vessel = AIS_VESSELS.get(int(target))

        if vessel is not None:
            return vessel

        # Search through all AISStream vessels
        for item in AIS_VESSELS.values():

            try:

                if str(item.get("mmsi")) == target:
                    return item

            except Exception:
                continue

    except Exception as e:

        print(
            "AISStream vessel search error:",
            e
        )

    # ========================================================
    # DEMO FALLBACK
    # ========================================================

    try:

        fallback = load_fallback_vessels()

        # Direct lookup
        vessel = fallback.get(target)

        if vessel is not None:
            return vessel

        # Search fallback records
        for item in fallback.values():

            try:

                if str(item.get("mmsi")) == target:
                    return item

            except Exception:
                continue

    except Exception as e:

        print(
            "Fallback vessel search error:",
            e
        )

    # ========================================================
    # NOT FOUND
    # ========================================================

    print(
        f"MMSI {target} not found in "
        f"AISStream or fallback."
    )

    return None


# ============================================================
# AIS RISK FOR ONE VESSEL
# ============================================================


# ============================================================
# AIS RISK FOR ONE VESSEL
# ============================================================



# ============================================================
# SELECT VESSEL FROM LOGIN
# ============================================================

@app.post("/select-vessel")
async def select_vessel(payload: dict):

    global MONITORED_MMSI
    global NDBC_ACTIVE_STATION
    global NDBC_ACTIVE_STATION_DISTANCE_KM

    # --------------------------------------------------------
    # READ LOGIN DETAILS
    # --------------------------------------------------------

    raw_mmsi = payload.get(
        "mmsi"
    )

    ship_name = str(
        payload.get(
            "ship_name",
            ""
        )
    ).strip()

    ship_type = str(
        payload.get(
            "ship_type",
            ""
        )
    ).strip()


    # --------------------------------------------------------
    # VALIDATE MMSI
    # --------------------------------------------------------

    mmsi_text = str(
        raw_mmsi
    ).strip()

    if (
        not mmsi_text.isdigit()
        or len(mmsi_text) != 9
    ):

        raise HTTPException(
            status_code=422,
            detail="MMSI must contain exactly 9 digits."
        )


    mmsi = int(
        mmsi_text
    )


    print()
    print(
        "============================================================"
    )
    print(
        "LOGIN VESSEL SELECTION"
    )
    print(
        "============================================================"
    )

    print(
        "LOGIN SHIP NAME:",
        ship_name
    )

    print(
        "LOGIN MMSI:",
        mmsi
    )

    print(
        "LOGIN SHIP TYPE:",
        ship_type
    )


    # --------------------------------------------------------
    # TRIGGER EXISTING AIS -> NDBC PIPELINE
    # --------------------------------------------------------

    result = await get_live_vessel_risk(
        mmsi
    )

    # --------------------------------------------------------
    # CAPTURE NDBC STATE FOR THIS LOGIN REQUEST
    # --------------------------------------------------------
    # A background NDBC worker can run at the same time and
    # change the global NDBC_OFFLINE_MODE. Do not allow that
    # race to change the result of a successful live login.
    #
    # If this request has a valid NDBC station and the required
    # live observations, this login is ONLINE.
    # --------------------------------------------------------

    login_ndbc_online = (
        MONITORED_NDBC_STATION is not None
        and len(LSTM_WAVE_HISTORY) >= 8
        and len(NDBC_XGB_HISTORY) >= 3
    )

    if login_ndbc_online:

        print(
            "LOGIN NDBC STATE: ONLINE"
        )

    else:

        print(
            "LOGIN NDBC STATE: OFFLINE / NO LIVE NDBC HISTORY"
        )


    # --------------------------------------------------------
    # KEEP MONITORED MMSI AVAILABLE TO NDBC WORKER
    # --------------------------------------------------------

    MONITORED_MMSI = mmsi


    # --------------------------------------------------------
    # SYNCHRONIZE NDBC HEALTH STATUS
    # --------------------------------------------------------

    if (
        MONITORED_NDBC_STATION is not None
    ):

        NDBC_ACTIVE_STATION = (
            MONITORED_NDBC_STATION
        )

        NDBC_ACTIVE_STATION_DISTANCE_KM = (
            MONITORED_NDBC_DISTANCE_KM
        )


    # --------------------------------------------------------
    # LSTM IMMEDIATE PREDICTION
    # --------------------------------------------------------

    lstm_result = None

    if (
        len(LSTM_WAVE_HISTORY) >= 8
    ):

        try:

            lstm_result = await asyncio.to_thread(
                run_lstm_prediction,
                LSTM_WAVE_HISTORY[-8:]
            )

            if isinstance(
                lstm_result,
                dict
            ):

                lstm_result[
                    "mode"
                ] = (
                    "ONLINE"
                    if login_ndbc_online
                    else "OFFLINE"
                )

                lstm_result[
                    "data_source"
                ] = (
                    "NDBC"
                    if login_ndbc_online
                    else "LOCAL_CACHE"
                )

        except Exception as e:

            print(
                "LOGIN LSTM ERROR:",
                e
            )

            lstm_result = {
                "status": "error",
                "error": str(e)
            }


    # --------------------------------------------------------
    # NDBC RESULT
    # --------------------------------------------------------

    # Use the station that actually produced the XGBoost result.
    actual_ndbc_station = (
        NDBC_LAST_XGB_RESULT.get("station")
        if isinstance(NDBC_LAST_XGB_RESULT, dict)
        else None
    )

    if actual_ndbc_station:
        NDBC_ACTIVE_STATION = actual_ndbc_station

    ndbc_result = {

        "active_station":
            NDBC_ACTIVE_STATION,

        "station_distance_km":
            NDBC_ACTIVE_STATION_DISTANCE_KM,

        "lstm_observations":
            len(
                LSTM_WAVE_HISTORY
            ),

        "xgboost_observations":
            len(
                NDBC_XGB_HISTORY
            ),

        "mode":
            (
                "ONLINE"
                if login_ndbc_online
                else "OFFLINE"
            ),

        "data_source":
            (
                "NDBC"
                if login_ndbc_online
                else "LOCAL_CACHE"
            ),

        "xgboost":
            NDBC_LAST_XGB_RESULT,

        "lstm":
            lstm_result
    }


    # --------------------------------------------------------
    # FINAL LOGIN RESPONSE
    # --------------------------------------------------------

    return {

        "status":
            "success",

        "login": {

            "ship_name":
                ship_name,

            "mmsi":
                mmsi,

            "ship_type":
                ship_type
        },

        "vessel":
            result.get(
                "vessel"
            ),

        "random_forest":
            result.get(
                "random_forest"
            ),

        "xgboost":
            result.get(
                "xgboost"
            ),

        "ndbc":
            ndbc_result
    }


@app.get(
    "/ais/risk/{mmsi}"
)
async def get_live_vessel_risk(
    mmsi: int
):
    global ACTIVE_MONITORED_MMSI
    global MONITORED_MMSI
    global MONITORED_NDBC_STATION
    global MONITORED_NDBC_DISTANCE_KM
    global MONITORED_VESSEL_LAT
    global MONITORED_VESSEL_LON
    global NDBC_LAST_TIMESTAMP
    global NDBC_LAST_XGB_RESULT
    global NDBC_OFFLINE_MODE

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


    # ========================================================
    # SELECT AIS VESSEL FOR NDBC MONITORING
    # ========================================================

    vessel_changed = (
    ACTIVE_MONITORED_MMSI
    != mmsi
    )


    if vessel_changed:

        set_active_monitored_vessel(
            mmsi
        )

        MONITORED_NDBC_STATION = None

        MONITORED_NDBC_DISTANCE_KM = None

        MONITORED_VESSEL_LAT = (
            latitude
        )

        MONITORED_VESSEL_LON = (
            longitude
        )

        NDBC_LAST_TIMESTAMP = None

        # Keep cached NDBC/LSTM history when switching AIS vessels
        LSTM_WAVE_HISTORY.clear()

        NDBC_XGB_HISTORY.clear()


        print()
        print(
            "============================================================"
        )

        print(
            "AIS MONITORED VESSEL CHANGED"
        )

        print(
            "============================================================"
        )

        print(
            "MMSI:",
            MONITORED_MMSI
        )

        print(
            "Position:",
            latitude,
            longitude
        )

        print(
            "NDBC histories reset."
        )


        # ====================================================
        # IMMEDIATELY BACKFILL:
        #
        # LSTM    = 8 observations
        # XGBOOST = 3 observations
        # ====================================================

        startup_history = (
            await asyncio.to_thread(
                fetch_ndbc_recent_observations,
                8,
                latitude,
                longitude
            )
        )


        if len(startup_history) >= 8:
            # ------------------------------------------------
            # LIVE NDBC DATA SUCCESS
            # ------------------------------------------------
            # Complete history came from the live NDBC request.
            # Mark the NDBC system ONLINE.
            # ------------------------------------------------

            NDBC_OFFLINE_MODE = False

            print()
            print(
                "NDBC LIVE DATA AVAILABLE -> ONLINE MODE"
            )


            LSTM_WAVE_HISTORY.clear()


            for observation in startup_history:

                add_wave_observation(
                    observation["VHM0"],
                    observation["VTPK"],
                    observation["VPED"],
                    observation["timestamp"]
                )


            NDBC_XGB_HISTORY.clear()

            xgb_required = [
                "WVHT", "WSPD", "GST", "DPD",
                "APD", "PRES", "ATMP", "WTMP"
            ]

            valid_xgb = [
                observation
                for observation in startup_history
                if all(
                    observation.get(feature) is not None
                    for feature in xgb_required
                )
            ]

            NDBC_XGB_HISTORY.extend(
                valid_xgb[-3:]
            )


            NDBC_LAST_TIMESTAMP = (
                startup_history[-1]["timestamp"]
            )


            save_offline_wave_cache()


            print()
            print(
                "AIS -> NDBC HISTORICAL BACKFILL"
            )

            print(
                "LSTM:",
                len(LSTM_WAVE_HISTORY),
                "/ 8"
            )

            print(
                "XGBoost:",
                len(NDBC_XGB_HISTORY),
                "/ 3"
            )


            try:

                NDBC_LAST_XGB_RESULT = (
                    run_ndbc_xgb_prediction(
                        NDBC_XGB_HISTORY
                    )
                )

                NDBC_LAST_XGB_RESULT[
                    "mode"
                ] = "ONLINE"

                NDBC_LAST_XGB_RESULT[
                    "data_source"
                ] = "NDBC"

                NDBC_OFFLINE_MODE = False


                print(
                    "BACKFILLED XGBoost RESULT:"
                )

                print(
                    NDBC_LAST_XGB_RESULT
                )


            except Exception as e:

                print(
                    "Backfilled XGBoost error:",
                    e
                )


        else:

            print(
                "No complete NDBC history found "
                "for selected AIS vessel."
            )


    else:

        MONITORED_VESSEL_LAT = (
            latitude
        )

        MONITORED_VESSEL_LON = (
            longitude
        )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

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


    # ========================================================
    # NORMAL XGBOOST VESSEL RISK
    # ========================================================

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


    # ========================================================
    # LSTM WAVE FORECAST
    # ========================================================

    lstm_result = None

    if len(LSTM_WAVE_HISTORY) >= 8:

        try:

            lstm_result = run_lstm_prediction(
                LSTM_WAVE_HISTORY[-8:]
            )

        except Exception as e:

            lstm_result = {
                "status": "error",
                "error": str(e)
            }

    # ========================================================
    # FINAL AIS RISK RESPONSE
    # ========================================================

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
            xgb_result,

        "ndbc_xgboost":
            NDBC_LAST_XGB_RESULT,

        "lstm":
            lstm_result

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



# ============================================================
# HIGH WAVE XGBOOST
# ============================================================



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



# ============================================================
# WAVE HISTORY
# ============================================================



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



# ============================================================
# PPO ROUTE FROM AIS - SAFE VERSION
# ============================================================

@app.post(
    "/optimize-route-from-ais"
)
def optimize_route_from_ais(
    request: AISRouteRequest
):

    # --------------------------------------------------------
    # Route request also defines the monitored vessel
    # --------------------------------------------------------

    set_active_monitored_vessel(
        request.mmsi
    )

    # ========================================================
    # CHECK PPO MODEL
    # ========================================================

    if ppo_model is None:

        raise HTTPException(
            status_code=500,
            detail="PPO model is not loaded."
        )

    # ========================================================
    # FIND VESSEL
    # AISStream is the live vessel source through find_vessel_by_mmsi()
    # ========================================================

    vessel = find_vessel_by_mmsi(
        request.mmsi
    )

    if vessel is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Vessel MMSI {request.mmsi} "
                "not found."
            )
        )

    # ========================================================
    # CURRENT VESSEL DATA
    # ========================================================

    current_lat = float(
        vessel["latitude"]
    )

    current_lon = float(
        vessel["longitude"]
    )

    current_speed = float(
        vessel.get(
            "speed",
            vessel.get(
                "sog",
                0.0
            )
        )
    )

    current_heading = float(
        vessel.get(
            "heading",
            vessel.get(
                "course",
                vessel.get(
                    "cog",
                    0.0
                )
            )
        )
    )

    destination_lat = float(
        request.destination_lat
    )

    destination_lon = float(
        request.destination_lon
    )

    # ========================================================
    # BASIC COORDINATE VALIDATION
    # ========================================================

    if not (
        -90.0 <= current_lat <= 90.0
        and
        -180.0 <= current_lon <= 180.0
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid current vessel coordinates."
        )

    if not (
        -90.0 <= destination_lat <= 90.0
        and
        -180.0 <= destination_lon <= 180.0
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid destination coordinates."
        )

    # ========================================================
    # OCEAN VALIDATION
    # ========================================================

    if point_is_land(
        current_lat,
        current_lon
    ):

        raise HTTPException(
            status_code=400,
            detail="Current vessel position is on land."
        )

    if point_is_land(
        destination_lat,
        destination_lon
    ):

        raise HTTPException(
            status_code=400,
            detail="Destination is on land."
        )

    # ========================================================
    # DEFAULT HAZARD STATE
    # ========================================================

    route_mode = "OPTIMIZED"

    hazard = "NO_WAVE_DATA"

    wave_status = "NO_WAVE_DATA"

    wave_prediction = None

    xgb_probability = None

    # ========================================================
    # MONITORING PAGE HAZARD
    # ========================================================

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

    # ========================================================
    # NDBC XGBOOST HIGH-WAVE MODEL
    # ========================================================

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

            hazard = "HIGH_WAVE"

            wave_status = "HIGH_WAVE"

            route_mode = "SAFEST"

        else:

            hazard = "NORMAL"

            wave_status = "LOW"

            route_mode = "OPTIMIZED"

    # ========================================================
    # LSTM FALLBACK
    # ========================================================

    elif (
    not request.route_hazard
    and
    len(LSTM_WAVE_HISTORY) >= 8
    ):

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

                hazard = "HIGH_WAVE"

                route_mode = "SAFEST"

            elif wave_prediction >= 2.0:

                hazard = "MODERATE"

                route_mode = "OPTIMIZED"

            else:

                hazard = "NORMAL"

                route_mode = "OPTIMIZED"

        except Exception as e:

            print(
                "LSTM route prediction error:",
                e
            )

    # ========================================================
    # INITIAL ROUTE INFORMATION
    # ========================================================

    initial_distance = haversine_km(
        current_lat,
        current_lon,
        destination_lat,
        destination_lon
    )

    direct_bearing = bearing_degrees(
        current_lat,
        current_lon,
        destination_lat,
        destination_lon
    )

    heading_error = angle_difference(
        current_heading,
        direct_bearing
    )

    # ========================================================
    # PPO ACTION MAP
    # ========================================================

    turn_amounts = {

        0: -10.0,
        1: -5.0,
        2: 0.0,
        3: 5.0,
        4: 10.0

    }

    # ========================================================
    # ROUTE START
    # ========================================================

    route_points = [

        {
            "latitude":
                current_lat,

            "longitude":
                current_lon,

            "action":
                None,

            "turn_angle":
                0.0,

            "ppo_turn_angle":
                0.0,

            "heading":
                current_heading,

            "step":
                0
        }

    ]

    simulation_lat = current_lat

    simulation_lon = current_lon

    simulation_heading = current_heading

    # ========================================================
    # ROUTE SIMULATION
    # ========================================================

    max_steps = 150

    reached_destination = False

    for step in range(
        max_steps
    ):

        distance = haversine_km(
            simulation_lat,
            simulation_lon,
            destination_lat,
            destination_lon
        )

        # ----------------------------------------------------
        # DESTINATION REACHED
        # ----------------------------------------------------

        if distance <= 2.0:

            reached_destination = True

            break

        # ----------------------------------------------------
        # DIRECT BEARING
        # ----------------------------------------------------

        desired_bearing = bearing_degrees(
            simulation_lat,
            simulation_lon,
            destination_lat,
            destination_lon
        )

        heading_error = angle_difference(
            simulation_heading,
            desired_bearing
        )

        # ----------------------------------------------------
        # PPO OBSERVATION
        # ----------------------------------------------------

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

                heading_error / 180.0

            ],

            dtype=np.float32
        )

        # ----------------------------------------------------
        # PPO PREDICTION
        # ----------------------------------------------------

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

        # ====================================================
        # SAFETY CANDIDATES
        #
        # IMPORTANT:
        # Direct bearing is now included.
        # This prevents PPO from getting trapped near land.
        # ====================================================

        candidate_turns = [

            # PPO choice first
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
            ppo_turn - 90,

            # DIRECT DESTINATION BEARING
            angle_difference(
                simulation_heading,
                desired_bearing
            ),

            # Small corrections toward destination
            angle_difference(
                simulation_heading,
                desired_bearing
            ) + 10,

            angle_difference(
                simulation_heading,
                desired_bearing
            ) - 10

        ]

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        unique_turns = []

        for turn in candidate_turns:

            turn = float(turn)

            if not any(
                abs(
                    turn - existing
                ) < 0.01
                for existing
                in unique_turns
            ):

                unique_turns.append(
                    turn
                )

        # ----------------------------------------------------
        # STEP SIZE
        # ----------------------------------------------------

        step_distance = min(

            5.0,

            max(
                0.5,
                distance / 20.0
            )

        )

        selected = None

        earth_radius = 6371.0

        # ====================================================
        # TEST EACH CANDIDATE
        # ====================================================

        for turn in unique_turns:

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

            # ------------------------------------------------
            # NORMALIZE LONGITUDE
            # ------------------------------------------------

            if candidate_lon > 180:

                candidate_lon -= 360

            if candidate_lon < -180:

                candidate_lon += 360

            # ------------------------------------------------
            # LAND POINT CHECK
            # ------------------------------------------------

            if point_is_land(
                candidate_lat,
                candidate_lon
            ):

                continue

            # ------------------------------------------------
            # LAND SEGMENT CHECK
            # ------------------------------------------------

            if segment_crosses_land(

                simulation_lat,
                simulation_lon,

                candidate_lat,
                candidate_lon

            ):

                continue

            # ------------------------------------------------
            # ALSO CHECK SEGMENT TO DESTINATION
            #
            # If candidate is close enough to destination,
            # make sure final approach is safe.
            # ------------------------------------------------

            remaining_after_candidate = (
                haversine_km(
                    candidate_lat,
                    candidate_lon,
                    destination_lat,
                    destination_lon
                )
            )

            if remaining_after_candidate <= 8.0:

                if segment_crosses_land(

                    candidate_lat,
                    candidate_lon,

                    destination_lat,
                    destination_lon

                ):

                    continue

            # ------------------------------------------------
            # ------------------------------------------------
            # ROUTE CANDIDATE SCORING
            # ------------------------------------------------

            remaining_distance = haversine_km(
                candidate_lat,
                candidate_lon,
                destination_lat,
                destination_lon
            )

            turn_penalty = abs(float(turn))

            # NORMAL / OPTIMIZED:
            # Prefer efficient progress toward destination.
            if route_mode == "OPTIMIZED":

                candidate_score = (
                    remaining_distance
                    +
                    (turn_penalty * 0.02)
                )

            # SAFEST / HIGH_WAVE:
            # Prefer smoother and more conservative turns.
            else:

                candidate_score = (
                    remaining_distance
                    +
                    (turn_penalty * 0.12)
                )

            if hazard == "HIGH_WAVE":

                candidate_score += (
                    turn_penalty * 0.10
                )

            # Keep the best SAFE candidate.
            if (
                selected is None
                or
                candidate_score < selected[4]
            ):

                selected = (

                    candidate_lat,
                    candidate_lon,
                    candidate_heading,
                    turn,
                    candidate_score

                )


        # NO SAFE PPO STEP
        # ====================================================

        if selected is None:

            print(
                "PPO could not find a safe step."
            )

            print(
                "Switching to ocean-safe fallback."
            )

            break

        # ----------------------------------------------------
        # APPLY SELECTED STEP
        # ----------------------------------------------------

        (

            simulation_lat,
            simulation_lon,
            simulation_heading,
            selected_turn,
            _selected_score

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

    # ========================================================
    # SAFE FALLBACK ROUTE
    #
    # If PPO cannot produce a complete ocean-safe route,
    # search for a validated sea-only route using intermediate
    # waypoints.
    # ========================================================


    # ========================================================
    # GLOBAL OCEAN ROUTER
    #
    # Direct route is used only when it is ocean-safe.
    # If land blocks the direct route, A* searches through
    # ocean cells and creates a multi-turn safe route.
    # ========================================================

    def create_global_ocean_route(
        start_lat,
        start_lon,
        goal_lat,
        goal_lon,
        grid_step=0.10
    ):

        print()
        print("=" * 70)
        print("GLOBAL OCEAN ROUTER STARTED")
        print("=" * 70)

        start_lat = float(start_lat)
        start_lon = float(start_lon)
        goal_lat = float(goal_lat)
        goal_lon = float(goal_lon)

        # ========================================================
        # ENDPOINT VALIDATION
        # ========================================================

        if point_is_land(
            start_lat,
            start_lon
        ):

            print(
                "GLOBAL ROUTER: vessel is on land."
            )

            return None

        if point_is_land(
            goal_lat,
            goal_lon
        ):

            print(
                "GLOBAL ROUTER: destination is on land."
            )

            return None

        # ========================================================
        # DIRECT ROUTE
        #
        # Straight line is accepted ONLY if completely safe.
        # ========================================================

        if not segment_crosses_land(
            start_lat,
            start_lon,
            goal_lat,
            goal_lon
        ):

            print(
            )

            print(
                "GLOBAL ROUTER: direct path is ocean-safe."
            )

            print(
                "GLOBAL ROUTER: continuing to A* for route optimization."
            )

        else:

            print(
                "GLOBAL ROUTER: direct route crosses land."
            )

        # ========================================================
        # SEARCH AREA
        #
        # Give the router enough room to go around:
        # UK
        # France
        # Spain
        # Scandinavia
        # islands
        # peninsulas
        # ========================================================

        min_lat = min(
            start_lat,
            goal_lat
        )

        max_lat = max(
            start_lat,
            goal_lat
        )

        min_lon = min(
            start_lon,
            goal_lon
        )

        max_lon = max(
            start_lon,
            goal_lon
        )

        lat_span = (
            max_lat - min_lat
        )

        lon_span = (
            max_lon - min_lon
        )

        # --------------------------------------------------------
        # ADAPTIVE GLOBAL GRID
        # --------------------------------------------------------

        max_grid_cells = 90

        required_step_lat = (
            lat_span / max_grid_cells
            if lat_span > 0
            else grid_step
        )

        required_step_lon = (
            lon_span / max_grid_cells
            if lon_span > 0
            else grid_step
        )

        grid_step = max(
            grid_step,
            required_step_lat,
            required_step_lon
        )

        grid_step = min(
            grid_step,
            2.0
        )

        print(
            "GLOBAL ROUTER ADAPTIVE GRID STEP:",
            round(grid_step, 4)
        )


        # --------------------------------------------------------
        # SEARCH PADDING
        #
        # Keep a small buffer around the start/destination area.
        # The grid resolution is already adaptive for long routes.
        # --------------------------------------------------------

        padding_lat = max(
            3.0,
            grid_step * 5.0
        )

        padding_lon = max(
            3.0,
            grid_step * 5.0
        )

        min_lat = max(
            -80.0,
            min_lat - padding_lat
        )

        max_lat = min(
            80.0,
            max_lat + padding_lat
        )

        min_lon = max(
            -180.0,
            min_lon - padding_lon
        )

        max_lon = min(
            180.0,
            max_lon + padding_lon
        )

        # ========================================================
        # GRID SIZE
        # ========================================================

        lat_count = (
            int(
                math.ceil(
                    (
                        max_lat - min_lat
                    )
                    /
                    grid_step
                )
            )
            + 1
        )

        lon_count = (
            int(
                math.ceil(
                    (
                        max_lon - min_lon
                    )
                    /
                    grid_step
                )
            )
            + 1
        )

        # Prevent runaway memory/time.
        lat_count = min(lat_count, 100)

        lon_count = min(lon_count, 100)

        print(
            "GLOBAL ROUTER GRID:",
            lat_count,
            "x",
            lon_count
        )

        # ========================================================
        # GRID CONVERSION
        # ========================================================

        def node_to_coord(
            node
        ):

            row, column = node

            latitude = (
                min_lat
                +
                row * grid_step
            )

            longitude = (
                min_lon
                +
                column * grid_step
            )

            return (
                latitude,
                longitude
            )

        def coord_to_node(
            latitude,
            longitude
        ):

            row = int(
                round(
                    (
                        latitude - min_lat
                    )
                    /
                    grid_step
                )
            )

            column = int(
                round(
                    (
                        longitude - min_lon
                    )
                    /
                    grid_step
                )
            )

            row = max(
                0,
                min(
                    lat_count - 1,
                    row
                )
            )

            column = max(
                0,
                min(
                    lon_count - 1,
                    column
                )
            )

            return (
                row,
                column
            )

        # ========================================================
        # FIND SAFE GRID NODE NEAR REAL ENDPOINT
        # ========================================================

        def nearest_safe_node(
            latitude,
            longitude
        ):

            base_row, base_column = (
                coord_to_node(
                    latitude,
                    longitude
                )
            )

            # Adaptive endpoint search radius.
            # Coarser grids need more cells to reach the real
            # ocean endpoint while still respecting land checks.
            max_radius = max(
                8,
                min(
                    30,
                    int(
                        math.ceil(
                            1.5 / max(
                                grid_step,
                                0.01
                            )
                        )
                    )
                )
            )

            best_node = None
            best_distance = float(
                "inf"
            )

            for radius in range(
                max_radius + 1
            ):

                for row_change in range(
                    -radius,
                    radius + 1
                ):

                    for column_change in range(
                        -radius,
                        radius + 1
                    ):

                        row = (
                            base_row
                            +
                            row_change
                        )

                        column = (
                            base_column
                            +
                            column_change
                        )

                        if (
                            row < 0
                            or
                            row >= lat_count
                            or
                            column < 0
                            or
                            column >= lon_count
                        ):
                            continue

                        node = (
                            row,
                            column
                        )

                        node_lat, node_lon = (
                            node_to_coord(
                                node
                            )
                        )

                        if point_is_land(
                            node_lat,
                            node_lon
                        ):
                            continue

                        if segment_crosses_land(
                            latitude,
                            longitude,
                            node_lat,
                            node_lon
                        ):
                            continue

                        distance = (
                            haversine_km(
                                latitude,
                                longitude,
                                node_lat,
                                node_lon
                            )
                        )

                        if distance < best_distance:

                            best_distance = (
                                distance
                            )

                            best_node = node

                if best_node is not None:
                    break

            return best_node

        start_node = nearest_safe_node(
            start_lat,
            start_lon
        )

        goal_node = nearest_safe_node(
            goal_lat,
            goal_lon
        )

        if start_node is None:

            print(
                "GLOBAL ROUTER: no safe grid node near vessel."
            )

            return None

        if goal_node is None:

            print(
                "GLOBAL ROUTER: no safe grid node near destination."
            )

            return None

        print(
            "GLOBAL ROUTER START NODE:",
            start_node
        )

        print(
            "GLOBAL ROUTER GOAL NODE:",
            goal_node
        )

        # ========================================================
        # OCEAN NODE CACHE
        # ========================================================

        ocean_cache = {}

        def is_ocean_node(
            node
        ):

            if node in ocean_cache:
                return ocean_cache[node]

            latitude, longitude = (
                node_to_coord(
                    node
                )
            )

            result = not point_is_land(
                latitude,
                longitude
            )

            ocean_cache[node] = result

            return result

        # ========================================================
        # EDGE SAFETY CACHE
        # ========================================================

        edge_cache = {}

        def safe_edge(
            node_a,
            node_b
        ):

            key = (
                node_a,
                node_b
            )

            reverse_key = (
                node_b,
                node_a
            )

            if key in edge_cache:
                return edge_cache[key]

            if reverse_key in edge_cache:
                return edge_cache[reverse_key]

            lat1, lon1 = (
                node_to_coord(
                    node_a
                )
            )

            lat2, lon2 = (
                node_to_coord(
                    node_b
                )
            )

            result = (
                is_ocean_node(node_b)
                and
                not segment_crosses_land(
                    lat1,
                    lon1,
                    lat2,
                    lon2
                )
            )

            edge_cache[key] = result

            return result

        # ========================================================
        # A* HEURISTIC
        # ========================================================

        def heuristic(
            node
        ):

            latitude, longitude = (
                node_to_coord(
                    node
                )
            )

            return haversine_km(
                latitude,
                longitude,
                goal_lat,
                goal_lon
            )

        # ========================================================
        # 8-DIRECTION MOVEMENT
        # ========================================================

        directions = [

            (-1, -1),
            (-1,  0),
            (-1,  1),

            ( 0, -1),
            ( 0,  1),

            ( 1, -1),
            ( 1,  0),
            ( 1,  1)

        ]

        # ========================================================
        # A* SEARCH
        # ========================================================

        # ========================================================
        # FAST A* OCEAN SEARCH
        # ========================================================

        open_set = []

        heapq.heappush(
            open_set,
            (
                heuristic(
                    start_node
                ),
                0.0,
                start_node
            )
        )

        came_from = {}

        g_score = {
            start_node: 0.0
        }

        visited = set()

        found = False

        # --------------------------------------------------------
        # Practical search corridor.
        # This prevents very large global searches.
        # --------------------------------------------------------

        # Search the complete generated grid.
        #
        # The grid itself is bounded to prevent runaway memory
        # usage. Land rejection remains enforced by safe_edge().
        corridor_margin = max(
            lat_count,
            lon_count
        )

        min_search_row = max(
            0,
            min(
                start_node[0],
                goal_node[0]
            )
            -
            corridor_margin
        )

        max_search_row = min(
            lat_count - 1,
            max(
                start_node[0],
                goal_node[0]
            )
            +
            corridor_margin
        )

        min_search_column = max(
            0,
            min(
                start_node[1],
                goal_node[1]
            )
            -
            corridor_margin
        )

        max_search_column = min(
            lon_count - 1,
            max(
                start_node[1],
                goal_node[1]
            )
            +
            corridor_margin
        )

        corridor_rows = (
            max_search_row
            -
            min_search_row
            +
            1
        )

        corridor_columns = (
            max_search_column
            -
            min_search_column
            +
            1
        )

        max_iterations = (
            corridor_rows
            *
            corridor_columns
            *
            4
        )

        iterations = 0

        while (
            open_set
            and
            iterations < max_iterations
        ):

            iterations += 1

            (
                _priority,
                current_cost,
                current
            ) = heapq.heappop(
                open_set
            )

            if current in visited:
                continue

            visited.add(
                current
            )

            if current == goal_node:

                found = True

                break

            current_latitude, current_longitude = (
                node_to_coord(
                    current
                )
            )

            for direction in directions:

                row_change, column_change = (
                    direction
                )

                neighbor = (
                    current[0]
                    +
                    row_change,
                    current[1]
                    +
                    column_change
                )

                # Stay inside the practical corridor.
                if (
                    neighbor[0] < min_search_row
                    or
                    neighbor[0] > max_search_row
                    or
                    neighbor[1] < min_search_column
                    or
                    neighbor[1] > max_search_column
                ):
                    continue

                if neighbor in visited:
                    continue

                if not safe_edge(
                    current,
                    neighbor
                ):
                    continue

                neighbor_latitude, neighbor_longitude = (
                    node_to_coord(
                        neighbor
                    )
                )

                # ====================================================
                # HAZARD-AWARE A* COST
                # ====================================================

                movement_cost = (
                    haversine_km(
                        current_latitude,
                        current_longitude,
                        neighbor_latitude,
                        neighbor_longitude
                    )
                )

                if route_mode == "OPTIMIZED":

                    edge_cost = movement_cost

                else:

                    # SAFEST / HIGH_WAVE:
                    # Penalize routes close to land.
                    shore_distance = (
                        distance_to_shore_km(
                            neighbor_latitude,
                            neighbor_longitude
                        )
                    )

                    if shore_distance < 10.0:

                        shore_penalty = (
                            (10.0 - shore_distance)
                            * 2.5
                        )

                    else:

                        shore_penalty = 0.0

                    edge_cost = (
                        movement_cost
                        +
                        shore_penalty
                    )

                new_cost = (
                    current_cost
                    +
                    edge_cost
                )

                previous_cost = g_score.get(
                    neighbor,
                    float("inf")
                )

                if new_cost < previous_cost:

                    came_from[
                        neighbor
                    ] = current

                    g_score[
                        neighbor
                    ] = new_cost

                    priority = (
                        new_cost
                        +
                        heuristic(
                            neighbor
                        )
                    )

                    heapq.heappush(
                        open_set,
                        (
                            priority,
                            new_cost,
                            neighbor
                        )
                    )

        print(
            "GLOBAL ROUTER SEARCH:"
        )

        print(
            "Iterations:",
            iterations
        )

        print(
            "Visited:",
            len(visited)
        )

        print(
            "Corridor:",
            corridor_rows,
            "x",
            corridor_columns
        )

        if not found:

            print(
                "GLOBAL ROUTER: no ocean-safe route found."
            )

            print(
                "Iterations:",
                iterations
            )

            print(
                "Visited:",
                len(visited)
            )

            print(
                "Grid:",
                lat_count,
                "x",
                lon_count
            )

            print(
                "Open set remaining:",
                len(open_set)
            )

            print(
                "Start node:",
                start_node
            )

            print(
                "Goal node:",
                goal_node
            )

            if g_score:

                closest_node = min(
                    g_score,
                    key=lambda node: heuristic(node)
                )

                closest_lat, closest_lon = (
                    node_to_coord(
                        closest_node
                    )
                )

                print(
                    "Closest explored node:",
                    closest_node
                )

                print(
                    "Closest explored coordinate:",
                    closest_lat,
                    closest_lon
                )

                print(
                    "Distance from closest explored node:",
                    heuristic(
                        closest_node
                    ),
                    "km"
                )

            return None

        # ========================================================
        # RECONSTRUCT PATH
        # ========================================================

        nodes = []

        current = goal_node

        nodes.append(
            current
        )

        while current != start_node:

            current = came_from.get(
                current
            )

            if current is None:

                print(
                    "GLOBAL ROUTER: path reconstruction failed."
                )

                return None

            nodes.append(
                current
            )

        nodes.reverse()

        raw_route = [

            node_to_coord(
                node
            )

            for node in nodes

        ]

        # ========================================================
        # EXACT REAL ENDPOINTS
        # ========================================================

        raw_route[0] = (
            start_lat,
            start_lon
        )

        raw_route[-1] = (
            goal_lat,
            goal_lon
        )

        # ========================================================
        # CREATE TURNING ROUTE
        #
        # Keep important direction changes.
        # Do NOT reduce the whole route to a straight line.
        # ========================================================

        simplified = [

            raw_route[0]

        ]

        if len(raw_route) > 2:

            previous_direction = None

            for index in range(
                1,
                len(raw_route) - 1
            ):

                lat_a, lon_a = (
                    raw_route[
                        index - 1
                    ]
                )

                lat_b, lon_b = (
                    raw_route[
                        index
                    ]
                )

                lat_c, lon_c = (
                    raw_route[
                        index + 1
                    ]
                )

                direction_in = (
                    bearing_degrees(
                        lat_a,
                        lon_a,
                        lat_b,
                        lon_b
                    )
                )

                direction_out = (
                    bearing_degrees(
                        lat_b,
                        lon_b,
                        lat_c,
                        lon_c
                    )
                )

                turn = (
                    angle_difference(
                        direction_in,
                        direction_out
                    )
                )

                if (
                    previous_direction is None
                    or
                    turn >= 8.0
                ):

                    simplified.append(
                        raw_route[
                            index
                        ]
                    )

                    previous_direction = (
                        direction_out
                    )

            simplified.append(
                raw_route[-1]
            )

        else:

            simplified.append(
                raw_route[-1]
            )

        # ========================================================
        # DENSIFY ROUTE
        #
        # This makes the frontend display a smooth sequence of
        # navigation points instead of only a few huge jumps.
        # ========================================================

        final_route = []

        for index in range(
            len(simplified) - 1
        ):

            lat1, lon1 = (
                simplified[
                    index
                ]
            )

            lat2, lon2 = (
                simplified[
                    index + 1
                ]
            )

            distance = (
                haversine_km(
                    lat1,
                    lon1,
                    lat2,
                    lon2
                )
            )

            pieces = max(
                1,
                int(
                    math.ceil(
                        distance / 50.0
                    )
                )
            )

            for piece in range(
                pieces
            ):

                ratio = (
                    piece
                    /
                    pieces
                )

                latitude = (
                    lat1
                    +
                    (
                        lat2 - lat1
                    )
                    *
                    ratio
                )

                longitude = (
                    lon1
                    +
                    (
                        lon2 - lon1
                    )
                    *
                    ratio
                )

                final_route.append(
                    (
                        latitude,
                        longitude
                    )
                )

        final_route.append(
            simplified[-1]
        )

        # ========================================================
        # FINAL SAFETY VALIDATION
        # ========================================================

        for index, point in enumerate(
            final_route
        ):

            latitude, longitude = point

            if point_is_land(
                latitude,
                longitude
            ):

                print(
                    "GLOBAL ROUTER: point validation failed."
                )

                return None

            if index > 0:

                previous_latitude, previous_longitude = (
                    final_route[
                        index - 1
                    ]
                )

                if segment_crosses_land(
                    previous_latitude,
                    previous_longitude,
                    latitude,
                    longitude
                ):

                    print(
                        "GLOBAL ROUTER: segment validation failed."
                    )

                    return None

        # ========================================================
        # FINAL DISTANCE
        # ========================================================

        total_distance = 0.0

        for index in range(
            len(final_route) - 1
        ):

            total_distance += (
                haversine_km(
                    final_route[index][0],
                    final_route[index][1],
                    final_route[index + 1][0],
                    final_route[index + 1][1]
                )
            )

        print()
        print(
            "GLOBAL OCEAN ROUTER SUCCESS"
        )

        print(
            "Raw grid nodes:",
            len(raw_route)
        )

        print(
            "Turn waypoints:",
            len(simplified)
        )

        print(
            "Final route points:",
            len(final_route)
        )

        print(
            "Distance:",
            round(
                total_distance,
                2
            ),
            "km"
        )

        print(
            "=" * 70
        )

        return final_route

    def create_safe_fallback_route():

        print()
        print(
            "PPO route incomplete."
        )

        print(
            "Using GLOBAL OCEAN ROUTER..."
        )

        return create_global_ocean_route(
            current_lat,
            current_lon,
            destination_lat,
            destination_lon
        )

    # ========================================================
    # CHECK CURRENT PPO ROUTE
    # ========================================================

    ppo_complete = (

        reached_destination

        and

        len(route_points) >= 2

    )

    # --------------------------------------------------------
    # Check whether PPO route can safely reach destination
    # --------------------------------------------------------

    if ppo_complete:

        last_point = route_points[-1]

        if segment_crosses_land(

            last_point["latitude"],
            last_point["longitude"],

            destination_lat,
            destination_lon

        ):

            ppo_complete = False

    # ========================================================
    # FALLBACK IF PPO FAILED
    # ========================================================

    if not ppo_complete:

        fallback_route = (
            create_safe_fallback_route()
        )

        if fallback_route is None:

            raise HTTPException(

                status_code=400,

                detail=(
                    "No ocean-safe route "
                    "could be generated "
                    "between the vessel "
                    "and destination."
                )

            )

        # ----------------------------------------------------
        # Convert fallback route into waypoints
        # ----------------------------------------------------

        route_points = []

        for index, point in enumerate(
            fallback_route
        ):

            lat, lon = point

            route_points.append({

                "latitude":
                    float(lat),

                "longitude":
                    float(lon),

                "action":
                    None,

                "turn_angle":
                    None,

                "ppo_turn_angle":
                    None,

                "heading":
                    None,

                "step":
                    index
            })

            route_mode = (
                "SAFEST"
                if hazard == "HIGH_WAVE"
                else
                "OPTIMIZED"
            )

    else:

        # ----------------------------------------------------
        # PPO reached destination
        # Add exact destination only if safe.
        # ----------------------------------------------------

        last_point = route_points[-1]

        if not segment_crosses_land(

            last_point["latitude"],
            last_point["longitude"],

            destination_lat,
            destination_lon

        ):

            route_points.append({

                "latitude":
                    destination_lat,

                "longitude":
                    destination_lon,

                "action":
                    None,

                "turn_angle":
                    None,

                "ppo_turn_angle":
                    None,

                "heading":
                    None,

                "step":
                    len(route_points)
            })

    # ========================================================
    # FINAL LAND CHECK
    # ========================================================

    if route_crosses_land(
        route_points
    ):

        # ----------------------------------------------------
        # One last fallback attempt
        # ----------------------------------------------------

        fallback_route = (
            create_safe_fallback_route()
        )

        if fallback_route is None:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Generated route "
                    "intersects land and "
                    "no safe fallback "
                    "was found."
                )

            )

        route_points = [

            {

                "latitude":
                    float(lat),

                "longitude":
                    float(lon),

                "action":
                    None,

                "turn_angle":
                    None,

                "ppo_turn_angle":
                    None,

                "heading":
                    None,

                "step":
                    index

            }

            for index, (
                lat,
                lon
            )
            in enumerate(
                fallback_route
            )

        ]

        route_mode = (
            "SAFEST"
            if hazard == "HIGH_WAVE"
            else
            "OPTIMIZED"
        )

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

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

    # ========================================================
    # ESTIMATED TIME
    # ========================================================

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

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

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

        "route_validated":
            True,

        "waypoints":
            route_points,

        "waypoint_count":
            len(
                route_points
            )

    }





















