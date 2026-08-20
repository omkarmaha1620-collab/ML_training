import os
import gzip
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "XGBoost/data"
OUTPUT_FILE = "XGBoost/data/ndbc_high_wave_dataset.csv"

STATIONS = [
    "42001",
    "42058",
    "44065"
]

YEARS = [
    2023,
    2024,
    2025
]

BASE_URL = (
    "https://www.ndbc.noaa.gov/"
    "data/historical/stdmet/"
)

HIGH_WAVE_THRESHOLD = 3.0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# STANDARD NDBC COLUMNS
# ============================================================

COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "PTDY",
    "TIDE"
]


# ============================================================
# DOWNLOAD DATA
# ============================================================

all_data = []


for station in STATIONS:

    for year in YEARS:

        filename = f"{station}h{year}.txt.gz"

        url = BASE_URL + filename

        local_file = os.path.join(
            OUTPUT_DIR,
            filename
        )

        print("\nDownloading:")
        print(url)

        try:

            urllib.request.urlretrieve(
                url,
                local_file
            )

            print(
                "Downloaded:",
                filename
            )

        except Exception as e:

            print(
                "Skipping:",
                filename
            )

            print(
                "Reason:",
                e
            )

            continue


        # ====================================================
        # READ FILE
        # ====================================================

        try:

            with gzip.open(
                local_file,
                "rt",
                errors="ignore"
            ) as f:

                lines = f.readlines()


            data_lines = [
                line
                for line in lines
                if line.strip()
                and not line.startswith("#")
            ]

            if not data_lines:

                print(
                    "No data rows."
                )

                continue


            df = pd.read_csv(
                StringIO(
                    "".join(data_lines)
                ),
                sep=r"\s+",
                header=None,
                na_values=[
                    "MM",
                    "999",
                    "999.0",
                    "99",
                    "99.0",
                    "9999",
                    "9999.0"
                ]
            )


            # ------------------------------------------------
            # Handle 18 or 19 columns automatically
            # ------------------------------------------------

            if len(df.columns) == 19:

                df.columns = COLUMNS

            elif len(df.columns) == 18:

                df.columns = COLUMNS[:18]

                # Add missing final field
                df["TIDE"] = np.nan

            else:

                print(
                    "Unexpected columns:",
                    len(df.columns)
                )

                continue


            df["station"] = station

            all_data.append(df)

            print(
                "Rows loaded:",
                len(df)
            )


        except Exception as e:

            print(
                "Error processing:",
                filename
            )

            print(
                "Reason:",
                e
            )


# ============================================================
# COMBINE
# ============================================================

if not all_data:

    raise RuntimeError(
        "No NDBC observations were loaded."
    )


df = pd.concat(
    all_data,
    ignore_index=True
)


print("\n" + "=" * 60)
print("RAW NDBC DATA")
print("=" * 60)

print(
    "Rows:",
    len(df)
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "PTDY",
    "TIDE"
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    {
        "year": df["year"],
        "month": df["month"],
        "day": df["day"],
        "hour": df["hour"],
        "minute": df["minute"]
    },
    errors="coerce"
)


# ============================================================
# REMOVE INVALID PHYSICAL VALUES
# ============================================================

df.loc[
    (df["WSPD"] < 0) |
    (df["WSPD"] > 100),
    "WSPD"
] = np.nan


df.loc[
    (df["GST"] < 0) |
    (df["GST"] > 150),
    "GST"
] = np.nan


df.loc[
    (df["WVHT"] < 0) |
    (df["WVHT"] > 20),
    "WVHT"
] = np.nan


df.loc[
    (df["DPD"] < 0) |
    (df["DPD"] > 100),
    "DPD"
] = np.nan


df.loc[
    (df["APD"] < 0) |
    (df["APD"] > 100),
    "APD"
] = np.nan


df.loc[
    (df["PRES"] < 800) |
    (df["PRES"] > 1200),
    "PRES"
] = np.nan


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "station",
        "timestamp"
    ]
).reset_index(
    drop=True
)


# ============================================================
# KEEP ONLY ROWS WITH WAVE HEIGHT
# ============================================================

df = df.dropna(
    subset=["WVHT"]
).reset_index(
    drop=True
)


print(
    "\nRows with valid WVHT:",
    len(df)
)


# ============================================================
# CREATE LAG FEATURES
# ============================================================

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

        df[
            f"{feature}_t-{lag}"
        ] = df.groupby(
            "station"
        )[feature].shift(lag)


# ============================================================
# CREATE FUTURE WAVE HEIGHT
# ============================================================

df["future_WVHT"] = df.groupby(
    "station"
)["WVHT"].shift(-1)


# ============================================================
# DROP ONLY ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(
    subset=["future_WVHT"]
).reset_index(
    drop=True
)


# ============================================================
# FILL FEATURE MISSING VALUES
# ============================================================

feature_columns = []

for lag in [1, 2, 3]:

    for feature in base_features:

        feature_columns.append(
            f"{feature}_t-{lag}"
        )


# Forward/backward fill within station
df[feature_columns] = (
    df.groupby("station")[
        feature_columns
    ]
    .transform(
        lambda x: x.ffill().bfill()
    )
)


# Remaining missing values → global median
for column in feature_columns:

    median_value = df[column].median()

    df[column] = df[column].fillna(
        median_value
    )


# ============================================================
# CREATE TARGET
# ============================================================

df["high_wave"] = (
    df["future_WVHT"]
    >= HIGH_WAVE_THRESHOLD
).astype(int)


# ============================================================
# FINAL DATASET
# ============================================================

final_columns = (
    [
        "station",
        "timestamp"
    ]
    +
    feature_columns
    +
    [
        "future_WVHT",
        "high_wave"
    ]
)


df = df[
    final_columns
]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NDBC DATASET CREATED")
print("=" * 60)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(df)
)

print(
    "Features:",
    len(feature_columns)
)

print("\nStations:")

print(
    df["station"].value_counts()
)

print("\nTarget distribution:")

print(
    df["high_wave"].value_counts()
)

print("\nTarget percentage:")

print(
    (
        df["high_wave"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)

print("\nFuture wave-height statistics:")

print(
    df["future_WVHT"].describe()
)

print("\nDONE!")