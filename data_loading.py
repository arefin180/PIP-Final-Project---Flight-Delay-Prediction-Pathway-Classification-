import os

import pandas as pd

SUBSET_PATH = "data/raw/flights_subset.csv"
PROCESSED_PATH = "data/processed/flights_processed.csv"

LEAKAGE_COLUMNS = [
    "DEP_DELAY",
    "DEP_TIME",
    "TAXI_OUT",
    "TAXI_IN",
    "WHEELS_OFF",
    "WHEELS_ON",
    "ARR_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "CANCELLED",
    "DIVERTED",
    "CANCELLATION_CODE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

CLOCK_COLUMNS_TO_REPLACE = ["CRS_DEP_TIME", "CRS_ARR_TIME"]

OTHER_DROP_COLUMNS = ["OP_CARRIER_FL_NUM", "Unnamed: 27"]

FEATURE_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "MONTH",
    "DAY_OF_WEEK",
    "DEP_HOUR",
    "ARR_HOUR",
]

TARGET_COLUMN = "delay_class"
TARGET_NAME_COLUMN = "delay_class_name"
CLASS_NAMES = {0: "on_time", 1: "minor", 2: "major"}


def _hour_from_clock_int(clock_series):
    clock_series = pd.to_numeric(clock_series, errors="coerce")
    clock_series = clock_series.replace(2400, 0)
    hour = (clock_series // 100).astype("Int64")
    return hour


def _classify_delay(delay_minutes):
    if pd.isna(delay_minutes):
        return pd.NA
    if delay_minutes < 15:
        return 0
    elif delay_minutes <= 60:
        return 1
    else:
        return 2


def load_raw_subset(path=SUBSET_PATH):
    return pd.read_csv(path, low_memory=False)


def build_target_and_features(df):
    df = df.copy()

    rows_before = len(df)
    df = df[df["ARR_DELAY"].notna()].copy()
    rows_after = len(df)

    df[TARGET_COLUMN] = df["ARR_DELAY"].apply(_classify_delay)
    df[TARGET_NAME_COLUMN] = df[TARGET_COLUMN].map(CLASS_NAMES)

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek

    df["DEP_HOUR"] = _hour_from_clock_int(df["CRS_DEP_TIME"])
    df["ARR_HOUR"] = _hour_from_clock_int(df["CRS_ARR_TIME"])

    columns_to_drop = (
        LEAKAGE_COLUMNS
        + CLOCK_COLUMNS_TO_REPLACE
        + OTHER_DROP_COLUMNS
        + ["ARR_DELAY"]
    )
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=columns_to_drop)

    keep_columns = [c for c in FEATURE_COLUMNS if c in df.columns] + [
        TARGET_COLUMN,
        TARGET_NAME_COLUMN,
    ]
    df = df[keep_columns]

    print(f"Rows before dropping missing-outcome rows : {rows_before:,}")
    print(f"Rows after                                : {rows_after:,}")
    print(f"Rows dropped (cancelled/diverted/missing)  : "
          f"{rows_before - rows_after:,}")

    return df


def load_data(path=SUBSET_PATH):
    raw = load_raw_subset(path)
    return build_target_and_features(raw)


if __name__ == "__main__":
    processed = load_data()
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)

    print(f"\nSaved cleaned data to: {PROCESSED_PATH}")
    print(f"Final shape: {processed.shape}")
    print("\nClass balance:")
    print(processed[TARGET_NAME_COLUMN].value_counts())
