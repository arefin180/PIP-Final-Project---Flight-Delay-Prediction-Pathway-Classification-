import numpy as np
import pandas as pd

import config


def load_raw_data(path=None):
    if path is None:
        path = config.get_data_path()

    print(f"[LOAD] Reading file: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\n\nCould not find the data file at:\n    {path}\n\n"
            f"FIX THIS BY EITHER:\n"
            f"  (a) Running  python -m src.make_sample_data  to create test data, or\n"
            f"  (b) Putting the real dataset in {config.RAW_DATA_DIR} and\n"
            f"      setting USE_SAMPLE_DATA = False in config.py\n"
        )

    df = pd.read_csv(path)

    print(f"[LOAD] Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def drop_forbidden_columns(df):
    print("\n[LEAKAGE GUARD] Removing columns that are not known before departure...")

    found = [c for c in config.FORBIDDEN_COLUMNS if c in df.columns]
    not_found = [c for c in config.FORBIDDEN_COLUMNS if c not in df.columns]

    df = df.drop(columns=found)

    for col in found:
        print(f"    REMOVED: {col}")
    if not_found:
        print(f"    (not present in this file, nothing to remove: {not_found})")

    print(f"[LEAKAGE GUARD] Removed {len(found)} leakage columns.")
    return df, found


def create_target(df):
    print("\n[TARGET] Building the three delay classes...")

    delay_col = config.SOURCE_DELAY_COLUMN

    if delay_col not in df.columns:
        raise KeyError(
            f"The column '{delay_col}' is missing from the data file, so we "
            f"cannot build the target. Check the column names in your file "
            f"and update SOURCE_DELAY_COLUMN in config.py if they differ."
        )

    n_before = len(df)
    df = df[df[delay_col].notna()].copy()
    n_dropped = n_before - len(df)
    print(f"    Dropped {n_dropped:,} rows with no {delay_col} value "
          f"(we cannot learn from a flight whose outcome is unknown).")

    conditions = [
        df[delay_col] < config.MINOR_DELAY_THRESHOLD,
        df[delay_col] <= config.MAJOR_DELAY_THRESHOLD,
    ]
    choices = [0, 1]
    df[config.TARGET_COLUMN] = np.select(conditions, choices, default=2)

    print("\n    Class distribution:")
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    for class_number, count in counts.items():
        name = config.CLASS_NAMES[class_number]
        percent = 100 * count / len(df)
        print(f"      {class_number} {name:<9} {count:>8,}  ({percent:5.2f}%)")

    print("\n    NOTE: these classes are imbalanced. Most flights are on time.")
    print("    That is why we use f1_macro and class_weight='balanced' later,")
    print("    instead of plain accuracy.")

    return df


def engineer_features(df):
    print("\n[FEATURES] Creating new features from the schedule...")

    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")

    n_bad_dates = df[config.DATE_COLUMN].isna().sum()
    if n_bad_dates > 0:
        print(f"    Dropping {n_bad_dates:,} rows with an unreadable date.")
        df = df[df[config.DATE_COLUMN].notna()].copy()

    df["MONTH"] = df[config.DATE_COLUMN].dt.month
    df["DAY_OF_WEEK"] = df[config.DATE_COLUMN].dt.dayofweek
    print("    Created MONTH and DAY_OF_WEEK from FL_DATE.")

    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100) % 24
    df["ARR_HOUR"] = (df["CRS_ARR_TIME"] // 100) % 24
    print("    Created DEP_HOUR and ARR_HOUR from the scheduled times.")

    return df


def apply_row_cap(df):
    if config.MAX_ROWS is None or len(df) <= config.MAX_ROWS:
        print(f"\n[ROW CAP] No cap applied. Using all {len(df):,} rows.")
        return df

    print(f"\n[ROW CAP] Dataset has {len(df):,} rows, cap is {config.MAX_ROWS:,}.")

    n_days_before = df[config.DATE_COLUMN].dt.date.nunique()

    if config.SAMPLING_METHOD == "stratified_by_day":
        keep_fraction = config.MAX_ROWS / len(df)

        df = (
            df.groupby(df[config.DATE_COLUMN].dt.date, group_keys=False)
              .sample(frac=keep_fraction, random_state=config.RANDOM_STATE)
              .sort_values(config.DATE_COLUMN)
              .reset_index(drop=True)
              .copy()
        )

        print(f"    Kept {keep_fraction:.1%} of the flights on EVERY calendar day.")
        print(f"    Seed = {config.RANDOM_STATE}, so this selection is reproducible.")

    elif config.SAMPLING_METHOD == "oldest":
        df = df.sort_values(config.DATE_COLUMN).head(config.MAX_ROWS).copy()
        print(f"    Kept the OLDEST {config.MAX_ROWS:,} rows to preserve time order.")

    elif config.SAMPLING_METHOD == "random":
        df = df.sample(n=config.MAX_ROWS, random_state=config.RANDOM_STATE).copy()
        print(f"    Took a flat RANDOM sample of {config.MAX_ROWS:,} rows "
              f"(seed={config.RANDOM_STATE}).")

    else:
        raise ValueError(
            f"\n\nconfig.SAMPLING_METHOD is set to '{config.SAMPLING_METHOD}', "
            f"which is not a method this code knows.\n"
            f"Valid options: 'stratified_by_day', 'oldest', 'random'.\n"
        )

    n_days_after = df[config.DATE_COLUMN].dt.date.nunique()

    print(f"    Rows now      : {len(df):,}")
    print(f"    Date range now: {df[config.DATE_COLUMN].min().date()} "
          f"to {df[config.DATE_COLUMN].max().date()}")
    print(f"    Distinct days : {n_days_after} (was {n_days_before})")

    if n_days_after < n_days_before:
        print(f"    >>> WARNING: the cap removed {n_days_before - n_days_after} "
              f"whole day(s) from the period. <<<")
        print("    >>> Months and seasonality are now only partly covered.  <<<")

    print("    >>> Remember to state this row cap in the report as a")
    print("    >>> computational decision (requirements Section 11). <<<")
    return df


def get_prepared_data(path=None):
    print("\n" + "=" * 70)
    print("DATA LOADING AND PREPARATION")
    print("=" * 70)

    df = load_raw_data(path)
    n_original = len(df)

    df, removed_columns = drop_forbidden_columns(df)
    df = create_target(df)
    df = engineer_features(df)
    df = apply_row_cap(df)

    info = {
        "rows_in_file": n_original,
        "rows_after_preparation": len(df),
        "leakage_columns_removed": removed_columns,
        "columns_remaining": list(df.columns),
    }

    print(f"\n[DONE] Prepared table: {len(df):,} rows, {len(df.columns)} columns.")
    print("=" * 70)

    return df, info


if __name__ == "__main__":
    config.print_config()
    config.make_folders()

    data, facts = get_prepared_data()

    print("\nFirst 5 rows of the prepared table:")
    print(data.head())

    print("\nColumns now available to the model:")
    print(f"  Numeric    : {config.NUMERIC_FEATURES}")
    print(f"  Categorical: {config.CATEGORICAL_FEATURES}")
    print(f"  Target     : {config.TARGET_COLUMN}")
