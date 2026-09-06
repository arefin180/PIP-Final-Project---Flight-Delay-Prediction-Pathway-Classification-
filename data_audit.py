import os

import pandas as pd

from src.data_loading import SUBSET_PATH, TARGET_NAME_COLUMN, load_data, load_raw_subset

METRICS_DIR = "results/metrics"


def check_missing_values(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })
    return report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def check_duplicates(df):
    duplicate_count = int(df.duplicated().sum())
    duplicate_percent = round(duplicate_count / len(df) * 100, 3)
    return duplicate_count, duplicate_percent


def check_impossible_values(df):
    rules = {}

    if "DISTANCE" in df.columns:
        rules["DISTANCE >= 1"] = int((df["DISTANCE"] < 1).sum())
    if "CRS_ELAPSED_TIME" in df.columns:
        rules["CRS_ELAPSED_TIME >= 1"] = int((df["CRS_ELAPSED_TIME"] < 1).sum())
    if "MONTH" in df.columns:
        rules["MONTH in 1..12"] = int((~df["MONTH"].between(1, 12)).sum())
    for hour_col in ["DEP_HOUR", "ARR_HOUR"]:
        if hour_col in df.columns:
            rules[f"{hour_col} in 0..23"] = int((~df[hour_col].between(0, 23)).sum())

    return pd.DataFrame({"rule": list(rules.keys()), "violations": list(rules.values())})


def flag_outliers_iqr(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound)
    count = int(is_outlier.sum())
    percent = round(count / len(df) * 100, 2)
    return count, percent, lower_bound, upper_bound


def class_distribution(processed_df):
    counts = processed_df[TARGET_NAME_COLUMN].value_counts()
    percents = (counts / len(processed_df) * 100).round(2)
    table = pd.DataFrame({"count": counts, "percent_of_dataset": percents})
    table.index.name = "class"
    return table


def run_full_audit(subset_path=SUBSET_PATH):
    print("Loading raw subset for auditing...")
    raw_df = load_raw_subset(subset_path)
    print(f"Raw subset shape: {raw_df.shape}\n")

    os.makedirs(METRICS_DIR, exist_ok=True)

    print("=== Missing values ===")
    missing_report = check_missing_values(raw_df)
    print(missing_report if len(missing_report) else "No missing values found.")
    missing_report.to_csv(os.path.join(METRICS_DIR, "audit_missing_values.csv"))

    print("\n=== Duplicate rows ===")
    dup_count, dup_percent = check_duplicates(raw_df)
    print(f"Duplicate rows: {dup_count:,} ({dup_percent}% of dataset)")

    print("\nBuilding processed features to check derived columns...")
    processed_df = load_data(subset_path)

    print("\n=== Impossible values ===")
    impossible_report = check_impossible_values(processed_df)
    print(impossible_report)
    impossible_report.to_csv(
        os.path.join(METRICS_DIR, "audit_impossible_values.csv"), index=False
    )

    print("\n=== Outliers (flagged, not removed) ===")
    for column in ["DISTANCE", "CRS_ELAPSED_TIME"]:
        if column in processed_df.columns:
            count, percent, lower, upper = flag_outliers_iqr(processed_df, column)
            print(f"{column}: {count:,} flagged ({percent}%), "
                  f"expected range [{lower:.1f}, {upper:.1f}]")

    print("\n=== Class distribution (after cleaning) ===")
    dist = class_distribution(processed_df)
    print(dist)
    dist.to_csv(os.path.join(METRICS_DIR, "audit_class_distribution.csv"))

    largest = dist["count"].max()
    smallest = dist["count"].min()
    print(f"\nImbalance ratio (largest : smallest) = {largest / smallest:.1f} : 1")

    print(f"\nAll audit results saved to: {METRICS_DIR}/")
    print("Copy the numbers above into Section 3.5 and Section 4.1 of the report.")


if __name__ == "__main__":
    run_full_audit()
