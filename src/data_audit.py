import pandas as pd

import config
from src.data_loading import get_prepared_data


def check_types(df):
    print("\n" + "-" * 70)
    print("CHECK 1: COLUMN TYPES")
    print("-" * 70)

    rows = []
    for column in df.columns:
        rows.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "unique_values": df[column].nunique(),
        })

    types_table = pd.DataFrame(rows)
    print(types_table.to_string(index=False))
    return types_table


def check_missing(df):
    print("\n" + "-" * 70)
    print("CHECK 2: MISSING VALUES")
    print("-" * 70)

    missing_count = df.isna().sum()
    missing_percent = 100 * missing_count / len(df)

    missing_table = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values.round(3),
    })

    missing_table = missing_table.sort_values("missing_count", ascending=False)

    has_missing = missing_table[missing_table["missing_count"] > 0]

    if len(has_missing) == 0:
        print("  No missing values found in any column.")
    else:
        print(has_missing.to_string(index=False))
        print("\n  ACTION NEEDED: decide how to handle each of these in")
        print("  preprocessing, and write the reason in report section 5.")

    return missing_table


def check_duplicates(df):
    print("\n" + "-" * 70)
    print("CHECK 3: DUPLICATE ROWS")
    print("-" * 70)

    n_duplicates = df.duplicated().sum()
    percent = 100 * n_duplicates / len(df)

    print(f"  Exact duplicate rows: {n_duplicates:,}  ({percent:.3f}% of the data)")

    if n_duplicates > 0:
        print("\n  ACTION NEEDED: these should be removed before splitting.")
        print("  If the same row ends up in both train and test, the test")
        print("  score becomes dishonest.")
    else:
        print("  No duplicates found.")

    return n_duplicates


def check_impossible_values(df):
    print("\n" + "-" * 70)
    print("CHECK 4: IMPOSSIBLE VALUES")
    print("-" * 70)

    rules = {
        "DISTANCE": (1, None),
        "CRS_ELAPSED_TIME": (1, None),
        "DEP_HOUR": (0, 23),
        "ARR_HOUR": (0, 23),
        "MONTH": (1, 12),
        "DAY_OF_WEEK": (0, 6),
    }

    findings = []
    for column, (minimum, maximum) in rules.items():
        if column not in df.columns:
            continue

        values = df[column]
        bad_count = 0

        if minimum is not None:
            bad_count += (values < minimum).sum()
        if maximum is not None:
            bad_count += (values > maximum).sum()

        findings.append({
            "column": column,
            "allowed_min": minimum,
            "allowed_max": maximum,
            "actual_min": values.min(),
            "actual_max": values.max(),
            "impossible_values": int(bad_count),
        })

    findings_table = pd.DataFrame(findings)
    print(findings_table.to_string(index=False))

    total_bad = findings_table["impossible_values"].sum()
    if total_bad > 0:
        print(f"\n  ACTION NEEDED: {total_bad:,} impossible values found.")
        print("  Decide whether to remove those rows or correct them,")
        print("  and record the rule you used (requirements Section 4 says")
        print("  we must not edit values without documenting the rule).")
    else:
        print("\n  No impossible values found.")

    return findings_table


def check_label_consistency(df):
    print("\n" + "-" * 70)
    print("CHECK 5: LABEL CONSISTENCY")
    print("-" * 70)

    problems_found = False

    for column in config.CATEGORICAL_FEATURES:
        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str)

        n_raw = values.nunique()
        n_cleaned = values.str.strip().str.upper().nunique()

        print(f"  {column:<12} {n_raw} unique values", end="")

        if n_raw != n_cleaned:
            print(f"  -->  only {n_cleaned} after trimming spaces / fixing case!")
            print(f"                ACTION NEEDED: standardise {column}.")
            problems_found = True
        else:
            print("   (consistent)")

    if not problems_found:
        print("\n  All categorical labels look consistent.")

    return problems_found


def check_outliers(df):
    print("\n" + "-" * 70)
    print("CHECK 6: OUTLIERS (IQR rule)")
    print("-" * 70)

    rows = []
    for column in config.NUMERIC_FEATURES:
        if column not in df.columns:
            continue

        values = df[column].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        n_outliers = ((values < lower_fence) | (values > upper_fence)).sum()

        rows.append({
            "column": column,
            "Q1": round(q1, 1),
            "Q3": round(q3, 1),
            "lower_fence": round(lower_fence, 1),
            "upper_fence": round(upper_fence, 1),
            "n_outliers": int(n_outliers),
            "percent": round(100 * n_outliers / len(values), 2),
        })

    outlier_table = pd.DataFrame(rows)
    print(outlier_table.to_string(index=False))
    print("\n  NOTE: outliers are flagged for INSPECTION, not automatic deletion.")
    print("  A very long flight is unusual but genuine. Only remove a value")
    print("  if you can explain why it is wrong.")

    return outlier_table


def check_class_distribution(df):
    print("\n" + "-" * 70)
    print("CHECK 7: CLASS DISTRIBUTION (the target)")
    print("-" * 70)

    counts = df[config.TARGET_COLUMN].value_counts().sort_index()

    rows = []
    for class_number, count in counts.items():
        rows.append({
            "class_number": class_number,
            "class_name": config.CLASS_NAMES[class_number],
            "count": count,
            "percent": round(100 * count / len(df), 2),
        })

    class_table = pd.DataFrame(rows)
    print(class_table.to_string(index=False))

    ratio = counts.max() / counts.min()
    print(f"\n  Imbalance ratio (largest class / smallest class): {ratio:.1f} to 1")

    if ratio > 3:
        print("\n  THIS IS IMBALANCED DATA. Consequences for our project:")
        print("    - Accuracy alone would be misleading. We report per-class")
        print("      precision, recall and F1 instead (requirements Section 7).")
        print("    - We use class_weight='balanced' so models pay more")
        print("      attention to the rare classes.")
        print("    - We tune using f1_macro, which treats all three classes")
        print("      as equally important.")

    return class_table


def run_full_audit(df, save=True):
    print("\n" + "=" * 70)
    print("DATA AUDIT  (requirement DW-3)")
    print("=" * 70)
    print(f"Auditing {len(df):,} rows and {len(df.columns)} columns.")

    results = {
        "types": check_types(df),
        "missing": check_missing(df),
        "duplicates": check_duplicates(df),
        "impossible": check_impossible_values(df),
        "labels_inconsistent": check_label_consistency(df),
        "outliers": check_outliers(df),
        "class_distribution": check_class_distribution(df),
    }

    if save:
        config.make_folders()
        for name, table in results.items():
            if isinstance(table, pd.DataFrame):
                out = config.METRICS_DIR / f"audit_{name}.csv"
                table.to_csv(out, index=False)
        print(f"\n[SAVED] Audit tables written to: {config.METRICS_DIR}")
        print("        These go into report section 4 (Data audit).")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    config.print_config()
    data, _ = get_prepared_data()
    run_full_audit(data)
