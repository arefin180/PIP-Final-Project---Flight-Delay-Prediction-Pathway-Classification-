"""
split.py
========
Splits the data into a TRAINING set and a TEST set, using time.

THE ONE IDEA YOU MUST UNDERSTAND FROM THIS FILE:

    We do NOT split randomly.

    Flight data is ordered in time. If we split randomly, some flights from
    December would land in the training set and some from January would land
    in the test set. The model would then be learning from the future to
    predict the past.

    That is not how the model would ever be used in real life. In real life
    you know the past and you are guessing about tomorrow. So we split by
    DATE: the earliest days become training data, the latest days become
    test data.

    This is what requirement DW-5 means by "time-aware splitting when the
    data requires it".

THE SECOND IDEA:

    The test set is SEALED. From the moment this file runs, nobody looks at
    the test set again until the single final evaluation at the very end.
    Requirement Section 14 calls repeatedly checking the test set "data
    leakage" and it costs marks.

WORKFLOW STAGE COVERED: DW-5
OWNER: Member 2

HOW TO RUN:
    python -m src.split
"""

import pandas as pd

import config
from src.data_loading import get_prepared_data


def remove_duplicates_before_split(df):
    """
    Deletes exact duplicate rows.

    THIS MUST HAPPEN BEFORE THE SPLIT. If a duplicated flight ends up with
    one copy in training and one copy in test, the model gets tested on a
    row it has already memorised, and our test score becomes a lie.
    """
    print("\n[CLEAN] Removing duplicate rows before splitting...")

    n_before = len(df)
    df = df.drop_duplicates().copy()
    n_removed = n_before - len(df)

    print(f"    Removed {n_removed:,} duplicate rows.")
    print(f"    Rows remaining: {len(df):,}")
    print("    (Done BEFORE the split so no flight can appear on both sides.)")

    return df


def time_aware_split(df, test_size=None):
    """
    Splits the table by date: oldest rows for training, newest for testing.

    HOW IT WORKS, STEP BY STEP:
        1. Get the list of every unique DATE in the data.
        2. Sort those dates from earliest to latest.
        3. Work out the cut-off date, so that the last `test_size` share
           of DAYS falls after it.
        4. Rows on or before the cut-off  -> training set
           Rows after the cut-off         -> test set

    WHY WE SPLIT ON DAYS, NOT ON ROWS:
        If we split on rows, a single day could be chopped in half, with
        some of its flights in training and some in test. Flights on the
        same day share weather and congestion, so that would leak
        information across the boundary. Splitting whole days avoids it.

    Parameters
    ----------
    df : pandas.DataFrame
        The prepared table.
    test_size : float, optional
        Share of the time period to hold back. Defaults to config.TEST_SIZE.

    Returns
    -------
    (train_df, test_df, split_info)
    """
    if test_size is None:
        test_size = config.TEST_SIZE

    print("\n" + "=" * 70)
    print("TIME-AWARE TRAIN / TEST SPLIT  (requirement DW-5)")
    print("=" * 70)

    date_col = config.DATE_COLUMN

    df[date_col] = pd.to_datetime(df[date_col])

    unique_dates = sorted(df[date_col].unique())
    n_days = len(unique_dates)
    print(f"\n  The data covers {n_days} distinct days.")
    print(f"  Earliest day: {pd.Timestamp(unique_dates[0]).date()}")
    print(f"  Latest day  : {pd.Timestamp(unique_dates[-1]).date()}")

    n_train_days = int(n_days * (1 - test_size))

    if n_train_days < 1 or n_train_days >= n_days:
        raise ValueError(
            f"Cannot split {n_days} days with test_size={test_size}. "
            f"There is not enough data. Use a longer date range."
        )

    cutoff_date = pd.Timestamp(unique_dates[n_train_days - 1])
    print(f"\n  Cut-off date: {cutoff_date.date()}")
    print(f"    Training = the first  {n_train_days} days (on or before the cut-off)")
    print(f"    Testing  = the last   {n_days - n_train_days} days (after the cut-off)")

    train_df = df[df[date_col] <= cutoff_date].copy()
    test_df = df[df[date_col] > cutoff_date].copy()

    print(f"\n  Training rows: {len(train_df):,}  ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Test rows    : {len(test_df):,}  ({100*len(test_df)/len(df):.1f}%)")


    print("\n  --- Verifying the split ---")

    train_dates = set(train_df[date_col].unique())
    test_dates = set(test_df[date_col].unique())
    overlap = train_dates & test_dates      

    if len(overlap) == 0:
        print("    PASS: no date appears in both training and test.")
    else:
        raise AssertionError(
            f"SPLIT FAILED: {len(overlap)} dates appear on both sides. "
            f"This would be data leakage. Do not continue."
        )

    latest_train = train_df[date_col].max()
    earliest_test = test_df[date_col].min()

    if latest_train < earliest_test:
        print(f"    PASS: every training day ({latest_train.date()} at the latest)")
        print(f"          comes before every test day ({earliest_test.date()} at the earliest).")
    else:
        raise AssertionError("SPLIT FAILED: training dates are not all before test dates.")

    print("\n  --- Class balance on each side ---")
    print(f"    {'class':<10} {'train %':>10} {'test %':>10}")

    train_percent = train_df[config.TARGET_COLUMN].value_counts(normalize=True).sort_index()
    test_percent = test_df[config.TARGET_COLUMN].value_counts(normalize=True).sort_index()

    for class_number, class_name in enumerate(config.CLASS_NAMES):
        tr = 100 * train_percent.get(class_number, 0)
        te = 100 * test_percent.get(class_number, 0)
        print(f"    {class_name:<10} {tr:>9.2f}% {te:>9.2f}%")

    print("\n    Any difference here is REAL, not a bug: delays are seasonal.")
    print("    Discuss it in report section 9 as distribution shift.")

    split_info = {
        "n_days_total": n_days,
        "n_days_train": n_train_days,
        "n_days_test": n_days - n_train_days,
        "cutoff_date": str(cutoff_date.date()),
        "n_rows_train": len(train_df),
        "n_rows_test": len(test_df),
        "test_size_setting": test_size,
    }

    print("\n" + "=" * 70)
    print("  >>> THE TEST SET IS NOW SEALED. <<<")
    print("  Do not look at it, score against it, or tune using it")
    print("  until the single final evaluation in evaluate.py.")
    print("=" * 70)

    return train_df, test_df, split_info


def get_train_test(df=None):
    """
    Convenience function: clean the duplicates, then split.

    This is the function other files should call.
    """
    if df is None:
        df, _ = get_prepared_data()

    df = remove_duplicates_before_split(df)
    return time_aware_split(df)


if __name__ == "__main__":
    config.print_config()

    data, _ = get_prepared_data()
    train, test, info = get_train_test(data)

    print("\nSplit summary:")
    for key, value in info.items():
        print(f"  {key}: {value}")
