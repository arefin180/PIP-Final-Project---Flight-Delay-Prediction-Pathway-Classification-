import sys

import pandas as pd

import config


SOURCE_FILE = config.RAW_DATA_DIR / "2018.csv"

OUTPUT_FILE = config.RAW_DATA_DIR / "flights_subset.csv"

START_DATE = "2018-10-01"
END_DATE = "2018-12-31"

CHUNK_SIZE = 500_000

DATE_COLUMN = "FL_DATE"


def extract(source, output, start_date, end_date, date_column, chunk_size):
    print("=" * 70)
    print("EXTRACTING A DATE RANGE FROM A LARGE CSV")
    print("=" * 70)

    if not source.exists():
        print(f"\nERROR: could not find the source file:\n    {source}\n")
        print("Check that:")
        print("  1. You have downloaded the dataset")
        print("  2. It is in data/raw/")
        print("  3. SOURCE_FILE at the top of this file matches its name")
        sys.exit(1)

    size_mb = source.stat().st_size / (1024 * 1024)
    print(f"\n  Source     : {source.name}  ({size_mb:,.1f} MB)")
    print(f"  Keeping    : {start_date} to {end_date} (inclusive)")
    print(f"  Date column: {date_column}")
    print(f"  Chunk size : {chunk_size:,} rows")
    print(f"  Output     : {output.name}")

    peek = pd.read_csv(source, nrows=5)

    if date_column not in peek.columns:
        print(f"\nERROR: there is no column called '{date_column}' in this file.")
        print(f"\nThe columns it does have are:")
        for column in peek.columns:
            print(f"    {column}")
        print(f"\nFIX: change DATE_COLUMN at the top of this file to whichever")
        print(f"     of those is the flight date.")
        sys.exit(1)

    print(f"\n  Columns in source: {len(peek.columns)}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    kept_chunks = []
    rows_read = 0
    rows_kept = 0
    chunk_number = 0

    print("\n  Reading...")

    for chunk in pd.read_csv(source, chunksize=chunk_size, low_memory=False):
        chunk_number += 1
        rows_read += len(chunk)

        chunk[date_column] = pd.to_datetime(chunk[date_column], errors="coerce")

        in_range = (
            chunk[date_column].notna()
            & (chunk[date_column] >= start)
            & (chunk[date_column] <= end)
        )
        matching = chunk[in_range]

        if len(matching) > 0:
            kept_chunks.append(matching)
            rows_kept += len(matching)

        print(f"    chunk {chunk_number:>3}: read {rows_read:>10,} rows, "
              f"kept {rows_kept:>9,}", end="\r")

    print()

    if rows_kept == 0:
        print(f"\nERROR: no rows matched {start_date} to {end_date}.")
        print("\nCheck that this file actually covers those dates. A file")
        print("named 2018.csv contains 2018 only, so asking for 2019 dates")
        print("will return nothing.")
        sys.exit(1)

    print("\n  Combining and writing...")
    result = pd.concat(kept_chunks, ignore_index=True)

    result = result.sort_values(date_column).reset_index(drop=True)

    result[date_column] = result[date_column].dt.strftime("%Y-%m-%d")

    result.to_csv(output, index=False)

    out_size_mb = output.stat().st_size / (1024 * 1024)

    print("\n" + "-" * 70)
    print("DONE")
    print("-" * 70)
    print(f"  Rows read    : {rows_read:,}")
    print(f"  Rows kept    : {rows_kept:,}  ({100*rows_kept/rows_read:.2f}%)")
    print(f"  File size    : {size_mb:,.1f} MB  ->  {out_size_mb:,.1f} MB")
    print(f"  Saved to     : {output}")

    dates = pd.to_datetime(result[date_column])
    n_days = dates.dt.date.nunique()

    print(f"\n  Date range   : {dates.min().date()} to {dates.max().date()}")
    print(f"  Distinct days: {n_days}")

    n_test_days = int(n_days * config.TEST_SIZE)
    n_train_days = n_days - n_test_days

    print(f"\n  With TEST_SIZE = {config.TEST_SIZE}, your split will be:")
    print(f"    training: {n_train_days} days")
    print(f"    testing : {n_test_days} days")

    if n_test_days < 14:
        print("\n  " + "!" * 62)
        print("  !  WARNING: your test set is only "
              f"{n_test_days} days.")
        print("  !")
        print("  !  With roughly 6% of flights in the 'major' class, a test")
        print("  !  set this short may contain very few of them. Per-class")
        print("  !  scores will be unstable and hard to defend.")
        print("  !")
        print("  !  Consider widening START_DATE to cover more months.")
        print("  " + "!" * 62)

    n_months = pd.to_datetime(result[date_column]).dt.month.nunique()
    if n_months < 2:
        print("\n  NOTE: this subset covers a single month, so MONTH will be")
        print("  a constant. The preprocessing step will warn about this and")
        print("  continue, but the feature carries no information, and report")
        print("  sections 4.2 and 9.5 will have no seasonal pattern to discuss.")

    print("\n" + "-" * 70)
    print("NEXT STEPS")
    print("-" * 70)
    print(f"  1. In config.py set:")
    print(f"         REAL_DATA_FILENAME = \"{output.name}\"")
    print(f"         USE_SAMPLE_DATA = False")
    print(f"  2. Run:  python run_all.py")
    print(f"\n  3. Record in report section 3 that you used a subset,")
    print(f"     which dates, and why. Requirement Section 11 asks for")
    print(f"     computational decisions to be documented, and this is one.")
    print("=" * 70)

    return result


def main():
    extract(
        source=SOURCE_FILE,
        output=OUTPUT_FILE,
        start_date=START_DATE,
        end_date=END_DATE,
        date_column=DATE_COLUMN,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
