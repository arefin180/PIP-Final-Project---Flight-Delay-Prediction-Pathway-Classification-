import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract_subset import extract_date_range

RAW_INPUT_PATH = "data/raw/2018.csv"
RAW_SUBSET_PATH = "data/raw/flights_subset.csv"
SUBSET_START_DATE = "2018-10-01"
SUBSET_END_DATE = "2018-12-31"
DATE_COLUMN = "FL_DATE"

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/yuanyuwendymu/"
    "airline-delay-and-cancellation-data-2009-2018?select=2018.csv"
)


def check_raw_file_present():
    if not os.path.exists(RAW_INPUT_PATH):
        print("Raw data file not found at:", RAW_INPUT_PATH)
        print()
        print("This dataset is hosted on Kaggle and requires a (free) Kaggle")
        print("account, so this script does not fetch it automatically.")
        print("To get the file:")
        print(f"  1. Go to: {KAGGLE_URL}")
        print("  2. Download '2018.csv' (log in if asked).")
        print(f"  3. Place it at: {RAW_INPUT_PATH}")
        print()
        print("Alternative, if you have a Kaggle API token set up:")
        print("  kaggle datasets download -d yuanyuwendymu/airline-delay-and"
              "-cancellation-data-2009-2018 -f 2018.csv -p data/raw")
        return False
    return True


def main():
    if not check_raw_file_present():
        return

    os.makedirs(os.path.dirname(RAW_SUBSET_PATH), exist_ok=True)

    print(f"Extracting {SUBSET_START_DATE} to {SUBSET_END_DATE} "
          f"from {RAW_INPUT_PATH} ...")

    summary = extract_date_range(
        input_path=RAW_INPUT_PATH,
        output_path=RAW_SUBSET_PATH,
        start_date=SUBSET_START_DATE,
        end_date=SUBSET_END_DATE,
        date_column=DATE_COLUMN,
    )

    print()
    print("Done.")
    print(f"  Rows scanned : {summary['rows_seen']:,}")
    print(f"  Rows kept    : {summary['rows_kept']:,}")
    print(f"  Chunks read  : {summary['chunks_read']}")
    print(f"  Time taken   : {summary['seconds_elapsed']} s")
    print(f"  Saved to     : {RAW_SUBSET_PATH}")
    print()
    print("Record the 'rows kept' figure and today's date in Section 3.1 "
          "of the report (rows in raw file / date downloaded).")


if __name__ == "__main__":
    main()
