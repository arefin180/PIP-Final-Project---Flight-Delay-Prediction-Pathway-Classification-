import time

import pandas as pd


def extract_date_range(input_path, output_path, start_date, end_date,
                        date_column="FL_DATE", chunksize=200_000):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    total_rows_seen = 0
    total_rows_kept = 0
    chunk_number = 0
    header_written = False

    start_time = time.time()

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_number += 1
        total_rows_seen += len(chunk)

        chunk[date_column] = pd.to_datetime(chunk[date_column], errors="coerce")

        mask = (chunk[date_column] >= start_ts) & (chunk[date_column] <= end_ts)
        matching_rows = chunk[mask]

        total_rows_kept += len(matching_rows)

        if len(matching_rows) > 0:
            matching_rows.to_csv(
                output_path,
                mode="w" if not header_written else "a",
                header=not header_written,
                index=False,
            )
            header_written = True

        print(f"  chunk {chunk_number}: scanned {len(chunk):,} rows, "
              f"kept {len(matching_rows):,} rows in range")

    elapsed = time.time() - start_time

    if not header_written:
        print("WARNING: no rows matched the requested date range. "
              "Double-check the date column name and the start/end dates.")
        open(output_path, "w").close()

    return {
        "rows_seen": total_rows_seen,
        "rows_kept": total_rows_kept,
        "chunks_read": chunk_number,
        "seconds_elapsed": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract a date range from a large flight-data CSV."
    )
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("--date-column", default="FL_DATE")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    result = extract_date_range(
        args.input_path,
        args.output_path,
        args.start_date,
        args.end_date,
        date_column=args.date_column,
        chunksize=args.chunksize,
    )
    print(result)
