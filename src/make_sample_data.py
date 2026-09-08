# Creates a SMALL FAKE dataset so we can test that our code runs.
#
#This file does NOT create the project dataset. The data it produces is INVENTED. It is not real flight data.
#Its ONLY purpose is to let us check that every script runs without
#     crashing, before the real dataset is downloaded.
#
#NO number, table, plot, or conclusion in the report will go from
#     this file. The project requirements Section 14. 
#     
# TO RUN: [] python -m src.make_sample_data
#
# Member 4 AREFIN(tooling). Used by all members while developing.

import numpy as np
import pandas as pd

import config


def make_sample_data(n_rows=5000, seed=config.RANDOM_STATE):
    # Builds a fake flights table that looks like a real BTS flight file.
    #
    # We deliberately build in some realistic patterns (evening flights are
    # later, some airlines are worse, winter is worse)
    #
    # Parameters
    # 
    # n_rows : int
    #     How many fake flights to create.
    # seed : int
    #     The random seed. Using a fixed seed means this file produces the
    #     SAME fake data every time, so our testing is repeatable.
    #
    # Returns
    # 
    # pandas.DataFrame
    #  A table with the same column names as a real BTS flight file.

    # A "random number generator" with a fixed starting point.
    # Same seed in, same numbers out, every single time.
    rng = np.random.default_rng(seed)

    
    #STEP 1: Make up some airlines, airports, and dates
    
    carriers = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
    airports = [
        "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA",
        "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "MIA",
    ]

    # One year of dates, then pick n_rows of them at random.
    all_dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    flight_dates = rng.choice(all_dates, size=n_rows)

    
    #STEP 2: Picking an airline and a route for each flight
 #p= gives each airline a different share of flights, so the data is
    # imbalanced like real life.
    carrier = rng.choice(
        carriers,
        size=n_rows,
        p=[0.18, 0.16, 0.15, 0.20, 0.09, 0.08, 0.08, 0.06],
    )
    origin = rng.choice(airports, size=n_rows)
    dest = rng.choice(airports, size=n_rows)

    #A flight cannot go from an airport to itself, where that  happened, shift the destination to the next airport in the list.
    same = origin == dest
    dest[same] = [
        airports[(airports.index(a) + 1) % len(airports)] for a in origin[same]
    ]

    
    #STEP 3: Scheduled times
   
# Real BTS times are stored as an integer like 1435, meaning 14:35.
    
    dep_hour = rng.integers(5, 23, size=n_rows)      #flights from 05:00 to 22:00
    dep_minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=n_rows)
    crs_dep_time = dep_hour * 100 + dep_minute

    distance = rng.integers(150, 2800, size=n_rows)

    #journey time grows with distance, plus fixed time for taxiing.
    crs_elapsed = (distance / 8.0 + 40 + rng.normal(0, 10, n_rows)).round()
    crs_elapsed = np.clip(crs_elapsed, 40, 400)

    #Scheduled arrival = scheduled departure + journey length.
    arr_total_minutes = (dep_hour * 60 + dep_minute + crs_elapsed) % (24 * 60)
    arr_hour = (arr_total_minutes // 60).astype(int)
    arr_minute = (arr_total_minutes % 60).astype(int)
    crs_arr_time = arr_hour * 100 + arr_minute

    
    # STEP 4: Invent a delay, built from realistic influences
    
    # We start every flight comfortably early, and then add effects that push some of them into lateness.

    #THE NUMBERS BELOW, they are tuned so the finished sample has
    # roughly 80% on_time / 13% minor / 7% major, which is close to the real
    # class balance in US flight data.
    delay = rng.normal(-6, 10, n_rows)

    # EFFECT 1: Delays build up through the day. A 22:00 flight inherits
    # the day's accumulated lateness; a 06:00 flight starts fresh.
    delay += (dep_hour - 5) * 0.55

    # EFFECT 2: Some airlines are simply worse at punctuality.
    carrier_penalty = {
        "AA": 1.5, "DL": -2, "UA": 2, "WN": 0.5,
        "B6": 3.5, "AS": -2.5, "NK": 4.5, "F9": 4,
    }
    delay += np.array([carrier_penalty[c] for c in carrier])

    # EFFECT 3: Winter and summer are worse (snow, storms, holiday traffic).
    month = pd.Series(flight_dates).dt.month.to_numpy()
    month_penalty = np.where(np.isin(month, [12, 1, 2]), 3.0,
                     np.where(np.isin(month, [6, 7, 8]), 2.0, 0))
    delay += month_penalty

    # EFFECT 4: A few very congested airports.
    busy_airports = ["ORD", "EWR", "JFK", "SFO"]
    delay += np.where(np.isin(origin, busy_airports), 2.5, 0)

    # EFFECT 5: Random bad luck. About 13% of flights hit a serious problem
    # (weather, mechanical, missing crew). This creates our "major" class.
#An exponential distribution gives mostly smallish extra delays with
# an occasional very large one, which is how real delays behave.
    bad_luck = rng.random(n_rows) < 0.13
    delay += np.where(bad_luck, rng.exponential(75, n_rows), 0)

    arr_delay = delay.round(0)

    # STEP 5: Add the columns we are FORBIDDEN from using
# We include these on purpose. A real BTS file contains them, so our
 # loading code must prove it can spot and remove them.
    dep_delay = (arr_delay - rng.normal(3, 8, n_rows)).round(0)
    taxi_out = rng.integers(8, 35, size=n_rows)
    air_time = (crs_elapsed - taxi_out - rng.integers(4, 12, size=n_rows)).round()

 
    #STEP 6: Added realistic data problems for the audit step to find
    
    df = pd.DataFrame({
        "FL_DATE": pd.Series(flight_dates).dt.strftime("%Y-%m-%d"),
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": crs_dep_time,
        "DEP_TIME": crs_dep_time,          # FORBIDDEN (actual departure)
        "DEP_DELAY": dep_delay,            # FORBIDDEN (the leak)
        "TAXI_OUT": taxi_out,              # FORBIDDEN
        "CRS_ARR_TIME": crs_arr_time,
        "ARR_TIME": crs_arr_time,          # FORBIDDEN (actual arrival)
        "ARR_DELAY": arr_delay,            # our TARGET comes from this
        "CANCELLED": 0,                    # FORBIDDEN (outcome)
        "DIVERTED": 0,                     # FORBIDDEN (outcome)
        "CRS_ELAPSED_TIME": crs_elapsed,
        "AIR_TIME": air_time,              # FORBIDDEN
        "DISTANCE": distance,
    })

    #PROBLEM A: missing values. Real data always has gaps, and our audit
    # and cleaning code needs something to actually find.
    missing_idx = rng.choice(df.index, size=int(n_rows * 0.02), replace=False)
    df.loc[missing_idx, "ARR_DELAY"] = np.nan

    missing_dist = rng.choice(df.index, size=int(n_rows * 0.01), replace=False)
    df.loc[missing_dist, "DISTANCE"] = np.nan

    #PROBLEM B: duplicate rows. Real exports often repeat rows.
    dup_rows = df.sample(n=int(n_rows * 0.005), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    #Shuffling needed, so the duplicates are not all sitting at the bottom.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def main():
    """Creates the sample file and a warning note beside it."""
    config.make_folders()

    print("Creating SAMPLE data (fake, for testing code only)...")
    df = make_sample_data()

    out_path = config.SAMPLE_DATA_DIR / config.SAMPLE_DATA_FILENAME
    df.to_csv(out_path, index=False)

    #A warning note saved next to the file.
    note = (
        "SAMPLE DATA -- NOT THE PROJECT DATASET\n"
        "======================================\n\n"
        "The file sample_flights.csv in this folder is INVENTED data,\n"
        "created by src/make_sample_data.py.\n\n"
        "It exists to check that our code runs without\n"
        "crashing before the real dataset is available.\n\n"
        "The real dataset belongs in data/raw/, and config.USE_SAMPLE_DATA\n"
        "must be set to False before producing any result for the report.\n"
    )
    (config.SAMPLE_DATA_DIR / "SAMPLE_DATA_README.txt").write_text(note)

    print(f"  Saved {len(df):,} fake rows to: {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print("  Also wrote SAMPLE_DATA_README.txt warning note.")
    print()
    print("  REMINDER: this data is fake. Testing only.")


if __name__ == "__main__":
    main()
