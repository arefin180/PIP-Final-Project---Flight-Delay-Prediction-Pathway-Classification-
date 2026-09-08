# CENTRAL SETTINGS FILE FOR THE WHOLE PROJECT.


#Project requirements (Section 11, "Reproducibility") say we must
#     centralise paths, seeds and parameters, and that our code must NOT depend
#     on undocumented local paths. So every setting lives in one place.

# HOW TO USE IT:
#     Any other file can read a setting by writing:
#
#         import config
#         print(config.RANDOM_STATE)
#
#     If someone wants to change something like use more rows of data, need to
#     change it HERE, not inside other files.


#Member 4 AREFIN (reproducibility), For ALL members read

from pathlib import Path


# SECTION 1: FOLDER PATHS
#
# We build every path starting from the folder this file is in, this means the project works on any computer, in any folder, with no editing.

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"              #the real dataset
SAMPLE_DATA_DIR = DATA_DIR / "sample"        # small fake data, for code testing
PROCESSED_DATA_DIR = DATA_DIR / "processed" 

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"        # all .png plots
METRICS_DIR = RESULTS_DIR / "metrics"        # all .csv metric 

REPORT_DIR = PROJECT_ROOT / "report"


#
# SECTION 2: WHICH DATA FILE TO USE
#
# USE_SAMPLE_DATA = True  -> run on the small fake file (fast, for testing code)
# USE_SAMPLE_DATA = False -> run on the real dataset in data/raw/
#
#Sample file is ONLY for checking that the code runs.
# No result, number, or conclusion in the report will go from it.

USE_SAMPLE_DATA = False


# This is the October-December 2018 subset produced by:
#     python -m src.extract_subset
# from the full-year file data/raw/2018.csv 
# The subset has 1,796,121 rows covering 2018-10-01 to 2018-12-31 (92 days).
REAL_DATA_FILENAME = "flights_subset.csv"

SAMPLE_DATA_FILENAME = "sample_flights.csv"


def get_data_path():
    """
    Returns the full path of the data file we should load.

    We use a function (instead of a plain variable) so that the choice
    between sample and real data is made in ONE place only.
    """
    if USE_SAMPLE_DATA:
        return SAMPLE_DATA_DIR / SAMPLE_DATA_FILENAME
    else:
        return RAW_DATA_DIR / REAL_DATA_FILENAME



#RANDOM SEED

# Requirements Section 11, Using the same number everywhere means anyone re-running our code
# gets exactly the same numbers we did.

RANDOM_STATE = 42



#HOW many ROWS TO USE

# Real flight datasets can have millions of rows. K-Nearest Neighbours and
# Random Forest become extremely slow at that size, and we must be able to run
# the whole workflow live during the demo (Section 18).
#
# So we cap the number of rows. This is a COMPUTE decision, and we 
# documented it in the report as, " we didn't hide data"
#
# Set MAX_ROWS = None to use every row (only do this if your computer can cope).


# DEMO MODE

# Section 18 asks us to run the workflow LIVE, full run takes about
# 24 minutes, almost all of it K-Nearest Neighbours, which is too long to sit through in a demonstration.
#
# DEMO_MODE = True shrinks the dataset so the same code path finishes in a
# couple of minutes. NOTHING else changes: same models, split logic, seed,checks.
#
#   DEMO_MODE = False  -> 600,000 rows. THIS PRODUCED EVERY NUMBER IN THE
#                         REPORT AND IN results/.
#   DEMO_MODE = True   -> 60,000 rows. For running live, in front of people.
#
#DEMO RUN'S NUMBERS ISNT IN REPORT, as they come from a tenth of the data, results won't match.
# The workflow prints a warning when this is on, so demo run cannot be
# mistaken for the real one.
DEMO_MODE = False

MAX_ROWS = 60_000 if DEMO_MODE else 600_000

# HOW the rows are chosen when the cap reaches.
#
# Our data is 1,796,121 flights across the 92 days of October, November and
# December 2018. The cap therefore keeps roughly one flight in three.
#
# "stratified_by_day" keeps the SAME SHARE of flights from EVERY calendar
# day, so all 92 days survive the cut.

#"oldest": keeping the earliest 600,000 rows would stop somewhere
# in early November. We would lose December entirely, the month with the
# holiday traffic and the winter weather, and the month our time-aware
# split is supposed to test on. MONTH would collapse to one or two values
# and stop carrying information, and report sections 4.2 and 9.5 would have
# no monthly pattern and no distribution shift to discuss.
#
# DID NOT use "random": a flat random draw is close, but it does not
# guarantee that every single day survives, so the time series could end up with gaps.
#
# Options: "stratified_by_day", "oldest", "random"
SAMPLING_METHOD = "stratified_by_day"



#WHAT WE ARE PREDICTING

# We are doing MULTI-CLASS classification with three classes, based on how
# many minutes late the flight ARRIVED (the ARR_DELAY column).
#
#   Class 0 "on_time"  : arrived less than 15 minutes late
#   Class 1 "minor"    : arrived 15 to 60 minutes late
#   Class 2 "major"    : arrived more than 60 minutes late
#
# WHY 15 MINUTES? It is the official US Department of Transportation
# definition of a "delayed" flight.

TARGET_COLUMN = "DELAY_CLASS"       # the new column our code will create
SOURCE_DELAY_COLUMN = "ARR_DELAY"   # the raw column we create it from

#The boundaries, minutes.
MINOR_DELAY_THRESHOLD = 15
MAJOR_DELAY_THRESHOLD = 60


CLASS_NAMES = ["on_time", "minor", "major"]



# SECTION 6: FEATURES WE ARE ALLOWED TO USE
# We decided our prediction moment is BEFORE THE PLANE DEPARTS
# (imagining a passenger checking, the day before, whether their flight
# will be late).
#
# That means we may ONLY use information that is genuinely known at that
# moment: the schedule. Anything that is only known AFTER the plane has
# left is called "leakage", using it would make our model look good but
# wouldn't be practical prediction, fills requirement FR-2.

# ---- Columns we ARE allowed to use (all known from the schedule) ----
SCHEDULE_COLUMNS = [
    "FL_DATE",           # date of the flight
    "OP_CARRIER",        # airline code, e.g. "AA"
    "ORIGIN",            # departure airport code, e.g. "JFK"
    "DEST",              # arrival airport code, e.g. "LAX"
    "CRS_DEP_TIME",      # SCHEDULED departure time
    "CRS_ARR_TIME",      # SCHEDULED arrival time
    "CRS_ELAPSED_TIME",  # SCHEDULED journey length in minutes
    "DISTANCE",          # distance in miles
]

#Columns we were NOT allowed to use (leakage)
# Every one of these is only known once the flight has already departed, or directly answers
#
# Our code deletes these automatically, so nobody can use one by accident.
FORBIDDEN_COLUMNS = [
    #Known only after departure
    "DEP_TIME",              # the ACTUAL departure time
    "DEP_DELAY",             # how late it ACTUALLY left
    "TAXI_OUT",              # minutes taxiing before take-off
    "WHEELS_OFF",            # actual take-off time
    "WHEELS_ON",             # actual landing time
    "TAXI_IN",               # minutes taxiing after landing
    "ARR_TIME",              # the ACTUAL arrival time
    "ACTUAL_ELAPSED_TIME",   # how long the flight ACTUALLY took
    "AIR_TIME",              # actual time in the air

    #OUTput
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",

    #Breakdowns of WHY it was late: only exist if it WAS late
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

#Features we CREATE ourselves from the schedule columns 
# These are all derived from FL_DATE and the scheduled times, so they are
# still known before departure.
ENGINEERED_NUMERIC_FEATURES = [
    "MONTH",           # 1-12,  from FL_DATE
    "DAY_OF_WEEK",     # 0=Monday ... 6=Sunday, from FL_DATE
    "DEP_HOUR",        # 0-23,  from CRS_DEP_TIME
    "ARR_HOUR",        # 0-23,  from CRS_ARR_TIME
]

# The lists the model actually receives:
NUMERIC_FEATURES = [
    "DISTANCE",
    "CRS_ELAPSED_TIME",
] + ENGINEERED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
]

#Columns we dropped once we have engineered better versions of it,
#
# CRS_DEP_TIME is stored as an integer like 1435, meaning 14:35. That is a
# CLOCK READING, not a quantity. Treating it as a number is wrong in two ways:
#
#   1. The gaps are fake. 1359 -> 1400 is one minute apart in real life, but
#      41 apart as a number. The model would see a huge jump where there is none.
#
#   2. The scale is enormous. Its values run to about 2400, while our scaled
#      features sit near 0 with a spread of 1. K-Nearest Neighbours measures
#      the distance between rows, so this one column would drown out every
#      other feature and effectively become the only thing the model looks at.
#
# We already extracted the useful part of it into DEP_HOUR and ARR_HOUR, which ARE scaled properly.
# So the raw versions are dropped.
DROP_AFTER_ENGINEERING = [
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
]



#SPLIT, TRAIN / TEST

# Flight data is time-ordered, so we must NOT split it randomly.
# A random split would let the model learn from June and be tested on May,
# which is predicting the past from the future. That is a form of leakage.
#
#So we sort by date and cut: the earliest 80% of DAYS become the
# training set, the latest 20% of DAYS become the test set.

TEST_SIZE = 0.20          # 20% test dataset
DATE_COLUMN = "FL_DATE"


# SECTION 8: PREPROCESSING SETTINGS

# Airport and airline codes are "categorical" -- they are labels, not numbers.
#Models need numbers, so we convert each code into its own yes/no column.
#
#PROBLEM was there are hundreds of airports. One column each would create a
#huge, slow, mostly-empty table.

#SOLUTION will be keeping only the most common categories and group everything else
# into a single "OTHER" bucket.

#The list of "most common" categories is worked out from the
# TRAINING data only. Working it out from all the data would be leakage.

TOP_N_CATEGORIES = {
    "OP_CARRIER": 15,   # keep the 15 most common airlines
    "ORIGIN": 30,       # keep the 30 busiest departure airports
    "DEST": 30,         # keep the 30 busiest arrival airports
}

OTHER_CATEGORY_LABEL = "OTHER"



#MODEL SETTINGS

# We compare a baseline against three models. All three are on the list of
# models taught in class(requirements Section 6).
#
# Cross-validation folds: how many times we split the TRAINING data during
# tuning. 3 is faster, 5 will be more reliable but slower to process.
CV_FOLDS = 3


# "f1_macro" averages the F1 score across all three classes EQUALLY.
# We chose it because our classes are imbalanced: about 80% of flights are
# on time. A metric like accuracy would let a model score 80% by always
# predicting "on_time" and never spotting a single delay.
TUNING_SCORE = "f1_macro"

#The settings we try for each model during tuning.
#may add more values, but more values = slower runs.
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1.0],           # how strongly to discourage large weights
    },
    "knn": {
        "n_neighbors": [15, 35],   #neighbours vote on each prediction
    },
    "random_forest": {
        "n_estimators": [100],     #how many trees in the forest
        "max_depth": [10, 20],     #how deep each tree may grow
    },
}



#PLOT SETTINGS


CLASS_COLORS = {
    "on_time": "#0173B2",   # blue
    "minor":   "#DE8F05",   # orange
    "major":   "#029E73",   # green
}

FIGURE_DPI = 150            
FIGURE_SIZE = (9, 5.5)      # default width, height inches



#HELPER GUIDE TO PRINT ALL SETTINGS

def print_config():
    """
    Prints the key settings.

    Every script calls this at the start, so the terminal output itself
    becomes a record of exactly which settings produced those results.
    That is part of what Section 11 asks for.
    """
    print("=" * 70)
    print("PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"  Data file        : {get_data_path()}")
    print(f"  Using sample data: {USE_SAMPLE_DATA}")
    if USE_SAMPLE_DATA:
        print("     >>> WARNING: sample data is for TESTING CODE ONLY. <<<")
        print("     >>> No report result may come from it.              <<<")
    print(f"  Random seed      : {RANDOM_STATE}")
    print(f"  Max rows         : {MAX_ROWS}  (method: {SAMPLING_METHOD})")
    if DEMO_MODE:
        print("     >>> DEMO MODE IS ON. Only a tenth of the data is used. <<<")
        print("     >>> These numbers will NOT match results/ and must     <<<")
        print("     >>> NOT be quoted in the report. Set DEMO_MODE = False <<<")
        print("     >>> to reproduce the submitted results.                <<<")
    print(f"  Test size        : {TEST_SIZE} (time-aware split)")
    print(f"  Classes          : {CLASS_NAMES}")
    print(f"  Numeric features : {NUMERIC_FEATURES}")
    print(f"  Categorical feats: {CATEGORICAL_FEATURES}")
    print(f"  Tuning score     : {TUNING_SCORE}")
    print("=" * 70)
    print()


def make_folders():
    """
    Creates the results folders if they do not already exist.

    Without this, saving a plot would crash on a freshly cloned project
    because results/figures/ would not exist yet.
    """
    for folder in [PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, SAMPLE_DATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
