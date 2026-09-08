## Programming in Python — Final-Term Project | Summer 25-26**

**Project title:** FLIGHT_DELAY_PREDICTOR

**Group members:**

| Member | Name    | ID          | Working area |
| M1     | ESRAT   | 23-50916-1  | Data provenance, audit, documentation |
| M2     | AUVROW  | 22-48920-3  | EDA, split design, preprocessing, leakage control |
| M3     | NABIL   | 24-60205-3  | Baseline, models, tuning |
| M4     | AREFIN  | 22-48766-3  | Evaluation, plots, error analysis, reproducibility |



## What this project does

Given a scheduled flight, predict how late it will arrive, using **only
information known before the plane departs**.

**Task type:** multi-class classification

| Class | Meaning    | Arrival delay |
| 0     | `on_time`  | under 15 minutes |
| 1     | `minor`    | 15 to 60 minutes |
| 2     | `major`    | over 60 minutes |

The 15-minute boundary is the official US Department of Transportation
definition of a delayed flight.

### The prediction moment

We predict **before the aircraft departs**,  imagining a passenger checking
the day before their flight. This means the model may only use scheduled
information: airline, route, scheduled times, distance, date.

Anything known only after departure is deliberately excluded.



## Setup

## Requirements

Python 3.10 or newer.

## Install

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Get the data

The results in `results/` were produced from **US domestic flights,
October to December 2018**.

Using the 2018 US flight on-time file and place it in `data/raw/` as
   `2018.csv`. The originating publisher is the US Bureau of
   Transportation Statistics;
   `data_dictionary.md`.

Cutted it down to the October–December quarter:
   ```bash
   python -m src.extract_subset
   ```
   This reads the 851.6 MB full-year file in chunks, so it never loads
   7.2 million rows into memory at once and writes
   `data/raw/flights_subset.csv` — 1,796,121 rows, covering
   2018-10-01 to 2018-12-31.

 `config.py` is already set for that file:
   ```python
   REAL_DATA_FILENAME = "flights_subset.csv"
   USE_SAMPLE_DATA = False
   ```

## Three months keeps a real seasonal signal and gives the time-aware split a wide enough test period to be meaningful, while cutting the file to a size that runs live in thedemo. Because the split takes the *latest* 20% of days, the test set landson 13–31 December, the holiday peak. That is a deliberately hard testperiod, and report discusses the distribution shift it causes. 

**Why not the whole quarter's 1.8 million rows.** `MAX_ROWS = 600_000` in
`config.py` caps the data for compute reasons, and
`SAMPLING_METHOD = "stratified_by_day"` keeps the same share (about one
flight in three) of **every** calendar day. All 92 days survive the cut, so
the monthly pattern, the weekday pattern and the time-aware split are all
preserved. Keeping the *oldest* 600,000 rows instead would have stopped in
early November and thrown December away.

## Test the code without the real data

A small **fake** dataset is included so the code can be run before the
real data was used:

```bash
python -m src.make_sample_data
```

> **The sample data was generated.** to verify the code runs.
> No number, plot, or conclusion in the report came from it. See
> `data/sample/SAMPLE_DATA_README.txt`.
>
> Everything currently in `results/figures/` and `results/metrics/` came
> from the real October–December 2018 data, not from this file.

---

## How to run

## Everything at once (demo)

```bash
python run_all.py
```

This runs all ten stages in order and prints what it is doing at each
step. Total run time depends on `MAX_ROWS` in `config.py`.

## One stage at a time (for developing)

Each module runs standalone and rebuilds whatever it needs:

```bash
python -m src.extract_subset     # cut 2018.csv down to Oct-Dec
python -m src.make_sample_data   # create fake test data
python -m src.data_loading       # load, remove leakage columns, build target
python -m src.data_audit         # data quality checks
python -m src.split              # time-aware train/test split
python -m src.eda                # exploratory plots and summaries
python -m src.preprocessing      # scaling, encoding, imputation
python -m src.leakage_check      # prove no leakage occurred
python -m src.models             # print model definitions
python -m src.train              # tune and train
python -m src.evaluate           # final test-set evaluation
python -m src.plots              # result charts
python -m src.error_analysis     # where the model fails, and why
```


## 4. File structure

flight_delay_project/
├── README.md                  <- this file
├── requirements.txt           <- Python package versions
├── config.py                  <- ALL settings: paths, seeds, features, thresholds
├── data_dictionary.md         <- variable documentation (M1)
├── run_all.py                 <- runs the entire workflow
│
├── data/
│   ├── raw/                   <- the real dataset
│   │   ├── 2018.csv           <-   full year, 7,213,446 rows
│   │   └── flights_subset.csv <-   Oct-Dec, 1,796,121 rows
│   ├── sample/                <- fake data for testing code only
│   └── processed/             <- anything the code generates
│
├── src/
│   ├── extract_subset.py      <- cut the year down to Oct-Dec      (M1)
│   ├── make_sample_data.py    <- generates fake test data          (M4)
│   ├── data_loading.py        <- load, drop leakage, build target  (M1)
│   ├── data_audit.py          <- quality checks                    (M1)
│   ├── split.py               <- time-aware split                  (M2)
│   ├── eda.py                 <- exploratory analysis and plots    (M2)
│   ├── preprocessing.py       <- scaling, encoding, imputation     (M2)
│   ├── leakage_check.py       <- automated leakage proof           (M2)
│   ├── models.py              <- model definitions and defences    (M3)
│   ├── train.py               <- tuning and training               (M3)
│   ├── evaluate.py            <- final test evaluation             (M4)
│   ├── plots.py               <- result charts                     (M4)
│   └── error_analysis.py      <- error patterns                    (M4)
│
├── verify_reproducibility.py  <- runs the pipeline twice, compares (M4)
├── member4_work.md            <- M4's completed-work write-up      (M4)
│
├── results/
│   ├── figures/               <- all .png charts
│   └── metrics/               <- all .csv tables




## Leakage control

Requirement FR-2 forbids any test-set information influencing the model.
Flight data makes this easy to get wrong, so we handle it in three layers.

## Layer 1 — Forbidden columns are deleted on load

These columns are removed in `data_loading.py` before anything else
happens. Each is known only **after** the aircraft has departed, so none
of them exists at our prediction moment.

| Column | Why it is banned |
| `DEP_DELAY` | How late it actually left. Correlates ~0.95 with arrival delay. **The mistake.** |
| `DEP_TIME` | The actual departure time |
| `TAXI_OUT`, `TAXI_IN` | Ground movement, after pushback |
| `WHEELS_OFF`, `WHEELS_ON` | Actual take-off and landing times |
| `ARR_TIME` | The actual arrival time |
| `ACTUAL_ELAPSED_TIME`, `AIR_TIME` | How long the flight really took |
| `CANCELLED`, `DIVERTED` | Describe the outcome itself |
| `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | Official reasons for a delay — they only exist if there **was** a delay |

The full list with explanations is in `config.FORBIDDEN_COLUMNS`.

## Layer 2 — Time-aware splitting, before any learning

We split by **date**, not randomly. The earliest 80% of days become
training data; the latest 20% become the test set.

A random split would put December flights in training and January flights
in testing — predicting the past from the future, which is not how the
model would be used.

We split on whole **days**, not rows, so that flights sharing the same
day's weather and congestion cannot straddle the boundary.

## Layer 3 — Preprocessing learns from training data only

Every transformation follows one rule:

> **LEARNING from train. APPLIED to train and test.**

| Step        | What is learned from training only |
| Imputation  | The median used to fill gaps |
| Scaling     | The mean and standard deviation |
| Encoding    | Which categories are common enough to keep |

We wrote this rather than using `sklearn.Pipeline`, so that every
member can read the fit/transform separation and explain.
Requirement DW-6 prefers a pipeline, but we backed it up with automated proof:

```bash
python -m src.leakage_check
```

This runs six tests and stops the workflow if any fails. Output goes to
`results/metrics/leakage_checks.csv`.

The table also carries one line marked **INFO** rather than PASS. That is
the feature-collision rate (limitation 7 below): a measurement we report
for the record, not a test that can pass or fail. Calling a measurement
"PASS" would overstate what was actually verified.


## Reproducibility

| Requirement    | How we meet it |
| Seeds          | `RANDOM_STATE = 42` in `config.py`, used in every random operation |
| Environment    | `requirements.txt` with pinned versions |
| No local paths | All paths built from `PROJECT_ROOT` in `config.py` |
| Run order      | `run_all.py` executes all stages in the correct sequence |
|Artifacts       | Every table saved to `results/metrics/`, every chart to `results/figures/` |
| Compute cost   | Training and prediction times printed and logged |

**Changing settings:** edit `config.py` only.

## Hardware and measured run time

Every figure and table in `results/` came from one run on this machine:

| Item               | Value |
| OS                 | Windows 11, AMD64 |
| CPU                | Intel64 Family 6 Model 165, 12 logical cores |
| Python             | 3.14.6 |
| Rows analysed      | 600,002 (`MAX_ROWS = 600_000`, stratified by day) |
| **Total run time** | **24 min 25 sec** |



| Stage                              | Time          | Note |
| Logistic regression tuning         | 27.0 s        | 2 settings x 3 folds |
| **KNN tuning**                     | **1,226.4 s** | **84% of the entire run** |
| Random forest tuning               | 117.0 s       | 2 settings x 3 folds |
| KNN test prediction                | 79.6 s        | vs 0.38 s for random forest |
| Everything else                    | ~15 s         | load, audit, split, EDA, preprocess, leakage, plots |

**KNN dominates the cost, and it finished last on quality.** It compares
every test flight against all 477,384 training flights across 84 features.
That is roughly 200x the prediction time of random forest for the worst
macro F1 of any real model. This trade-off is the compute half of the
model defence in report section 6.

### Demo mode

The full 24-minute run is too slow to sit through in a live demonstration
(Section 18). `config.py` therefore has:

```python
DEMO_MODE = True     #60,000 rows, finishes in approx 2 minutes
DEMO_MODE = False   #600,000 rows, PRODUCED EVERY SUBMITTED RESULT
```

Same models, split logic, seed, checks, only the row count
changes. The workflow prints a warning whenever demo mode is on.

**Demo-run numbers didn't appear in the report.** as it comes from a
tenth of the data


## 7. Known limitations

Stated per requirements Section 9 and Section 14 (not hiding
unfavourable results).

1. **Pre-departure prediction is genuinely hard.** Schedule-only features
   carry weak signal, especially for rare severe delays. Expect modest
   macro F1. Section 8.

2. **No weather data.** Weather causes a large share of real delays and we
   have none of it. This is the single biggest gap in the feature set.

3. **Row cap.** `MAX_ROWS` limits the data for compute reasons. This is a
   documented decision, not hidden filtering.

4. **Distribution shift.** The test set covers a different time period from
   training, and delays are seasonal, so the two periods are not identical.

5. **KNN cannot use class weights.** `KNeighborsClassifier` has no
   `class_weight` parameter, so it will under-predict rare classes for a
   structural reason. Documented as a model-specific difference under
   Section 8.

6. **Feature importance is biased.** Random Forest importance scores favour
   features with many distinct values as a hint, not a ranking
   of true causes.

7. **Identical inputs, different outcomes — a hard ceiling on accuracy.**
   Our features are deliberately coarse: scheduled times are rounded to the
   hour, the data spans 3 months, and only the 30 busiest airports keep
   their own identity while the remaining ~318 collapse into one `OTHER`
   bucket (about 35% of rows). The result is that **21.5% of test flights
   have a feature vector that also occurs in training**, and about **10% of
   repeated vectors carry more than one outcome**.

   These are different real flights that our feature set simply cannot tell
   apart. When the same input genuinely maps to different answers, no model
   — however good — can get all of them right. This caps the achievable
   score before any modelling decision is made, and it is a large part of
   why the macro F1 figures are modest.

   It is **not** leakage: training and test share zero calendar dates, so no
   individual flight is on both sides. `src/leakage_check.py` asserts that
   separately and reports the collision rate alongside it as a measurement.


## Probable Use cases

This model **should not** be used to:

- Make compensation or refund decisions about individual passengers
- Rank or penalise airlines, the data shows association, not cause
- Guarantee anything to a traveller, given the accuracy achieved

It **may reasonably** also be used to:

- Illustrate which scheduled factors are associated with delay
- Provide a rough, clearly-labelled likelihood estimate




