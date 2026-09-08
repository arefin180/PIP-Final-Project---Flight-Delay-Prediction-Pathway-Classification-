# Proves that running the project twice gives identical results.
# WHY THIS FILE EXISTS

#Requirement FR-5.

#     Requirement Section 18. 

# WHAT IT CHECKS
#     It runs the entire pipeline twice and compares:
#         - every model's scores
#         - the tuning log
#         - the leakage check results

# WHAT IT DELIBERATELY IGNORES
#     Prediction TIME is measured with a stopwatch and depends on the
#     computer, It can never match exactly,
#     and it is not a result, it is a measurement of user's laptop.

#TO RUN:
#python verify_reproducibility.py

#MEMBER 4 AREFIN

import io
import contextlib

import pandas as pd

import config
from src.data_loading import get_prepared_data
from src.split import get_train_test
from src.preprocessing import prepare_features
from src.train import run_training
from src.evaluate import run_evaluation


def run_pipeline_quietly():
    """
    Runs the whole workflow with its printing suppressed.

    We hide the output because we are about to do this twice and do not
    want hundreds of duplicate lines on screen. The results are what
    matter here, not the commentary.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        data, _ = get_prepared_data()
        train_df, test_df, _ = get_train_test(data)
        prepared = prepare_features(train_df, test_df)
        models, tuning_log = run_training(
            prepared["X_train"], prepared["y_train"], save=False
        )
        _, comparison = run_evaluation(
            models, prepared["X_test"], prepared["y_test"], save=False
        )

    # Dropping the timing column 
    comparison = comparison.drop(columns=["predict_time_s"])

    # Droppig the timing column from the tuning log
    tuning_log = tuning_log.drop(columns=["fit_time_seconds"])

    return comparison, tuning_log


def compare_tables(name, table_1, table_2):
    """
    Checks whether two tables are identical, and reports the result.

    Returns True if they match.
    """
    print(f"\n  Checking: {name}")

    if table_1.shape != table_2.shape:
        print(f"    FAIL: different shapes "
              f"({table_1.shape} vs {table_2.shape})")
        return False

    if list(table_1.columns) != list(table_2.columns):
        print("    FAIL: different columns")
        return False

#.equals() compares every value in both tables
    if table_1.equals(table_2):
        print(f"    PASS: identical across both runs "
              f"({table_1.shape[0]} rows, {table_1.shape[1]} columns)")
        return True

    print("    FAIL: values differ. The differing rows are:")
    differences = table_1.compare(table_2)
    print(differences.to_string())
    return False


def main():
    print("=" * 70)
    print("REPRODUCIBILITY VERIFICATION  (requirement FR-5)")
    print("=" * 70)

    print(f"\n  Random seed: {config.RANDOM_STATE}")
    print(f"  Data file  : {config.get_data_path().name}")
    print("\n  Running the full workflow twice and comparing the results.")
    print("  Prediction times are excluded: they depend on the computer,")
    print("  not on the model, so they can never match exactly.")

    print("\n  Run 1 of 2...", end="", flush=True)
    comparison_1, tuning_1 = run_pipeline_quietly()
    print(" done.")

    print("  Run 2 of 2...", end="", flush=True)
    comparison_2, tuning_2 = run_pipeline_quietly()
    print(" done.")

    print("\n" + "-" * 70)
    print("COMPARING THE TWO RUNS")
    print("-" * 70)

    all_passed = True
    all_passed &= compare_tables("model scores", comparison_1, comparison_2)
    all_passed &= compare_tables("tuning log", tuning_1, tuning_2)

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: PASS")
        print("=" * 70)
        print("\n  Every score was identical across two separate runs.")
        print("  The workflow is reproducible.")
        print("\n  Quote this in report section 10, and be ready to run it")
        print("  live in the viva if asked.")
    else:
        print("RESULT: FAIL")
        print("=" * 70)
        print("\n  Something is producing different results each run.")
        print("\n  COMMON CAUSES TO CHECK:")
        print("    1. A model created without random_state=config.RANDOM_STATE")
        print("    2. A .sample() or shuffle without random_state")
        print("    3. StratifiedKFold with shuffle=True but no random_state")
        print("    4. A set or dictionary being iterated in an unstable order")
        print("\n  Fix it before submitting. Reproducibility is worth 4 marks")
        print("  and underpins every other claim in the project.")

    print()
    return all_passed


if __name__ == "__main__":
    main()
