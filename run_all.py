#This runs the ENTIRE project from start to finish
#To run DEMO.

#Requirement Section 18.

#TO RUN: shell[] python run_all.py

# WHAT IT DOES, IN ORDER:
#     1.  Loads the data and remove leakage columns
#     2.  Audits the data for problems
#     3.  Splits by time into train and test
#     4.  Explores the training data
#     5.  Preprocess
#     6.  Verifies no leakage occurred
#     7.  Trains baselines and tune models
#     8.  Evaluates once on the test set
#     9.  Draws the result plots
#     10. Analyzes the errors

#THE ORDER CANNOT BE CHANGED:
#     Each stage depends on the one before it.The split
#     (stage 3) MUST happen before preprocessing (stage 5), or the
#     preprocessing would learn from test data, the project would
#     be invalid.
#Member 4 AREFIN (integration)

import time

import config
from src.data_loading import get_prepared_data
from src.data_audit import run_full_audit
from src.split import get_train_test
from src.eda import run_eda
from src.preprocessing import prepare_features
from src.leakage_check import check_no_leakage
from src.train import run_training
from src.evaluate import run_evaluation
from src.plots import make_all_result_plots
from src.error_analysis import run_error_analysis


def banner(step_number, title, owner):
    """Prints a clear heading so the terminal output is easy to follow."""
    print("\n\n")
    print("#" * 70)
    print(f"#  STEP {step_number}: {title}")
    print(f"#  Owner: {owner}")
    print("#" * 70)


def main():
    """Runs every stage of the project."""
    overall_start = time.time()

    print("=" * 70)
    print("FLIGHT DELAY PREDICTION -- FULL WORKFLOW")
    print("Programming in Python | Final-Term Project")
    print("=" * 70)

    config.print_config()
    config.make_folders()

#1.LOADING THE DATA
    
    banner(1, "LOAD DATA AND REMOVE LEAKAGE COLUMNS", "ESRAT")
    data, load_info = get_prepared_data()

    
#2.AUDITING DATA
    
    banner(2, "DATA AUDIT", "ESRAT")
    run_full_audit(data)

    
#3.SPLITTING BY TIME
    
    # This MUST come before preprocessing. Everything after this point
    # is only allowed to learn from the training half.
    banner(3, "TIME-AWARE TRAIN/TEST SPLIT", "AUVROW")
    train_df, test_df, split_info = get_train_test(data)

    
    #4.EXPLORING TRAINING DATA 
    
    banner(4, "EXPLORATORY DATA ANALYSIS", "AUVROW")
    run_eda(train_df)

    
    #5.PREPROCESS
    
    banner(5, "PREPROCESSING", "AUVROW")
    prepared = prepare_features(train_df, test_df)

    
    #6.PROOF OF NO LEAKAGE
    # If any check fails, this raises an error and stops process.
    banner(6, "LEAKAGE VERIFICATION", "AUVROW")
    check_no_leakage(train_df, test_df, prepared)

   
    #7.TRAINING
    
    banner(7, "BASELINE AND MODEL TRAINING", "NABIL")
    models, tuning_log = run_training(
        prepared["X_train"], prepared["y_train"]
    )

    
#8.EVALUATION (THE TEST SET OPENS HERE, ONCE)
    
    banner(8, "FINAL EVALUATION ON TEST SET", "AREFIN")
    results, comparison = run_evaluation(
        models, prepared["X_test"], prepared["y_test"]
    )

   
#9.PLOTS
    
 banner(9, "RESULT PLOTS", "AREFIN")
    make_all_result_plots(
        results, comparison, models, prepared["feature_names"]
    )

    
#10.ERROR ANALYSIS ON THE BEST MODEL
    
    banner(10, "ERROR ANALYSIS", "AREFIN")

    #"Best" is highest macro F1, which is the metric we chose in advance.
    # We did NOT pick it after seeing which model we liked.
    best_model_name = comparison.iloc[0]["model"]
    best_result = next(r for r in results if r["model"] == best_model_name)

    print(f"\nAnalyzing the best model by macro F1: {best_model_name}")
    print("(Metric was chosen before training,   not after seeing results.)")

    run_error_analysis(
        test_df,
        prepared["y_test"],
        best_result["y_predicted"],
        best_model_name,
    )

    
#SUMMARY
    
    total_time = time.time() - overall_start

    print("\n\n")
    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)

    print(f"\n  Total run time: {total_time:.1f} seconds")
    print(f"  Random seed   : {config.RANDOM_STATE}")

    print(f"\n  Data:")
    print(f"    Rows in file      : {load_info['rows_in_file']:,}")
    print(f"    Rows used         : {load_info['rows_after_preparation']:,}")
    print(f"    Leakage cols cut  : {len(load_info['leakage_columns_removed'])}")

    print(f"\n  Split:")
    print(f"    Cut-off date      : {split_info['cutoff_date']}")
    print(f"    Training rows     : {split_info['n_rows_train']:,}")
    print(f"    Test rows         : {split_info['n_rows_test']:,}")

    print(f"\n  Best model        : {best_model_name}")
    print(f"    macro F1        = {comparison.iloc[0]['macro_f1']:.4f}")
    print(f"    accuracy        = {comparison.iloc[0]['accuracy']:.4f}")

    print(f"\n  Outputs:")
    print(f"    Figures -> {config.FIGURES_DIR}")
    print(f"    Metrics -> {config.METRICS_DIR}")

    if config.USE_SAMPLE_DATA:
        print("\n" + "!" * 70)
        print("!  WARNING: THIS RUN USED FAKE SAMPLE DATA.")
        print("!")
        print("!  These numbers add no value, only to prove")
        print("!  the code runs.")
        print("!")
        print("!  Before producing anything for the report:")
        print("!    1. Need to put the real dataset in data/raw/")
        print("!    2. Need to set REAL_DATA_FILENAME in config.py to match it")
        print("!    3. Need to set USE_SAMPLE_DATA = False in config.py")
        print("!    4. Need to run this file again")
        print("!" * 70)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
