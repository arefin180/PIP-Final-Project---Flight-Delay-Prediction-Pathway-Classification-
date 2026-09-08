"""
Scores the models and produces the final comparison.

    The test set is used ONCE, here, at the very end, after every decision
    has already been made.


WHICH METRICS AND WHY as per req Sec 7

    Section 7 requires, for classification: a confusion matrix and
    class-wise precision, recall and F1. Accuracy only when informative.

    ACCURACY -- what fraction of predictions were right?
        Nearly useless for us on its own. Predicting "on_time" every single
        time scores about 79%. We report it, but only next to the baseline
        so the reader can see how misleading it is.

    PRECISION -- when the model said "delayed", how often was it right?
        Matters when a false alarm is costly. Telling passengers a flight
        will be late when it is not damages trust.

    RECALL -- of the flights that really were delayed, how many did we catch?
        Matters when a miss is costly. A missed delay means passengers get
        no warning at all.

    F1 -- the balance between precision and recall.
        One number for when you need a single figure. Its "macro" average
        treats all three classes equally, which is what we tune on.

WORKFLOW STAGE COVERED: DW-9
MEMBER 4 AREFIN

TO RUN: []  python -m src.evaluate
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config
from src.data_loading import get_prepared_data
from src.split import get_train_test
from src.preprocessing import prepare_features
from src.leakage_check import check_no_leakage
from src.train import run_training
from src.models import MODEL_DESCRIPTIONS



#SCORING ONE MODEL


def evaluate_one_model(name, model, X_test, y_test):
    """
    Calculates every required metric for a single model.

    Returns
    -------
    dict
        All the scores, plus the predictions themselves (which the error
        analysis file needs later).
    """
    #Timing the prediction, Section 8. 
    start = time.time()
    y_predicted = model.predict(X_test)
    predict_time = time.time() - start

    accuracy = accuracy_score(y_test, y_predicted)
    macro_f1 = f1_score(y_test, y_predicted, average="macro")
    weighted_f1 = f1_score(y_test, y_predicted, average="weighted")

# Per-class scores. zero_division=0 means: if a model never predicts a
    # class at all, record 0 rather than crashing. Our majority baseline
 # never predicts "minor" or "major", so without this it would fail.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_predicted,
        labels=[0, 1, 2],
        zero_division=0,
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predict_time_seconds": predict_time,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "y_predicted": y_predicted,
        "confusion_matrix": confusion_matrix(y_test, y_predicted, labels=[0, 1, 2]),
    }



# PRINTING ONE MODEL'S RESULTS

def print_model_results(result, y_test):
    """Prints one model's scores in a readable form."""
    name = result["model"]

    print("\n" + "-" * 70)
    print(f"RESULTS: {name}")
    print("-" * 70)
    print(f"  {MODEL_DESCRIPTIONS.get(name, '')}")

    print(f"\n  Overall accuracy : {result['accuracy']:.4f}")
    print(f"  Macro F1         : {result['macro_f1']:.4f}   <-- our main metric")
    print(f"  Weighted F1      : {result['weighted_f1']:.4f}")
    print(f"  Prediction time  : {result['predict_time_seconds']:.4f} seconds")

    print("\n  Per-class scores:")
    print(f"    {'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_flights':>11}")
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"    {class_name:<10} "
              f"{result['precision_per_class'][i]:>10.4f} "
              f"{result['recall_per_class'][i]:>10.4f} "
              f"{result['f1_per_class'][i]:>10.4f} "
              f"{result['support_per_class'][i]:>11,}")

    # THE CONFUSION MATRIX
    # Rows    = what the flight ACTUALLY was
    # Columns = what the model PREDICTED
    
    # The diagonal, top-left to bottom-right, holds the correct answers.
    # Everything off the diagonal is a mistake, and WHERE the mistakes sit
    # tells you what kind of mistake the model makes.
    print("\n  Confusion matrix (rows = actual, columns = predicted):")

    matrix = result["confusion_matrix"]
    header = "                " + "".join(f"{n:>12}" for n in config.CLASS_NAMES)
    print(f"                {'PREDICTED':>36}")
    print(header)

    for i, class_name in enumerate(config.CLASS_NAMES):
        row_label = f"  ACTUAL {class_name:<8}"
        row_values = "".join(f"{value:>12,}" for value in matrix[i])
        print(row_label + row_values)

    # A warning if the model has quietly given up on a class entirely.
    for i, class_name in enumerate(config.CLASS_NAMES):
        if result["recall_per_class"][i] == 0 and result["support_per_class"][i] > 0:
            print(f"\n    NOTE: this model never correctly identified a single "
                  f"'{class_name}' flight.")
            print(f"          It has effectively given up on that class. "
                  f"Say so in the report.")



# COMPARING EVERY MODEL

def build_comparison_table(results):
    """
    Puts every model's headline scores into one table.

    This table is required by FR-3 (meaningful comparison) and Section 10
    (a clear comparison table). It is probably the single most important
    table in the report.
    """
    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "accuracy": round(result["accuracy"], 4),
            "macro_f1": round(result["macro_f1"], 4),
            "weighted_f1": round(result["weighted_f1"], 4),
            "predict_time_s": round(result["predict_time_seconds"], 4),
        }
        # Adding the per-class recall
        for i, class_name in enumerate(config.CLASS_NAMES):
            row[f"recall_{class_name}"] = round(result["recall_per_class"][i], 4)
        rows.append(row)

    table = pd.DataFrame(rows)
    # Best macro F1 at the top.
    return table.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def interpret_comparison(table):
    """
    Prints plain-English guidance on how to read the comparison table.

    This is here because a table of numbers is not evidence until someone
    explains what it means. Requirement FR-4 says the conclusion must
    follow from the evidence, not from preference.
    """
    print("\n" + "=" * 70)
    print("HOW TO READ THIS COMPARISON")
    print("=" * 70)

    best = table.iloc[0]
    baseline_rows = table[table["model"].str.startswith("baseline")]

    if len(baseline_rows) > 0:
        best_baseline = baseline_rows.iloc[0]

        print(f"\n  Best model      : {best['model']}")
        print(f"    macro F1      = {best['macro_f1']:.4f}")
        print(f"    accuracy      = {best['accuracy']:.4f}")

        print(f"\n  Best baseline   : {best_baseline['model']}")
        print(f"    macro F1      = {best_baseline['macro_f1']:.4f}")
        print(f"    accuracy      = {best_baseline['accuracy']:.4f}")

        improvement = best["macro_f1"] - best_baseline["macro_f1"]
        print(f"\n  Improvement over baseline (macro F1): {improvement:+.4f}")

        if improvement < 0.05:
            print("\n    WARNING: the improvement over the baseline is small.")
            print("    Will be projected if the limitations are analyzed properly.")
            print("    DID NOT claiming a win we did not earn.")

       # Using "the best baseline" would be wrong, because the best baseline by
        # macro F1 is usually the random one, and the accuracy trap is
        # specifically about the MAJORITY baseline.
        majority = table[table["model"] == "baseline_majority"]

        if len(majority) > 0:
            majority = majority.iloc[0]
            best_accuracy = table["accuracy"].max()

            print("\n  NOTICE THE ACCURACY TRAP:")
            print(f"    The majority baseline scores {majority['accuracy']:.1%} accuracy")
            print(f"    while never predicting a single delay "
                  f"(recall on 'minor' = {majority['recall_minor']:.2f}, "
                  f"on 'major' = {majority['recall_major']:.2f}).")
            print(f"    That is HIGHER than the best real model's accuracy "
                  f"of {best_accuracy:.1%}.")
            print("\n    So by accuracy alone, the model that does nothing")
            print("    would look like the winner. This is precisely why")
            print("    Section 7 asks for class-wise metrics instead, and")
            print("    why we tuned on macro F1.")

    print("\n  BEFORE CLAIMING ONE MODEL IS BEST, CHECK:")
    print("    1. Is the difference big enough to be real? A gap of 0.01")
    print("       macro F1 probably is not (Section 8, 'Uncertainty').")
    print("    2. Does the winner do acceptably on the RARE classes, or")
    print("       does it only win by being good at 'on_time'?")
    print("    3. Is it fast enough, and can you explain how it works?")
    print("       Section 8 asks you to weigh runtime and interpretability")
    print("       alongside the score.")



#THE FINAL EVALUATION


def run_evaluation(models, X_test, y_test, save=True):
    """
    Scores every model on the test set. This happens exactly once.

    Returns
    -------
    (list, pandas.DataFrame)
        Detailed results for each model, and the comparison table.
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON THE TEST SET  (requirement DW-9)")
    print("=" * 70)
    print("\n  >>> THE TEST SET IS BEING OPENED NOW, FOR THE FIRST TIME. <<<")
    print("  >>> Whatever these numbers say, they are final.            <<<")
    print("  >>> Changing a model after seeing them would be test-set   <<<")
    print("  >>> tuning, which Section 14 forbids.                      <<<")
    print(f"\n  Test set: {len(X_test):,} flights the models have never seen.")

    print("\n  Actual class distribution in the test set:")
    for class_number, count in y_test.value_counts().sort_index().items():
        percent = 100 * count / len(y_test)
        print(f"    {config.CLASS_NAMES[class_number]:<10} {count:>7,}  ({percent:5.2f}%)")

    results = []
    for name, model in models.items():
        result = evaluate_one_model(name, model, X_test, y_test)
        results.append(result)
        print_model_results(result, y_test)

    comparison = build_comparison_table(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON  (requirement FR-3)")
    print("=" * 70)
    print("\nAll models used the same split, the same features, and the same")
    print("preprocessing, as Section 8 requires for a fair comparison.\n")
    print(comparison.to_string(index=False))

    interpret_comparison(comparison)

    if save:
        config.make_folders()

        comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

        
        # A SECOND COPY, WITHOUT THE TIMING COLUMN
       
        # Requirement FR-5 says a reviewer must be able to recreate
        # our tables. Every score here is fully reproducible, run the
        # project twice with the same seed and you get identical numbers.
        # But prediction TIME is measured with a stopwatch. It changes
        # depending on the computer, so it can never be byte-identical between two runs.

        #So we saved two files:
        #   model_comparison.csv          - everything, including timing
        #                                   (Section 8)
        #   model_comparison_metrics.csv  - scores only, and therefore
        #                                   exactly reproducible

        #The metrics-only file when checking reproducibility.
        metrics_only = comparison.drop(columns=["predict_time_s"])
        metrics_only.to_csv(
            config.METRICS_DIR / "model_comparison_metrics.csv", index=False)

        #Detailed per-class report for each model.
        detail_rows = []
        for result in results:
            for i, class_name in enumerate(config.CLASS_NAMES):
                detail_rows.append({
                    "model": result["model"],
                    "class": class_name,
                    "precision": round(result["precision_per_class"][i], 4),
                    "recall": round(result["recall_per_class"][i], 4),
                    "f1": round(result["f1_per_class"][i], 4),
                    "n_flights": int(result["support_per_class"][i]),
                })
        pd.DataFrame(detail_rows).to_csv(
            config.METRICS_DIR / "per_class_metrics.csv", index=False)

        print(f"\n[SAVED] {config.METRICS_DIR / 'model_comparison.csv'}")
        print(f"[SAVED] {config.METRICS_DIR / 'model_comparison_metrics.csv'}")
        print("        (the same table without timing, so it is exactly")
        print("         reproducible -- use this one to verify FR-5)")
        print(f"[SAVED] {config.METRICS_DIR / 'per_class_metrics.csv'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE -- the test set is now spent.")
    print("=" * 70)

    return results, comparison


if __name__ == "__main__":
    config.print_config()

    data, _ = get_prepared_data()
    train, test, _ = get_train_test(data)
    prepared = prepare_features(train, test)
    check_no_leakage(train, test, prepared)

    trained_models, _ = run_training(prepared["X_train"], prepared["y_train"])

    run_evaluation(trained_models, prepared["X_test"], prepared["y_test"])
