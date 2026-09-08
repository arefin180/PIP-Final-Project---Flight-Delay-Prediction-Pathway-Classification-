# Investigates WHERE and WHY the chosen model gets things wrong.
#
# WHY THIS FILE MATTERS MORE THAN THE SCORE ITSELF

#     "Macro F1 = 0.36" tells a reader almost nothing. It does not say
#     whether the model fails on evening flights, on one particular airline,
#     or on long-haul routes.
#
#     Requirement DW-10 asks us to "explain patterns, errors, uncertainty,
#     assumptions, bias risks, generalization limits, and what the model
#     should not be used for". Requirement Section 7 asks for
#     "subgroup or error-pattern analysis where possible".
#
#     This file produces that evidence. Under the mark scheme, evaluation and
#     error analysis together are worth 8 marks -- the largest single block
#     in the project. A weak model with strong error analysis scores better
#     than a strong model with none.
#
# WORKFLOW STAGE COVERED: DW-10
#Member 4 AREFIN
#
#TO RUN:[]
#     python -m src.error_analysis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _save(fig, filename):
    """Saves a figure and closes it."""
    config.make_folders()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVED] {path.name}")
    return path


#BUILDING A TABLE OF EVERY PREDICTION


def build_error_table(test_df, y_test, y_predicted):
    """
    Combines the original test rows with what the model predicted.

    We need the ORIGINAL (unprocessed) test table here, not the scaled one,
    because we want to group by readable values like "AA" and "hour 18" --
    not by scaled numbers like -0.43.

    Returns
    -------
    pandas.DataFrame
        The test rows plus three new columns:
            actual_class     what it really was
            predicted_class  what the model said
            is_correct       True or False
    """
    table = test_df.copy()

    table["actual_class"] = np.asarray(y_test)
    table["predicted_class"] = np.asarray(y_predicted)
    table["is_correct"] = table["actual_class"] == table["predicted_class"]

    # Readable names alongside the numbers.
    table["actual_name"] = table["actual_class"].map(
        lambda i: config.CLASS_NAMES[i])
    table["predicted_name"] = table["predicted_class"].map(
        lambda i: config.CLASS_NAMES[i])

    return table



#ERROR RATE BY GROUPING


def error_rate_by(error_table, column, min_rows=25):
    """
    Works out how often the model is wrong within each group.

    EXAMPLE: error_rate_by(errors, "OP_CARRIER") answers
             "which airlines does our model handle worst?"

    min_rows hides groups too small to say anything about. A carrier with
    4 test flights that were all wrong is not evidence of a weakness.
    """
    grouped = error_table.groupby(column)

    table = pd.DataFrame({
        "n_flights": grouped.size(),
        "n_correct": grouped["is_correct"].sum(),
        "error_rate_percent": (100 * (1 - grouped["is_correct"].mean())).round(2),
        "actual_delay_rate_percent": (
            100 * grouped["actual_class"].apply(lambda s: (s != 0).mean())
        ).round(2),
    })

    table = table[table["n_flights"] >= min_rows]
    return table.sort_values("error_rate_percent", ascending=False)



# WHICH MISTAKES ARE THE MODEL'S FAVOURITES?

def confusion_pairs(error_table):
    """
    Lists the most common actual->predicted mistakes.

    WHY THIS IS THE MOST USEFUL TABLE IN THE FILE:
        Not all mistakes cost the same.

        Predicting "on_time" for a flight that was actually "major"
        delayed is the WORST possible error: the passenger gets no
        warning at all about a serious delay.

        Predicting "minor" for something that was "major" is much less
        harmful: the passenger was at least warned to expect a delay.

        Section 7 asks about "false-positive/false-negative consequences".
    """
    mistakes = error_table[~error_table["is_correct"]]

    if len(mistakes) == 0:
        return pd.DataFrame()

    pairs = (
        mistakes
        .groupby(["actual_name", "predicted_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    pairs["percent_of_all_errors"] = (
        100 * pairs["count"] / len(mistakes)
    ).round(2)

    return pairs


def describe_error_severity(error_table):
    """
    Splits the mistakes into "dangerous" and "tolerable" and counts each.

    OUR DEFINITION, WHICH YOU SHOULD DEFEND IN THE REPORT:

        DANGEROUS  = the model said "on_time" but the flight was delayed.
                     The passenger receives no warning whatsoever.

        TOLERABLE  = the model warned of a delay but got the severity
                     wrong, or warned about a flight that was fine.
                     Annoying, but the passenger was not blindsided.

    This distinction is a judgement call, not a mathematical fact.
    """
    mistakes = error_table[~error_table["is_correct"]]

    if len(mistakes) == 0:
        return {}

    # Said on_time (class 0), but it was actually delayed (class 1 or 2).
    missed_delays = mistakes[
        (mistakes["predicted_class"] == 0) & (mistakes["actual_class"] != 0)
    ]

    # Specifically missed a MAJOR delay -- the worst case.
    missed_major = mistakes[
        (mistakes["predicted_class"] == 0) & (mistakes["actual_class"] == 2)
    ]

    # Warned about a delay that never happened.
    false_alarms = mistakes[
        (mistakes["predicted_class"] != 0) & (mistakes["actual_class"] == 0)
    ]

    # Warned, but got the severity wrong.
    wrong_severity = mistakes[
        (mistakes["predicted_class"] != 0) & (mistakes["actual_class"] != 0)
    ]

    return {
        "total_errors": len(mistakes),
        "missed_delays": len(missed_delays),
        "missed_major_delays": len(missed_major),
        "false_alarms": len(false_alarms),
        "wrong_severity": len(wrong_severity),
    }



#PLOTTING, ERROR RATE BY DEPARTURE HOUR


def plot_error_by_hour(error_table, model_name):
    """
    Line chart of the model's error rate against departure hour, with the
    true delay rate drawn alongside it.

    WHAT TO LOOK FOR:
        If the two lines rise together, the model is struggling exactly
        where delays are most common -- which is the worst place to
        struggle, because that is when passengers most need a warning.
    """
    print("\n  Plot 9: error rate by departure hour")

    by_hour = error_rate_by(error_table, "DEP_HOUR", min_rows=10).sort_index()

    if len(by_hour) == 0:
        print("    (not enough data per hour to plot)")
        return None

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    ax.plot(
        by_hour.index, by_hour["error_rate_percent"],
        marker="o", linewidth=2, markersize=6,
        color=config.CLASS_COLORS["major"], label="Model error rate",
    )
    ax.plot(
        by_hour.index, by_hour["actual_delay_rate_percent"],
        marker="s", linewidth=2, markersize=6, linestyle="--",
        color=config.CLASS_COLORS["minor"], label="True delay rate",
    )

    ax.set_title(
        f"Where {model_name} Goes Wrong, by Departure Hour\n"
        "Errors clustering where delays are common is the worst case",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Scheduled departure hour (24-hour clock)", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, f"09_error_by_hour_{model_name}.png")



# RUN THE FULL ERROR ANALYSIS


def run_error_analysis(test_df, y_test, y_predicted, model_name, save=True):
    """
    Produces every error table and chart for the chosen model.

    Returns
    -------
    dict
        Every table produced.
    """
    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS: {model_name}  (requirement DW-10)")
    print("=" * 70)

    error_table = build_error_table(test_df, y_test, y_predicted)

    n_total = len(error_table)
    n_correct = error_table["is_correct"].sum()
    n_wrong = n_total - n_correct

    print(f"\n  Test flights   : {n_total:,}")
    print(f"  Correct        : {n_correct:,}  ({100*n_correct/n_total:.2f}%)")
    print(f"  Wrong          : {n_wrong:,}  ({100*n_wrong/n_total:.2f}%)")

#WHICH MISTAKES, AND HOW BAD ARE THEY? 
    print("\n" + "-" * 70)
    print("MOST COMMON MISTAKES  (actual -> predicted)")
    print("-" * 70)

    pairs = confusion_pairs(error_table)
    if len(pairs) > 0:
        print(pairs.to_string(index=False))

    severity = describe_error_severity(error_table)
    if severity:
        print("\n" + "-" * 70)
        print("HOW SERIOUS ARE THE MISTAKES?")
        print("-" * 70)
        print(f"\n  Total mistakes            : {severity['total_errors']:,}")
        print(f"  Missed delays (said on_time"
              f" but flight was late)      : {severity['missed_delays']:,}")
        print(f"    ...of which were MAJOR delays missed entirely: "
              f"{severity['missed_major_delays']:,}")
        print(f"  False alarms (warned, but flight was fine): "
              f"{severity['false_alarms']:,}")
        print(f"  Right about a delay, wrong about severity : "
              f"{severity['wrong_severity']:,}")

        missed_major_percent = (
            100 * severity["missed_major_delays"] / severity["total_errors"]
        )
        print(f"""
    {missed_major_percent:.1f}% of all mistakes are major delays
    predicted as on_time. Those passengers get no warning at
    all about a serious delay. That is the costliest error
    our model makes, and Section 7 asks us to weigh exactly
    this kind of consequence.""")

    
# WHICH GROUPS DOES THE MODEL HANDLE WORST?
    print("\n" + "-" * 70)
    print("ERROR RATE BY AIRLINE")
    print("-" * 70)
    by_carrier = error_rate_by(error_table, "OP_CARRIER", min_rows=25)
    if len(by_carrier) > 0:
        print(by_carrier.to_string())
        print("\n  A high error rate here is a FAIRNESS concern worth raising")
        print("  in report section 9: passengers on some airlines get worse")
        print("  predictions than others.")

    print("\n" + "-" * 70)
    print("ERROR RATE BY MONTH")
    print("-" * 70)
    by_month = error_rate_by(error_table, "MONTH", min_rows=20).sort_index()
    if len(by_month) > 0:
        print(by_month.to_string())
        print("\n  Remember our test set covers only the LAST months of the")
        print("  data. If those months are unusual, our test score may not")
        print("  represent the whole year. That is distribution shift, and")
        print("  it belongs in report section 9.")

    print("\n" + "-" * 70)
    print("ERROR RATE BY ORIGIN AIRPORT (worst 10)")
    print("-" * 70)
    by_origin = error_rate_by(error_table, "ORIGIN", min_rows=25)
    if len(by_origin) > 0:
        print(by_origin.head(10).to_string())

    
# PLOT
    plot_error_by_hour(error_table, model_name)

    if save:
        config.make_folders()
        if len(pairs) > 0:
            pairs.to_csv(
                config.METRICS_DIR / f"errors_confusion_pairs_{model_name}.csv",
                index=False)
        if len(by_carrier) > 0:
            by_carrier.to_csv(
                config.METRICS_DIR / f"errors_by_carrier_{model_name}.csv")
        if len(by_month) > 0:
            by_month.to_csv(
                config.METRICS_DIR / f"errors_by_month_{model_name}.csv")
        print(f"\n[SAVED] Error tables -> {config.METRICS_DIR}")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "error_table": error_table,
        "confusion_pairs": pairs,
        "severity": severity,
        "by_carrier": by_carrier,
        "by_month": by_month,
        "by_origin": by_origin,
    }


if __name__ == "__main__":
    from src.data_loading import get_prepared_data
    from src.split import get_train_test
    from src.preprocessing import prepare_features
    from src.train import run_training
    from src.evaluate import run_evaluation

    config.print_config()

    data, _ = get_prepared_data()
    train, test, _ = get_train_test(data)
    prepared = prepare_features(train, test)
    trained_models, _ = run_training(prepared["X_train"], prepared["y_train"])
    results, comparison = run_evaluation(
        trained_models, prepared["X_test"], prepared["y_test"]
    )

#Analyzes the model that scored best on macro F1.
    best_name = comparison.iloc[0]["model"]
    best_result = next(r for r in results if r["model"] == best_name)

    run_error_analysis(
        test, prepared["y_test"], best_result["y_predicted"], best_name
    )
