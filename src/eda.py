"""
eda.py
======
Exploratory Data Analysis: looking at the data to understand it.

WHAT "EXPLORATORY" MEANS HERE
    We are not building a model yet. We are asking simple questions and
    letting the answers guide our later decisions:

        - How many flights are actually delayed?
        - Does the time of day matter?
        - Are some airlines worse than others?
        - Is there a seasonal pattern?

    Requirement DW-4 asks for "summaries and purposeful plots". The word
    PURPOSEFUL matters: every chart here answers a question we can state
    out loud. A chart that decorates the report without answering anything
    should be deleted.

IMPORTANT -- WHICH DATA DO WE EXPLORE?
    We explore the TRAINING data only.

    Studying the test set, even just by eye, influences the choices we make
    afterwards. That is a quiet form of leakage the requirements call
    "information from test data influencing model choice" (FR-2).

WORKFLOW STAGE COVERED: DW-4
OWNER: Member 2

HOW TO RUN:
    python -m src.eda
"""

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.data_loading import get_prepared_data
from src.split import get_train_test


def _save(fig, filename):
    """Saves a figure into results/figures/ and closes it to free memory."""
    config.make_folders()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVED] {path.name}")
    return path



def numeric_summary(train_df):
    """
    Prints the standard summary statistics for every numeric column.

    HOW TO READ THE OUTPUT:
        count = how many values there are
        mean  = the average
        std   = the spread (bigger = more varied)
        min   = the smallest value
        25%   = a quarter of values are below this
        50%   = the middle value (the median)
        75%   = three quarters of values are below this
        max   = the largest value

    WHAT TO LOOK FOR: a min or max that looks impossible, or a mean that is
    very far from the median (which means the data is lopsided).
    """
    print("\n" + "-" * 70)
    print("SUMMARY OF NUMERIC COLUMNS  (training data only)")
    print("-" * 70)

    summary = train_df[config.NUMERIC_FEATURES].describe().T.round(2)
    print(summary.to_string())

    config.make_folders()
    summary.to_csv(config.METRICS_DIR / "eda_numeric_summary.csv")
    return summary


def delay_rate_by(train_df, column, min_flights=30):
    """
    Works out the delay rate for each value of a column.

    EXAMPLE: delay_rate_by(train, "OP_CARRIER") answers
             "what percentage of each airline's flights are delayed?"

    The min_flights setting hides groups with too few flights. A carrier
    with 3 flights that were all late is not evidence of anything, and
    showing it as "100% delayed" would be misleading.
    """
    grouped = train_df.groupby(column)[config.TARGET_COLUMN]

    table = pd.DataFrame({
        "n_flights": grouped.size(),
        "delayed_percent": (grouped.apply(lambda s: (s != 0).mean()) * 100).round(2),
        "major_percent": (grouped.apply(lambda s: (s == 2).mean()) * 100).round(2),
    })

    table = table[table["n_flights"] >= min_flights]
    return table.sort_values("delayed_percent", ascending=False)



def plot_class_distribution(train_df):
    """
    Bar chart of how many flights fall into each class.

    QUESTION IT ANSWERS: how imbalanced is our target?

    WHY IT IS REQUIRED: requirements Section 10 asks us to "show the
    target/class distribution". It is also the single most important chart
    in the report, because the imbalance it reveals justifies every metric
    choice we make later.
    """
    print("\n  Plot 1: class distribution")

    counts = train_df[config.TARGET_COLUMN].value_counts().sort_index()
    labels = [config.CLASS_NAMES[i] for i in counts.index]
    colors = [config.CLASS_COLORS[name] for name in labels]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.6)

   
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        percent = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({percent:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title(
        "Flight Arrival Outcomes Are Heavily Imbalanced\n"
        f"Training data: {total:,} flights",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Delay class", fontsize=11)
    ax.set_ylabel("Number of flights", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "01_class_distribution.png")




def plot_delay_by_hour(train_df):
    """
    Line chart of the delay rate against the scheduled departure hour.

    QUESTION IT ANSWERS: does the time of day affect delays?

    WHAT WE EXPECT TO SEE: delays rising through the day. Airlines run
    aircraft on tight rotations, so a morning delay cascades into every
    later flight that aircraft operates. Early flights start with a clean
    slate.

    WHY THIS CHART EARNS ITS PLACE: if the line is flat, DEP_HOUR is a
    useless feature and we should say so. If it rises, we have justified
    keeping it.
    """
    print("\n  Plot 2: delay rate by departure hour")

    by_hour = delay_rate_by(train_df, "DEP_HOUR", min_flights=20)
    by_hour = by_hour.sort_index()

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    ax.plot(
        by_hour.index, by_hour["delayed_percent"],
        marker="o", linewidth=2, markersize=6,
        color=config.CLASS_COLORS["minor"], label="Any delay (15+ min)",
    )
    ax.plot(
        by_hour.index, by_hour["major_percent"],
        marker="s", linewidth=2, markersize=6, linestyle="--",
        color=config.CLASS_COLORS["major"], label="Major delay (60+ min)",
    )

    ax.set_title(
        "Delays Accumulate Through the Day\n"
        "Later departures are more likely to arrive late",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Scheduled departure hour (24-hour clock)", fontsize=11)
    ax.set_ylabel("Percentage of flights delayed (%)", fontsize=11)
   
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "02_delay_rate_by_hour.png")



def plot_delay_by_carrier(train_df):
    """
    Horizontal bar chart of the delay rate for each airline.

    QUESTION IT ANSWERS: are some airlines meaningfully worse than others?

    WHY HORIZONTAL BARS: airline codes read more comfortably along the side
    than squeezed under a vertical axis.

    A WARNING FOR THE REPORT: a high delay rate does not prove an airline
    is badly run. It might simply fly more routes through congested
    airports, or more evening flights. This chart shows an ASSOCIATION,
    not a cause -- and Section 10 explicitly forbids implying causation
    from association.
    """
    print("\n  Plot 3: delay rate by airline")

    by_carrier = delay_rate_by(train_df, "OP_CARRIER", min_flights=50)
    by_carrier = by_carrier.sort_values("delayed_percent")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(by_carrier) + 2)))

    ax.barh(
        by_carrier.index, by_carrier["delayed_percent"],
        color=config.CLASS_COLORS["minor"], edgecolor="black", linewidth=0.6,
    )


    for i, (carrier, row) in enumerate(by_carrier.iterrows()):
        ax.text(
            row["delayed_percent"] + 0.4, i,
            f"{row['delayed_percent']:.1f}%  (n={int(row['n_flights']):,})",
            va="center", fontsize=9,
        )

    ax.set_title(
        "Delay Rate Varies by Airline\n"
        "Association only -- route mix and schedule may explain the difference",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Percentage of flights delayed 15+ minutes (%)", fontsize=11)
    ax.set_ylabel("Airline code", fontsize=11)
    ax.set_xlim(0, by_carrier["delayed_percent"].max() * 1.35)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "03_delay_rate_by_carrier.png")




def plot_delay_by_month(train_df):
    """
    Bar chart of the delay rate for each month.

    QUESTION IT ANSWERS: is there a seasonal pattern?

    WHY THIS MATTERS MORE THAN IT LOOKS: our train/test split is by time.
    If delays are strongly seasonal, then our test period covers different
    months from our training period, and the model will face conditions it
    was not trained on.

    That is called DISTRIBUTION SHIFT, and this chart is the evidence for
    discussing it in report section 9. It is one of the most defensible
    limitations we can raise in the viva.
    """
    print("\n  Plot 4: delay rate by month")

    by_month = delay_rate_by(train_df, "MONTH", min_flights=20).sort_index()

    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    labels = [month_labels.get(m, str(m)) for m in by_month.index]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    ax.bar(
        labels, by_month["delayed_percent"],
        color=config.CLASS_COLORS["on_time"], edgecolor="black", linewidth=0.6,
    )

 
    overall = 100 * (train_df[config.TARGET_COLUMN] != 0).mean()
    ax.axhline(
        overall, color="black", linestyle="--", linewidth=1.5,
        label=f"Overall average ({overall:.1f}%)",
    )

    ax.set_title(
        "Delay Rate Changes Across the Year\n"
        "Seasonality means our time-based test period differs from training",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of flights delayed (%)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "04_delay_rate_by_month.png")




def run_eda(train_df):
    """
    Produces every summary and every plot.

    REMEMBER: we pass in the TRAINING data only.
    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS  (requirement DW-4)")
    print("=" * 70)
    print(f"\nExploring {len(train_df):,} TRAINING flights.")
    print("The test set is not examined here -- looking at it would")
    print("influence our decisions, which requirement FR-2 forbids.")

    numeric_summary(train_df)

    print("\n" + "-" * 70)
    print("GENERATING PLOTS")
    print("-" * 70)

    plot_class_distribution(train_df)
    plot_delay_by_hour(train_df)
    plot_delay_by_carrier(train_df)
    plot_delay_by_month(train_df)

    config.make_folders()
    delay_rate_by(train_df, "OP_CARRIER", 50).to_csv(
        config.METRICS_DIR / "eda_delay_by_carrier.csv")
    delay_rate_by(train_df, "DEP_HOUR", 20).to_csv(
        config.METRICS_DIR / "eda_delay_by_hour.csv")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print(f"  Plots  -> {config.FIGURES_DIR}")
    print(f"  Tables -> {config.METRICS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    config.print_config()

    data, _ = get_prepared_data()
    train, test, _ = get_train_test(data)
    run_eda(train)
