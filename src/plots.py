# Creates the diagnostic and comparison charts for final results
#
#SECTION 10 REQUIREMENTS
#     "Outcome or cluster context"   -> eda.py, plot 1 (class distribution)
#     "Exploratory relationship"     -> eda.py, plots 2, 3 and 4
#     "Model diagnostic"             -> THIS FILE (confusion matrix)
#     "Comparison"                   -> THIS FILE (model comparison chart)
#     "Accessibility"                -> followed throughout
#     "Integrity"                    -> followed throughout
#
# HOW WE MET THE ACCESSIBILITY RULE
#     Section 10. So in every chart here:
#         - the actual numbers are printed on the chart itself
#         - line charts use different marker shapes as well as colours
#         - every axis has a label and a unit
#         - titles say what the chart SHOWS
#
# HOW WE MET THE INTEGRITY RULE
#     Section 10.
#         - every bar chart starts its axis at zero
#         - no 3D effects anywhere
#         - baselines are plotted alongside the real models, so a weak
#           result isn't hidden
#         - titles describe association
#
# WORKFLOW STAGE COVERED: DW-9, Section 10.
#22-48766-3 AREFIN
#TO RUN: [] python -m src.plots

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _save(fig, filename):
    """Saves a figure to results/figures/ and closes it."""
    config.make_folders()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVED] {path.name}")
    return path



#CONFUSION MATRIX, required model diagonistics


def plot_confusion_matrix(result, filename=None):
    # Draws the confusion matrix as a coloured grid.
    #
    #TO READ IT:
    #     Each ROW is what the flight ACTUALLY was.
    #     Each COLUMN is what the model PREDICTED.
    #
    #     The diagonal running from top-left to bottom-right holds the
    #     correct answers. Everything else is a mistake.
    #
    #TO LOOK FOR:
    #     - A bright top-left cell with dim cells elsewhere means the model
    #       mostly just predicts "on_time".
    #     - An entire empty COLUMN means the model never predicts that class
    #       at all. That is the failure mode imbalanced data causes, and it
    #       is the single most important thing this chart reveals.
    
    name = result["model"]
    matrix = result["confusion_matrix"]

    #Converts each row to percentages of that row's total.
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    matrix_percent = 100 * matrix / row_totals

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    #Colour each cell by its ROW percentage rather than its raw count
    image = ax.imshow(matrix_percent, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)

    ax.set_xlabel("PREDICTED class", fontsize=11, fontweight="bold")
    ax.set_ylabel("ACTUAL class", fontsize=11, fontweight="bold")

    #Writes both numbers inside every cell.
    for i in range(len(config.CLASS_NAMES)):
        for j in range(len(config.CLASS_NAMES)):
            count = matrix[i, j]
            percent = matrix_percent[i, j]

            
            #numbers stay readable in every cell.
            text_color = "white" if percent > 50 else "black"

            ax.text(
                j, i, f"{count:,}\n({percent:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold",
            )

    accuracy = result["accuracy"]
    macro_f1 = result["macro_f1"]

    ax.set_title(
        f"Confusion Matrix: {name}\n"
        f"accuracy = {accuracy:.3f}   macro F1 = {macro_f1:.3f}\n"
        f"Percentages are row-wise (that is, recall)",
        fontsize=12, fontweight="bold", pad=15,
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage of the actual class (%)", fontsize=10)

    if filename is None:
        filename = f"05_confusion_matrix_{name}.png"
    return _save(fig, filename)



#MODEL COMPARISON, required "comparison" chart

def plot_model_comparison(comparison_table):
    """
    Bar chart comparing every model on accuracy and macro F1 side by side.

    WHY BOTH METRICS ON ONE CHART:
        This single picture makes the accuracy trap visible. The majority
        baseline will show a tall accuracy bar next to a short macro F1
        bar, and anyone looking at it immediately understands why we did
        not judge by accuracy.

    INTEGRITY NOTE: the y-axis starts at 0. Starting it higher would
    exaggerate small differences between the models, which Section 10
    explicitly forbids.
    """
    print("\n  Plot 6: model comparison")

    table = comparison_table.sort_values("macro_f1", ascending=True)

    models = table["model"].tolist()
    y_positions = np.arange(len(models))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.75 * len(models) + 2)))

    bars_accuracy = ax.barh(
        y_positions + bar_height / 2, table["accuracy"],
        height=bar_height, label="Accuracy",
        color=config.CLASS_COLORS["on_time"], edgecolor="black", linewidth=0.6,
    )
    bars_f1 = ax.barh(
        y_positions - bar_height / 2, table["macro_f1"],
        height=bar_height, label="Macro F1  (our main metric)",
        color=config.CLASS_COLORS["minor"], edgecolor="black", linewidth=0.6,
        hatch="//",  
    )

    
    for bar_group in [bars_accuracy, bars_f1]:
        for bar in bar_group:
            width = bar.get_width()
            ax.text(
                width + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score (0 to 1)", fontsize=11)
    ax.set_xlim(0, 1.0)     
    ax.set_title(
        "Accuracy Disagrees With Macro F1\n"
        "The majority baseline scores high on accuracy while detecting no delays",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "06_model_comparison.png")



# PLOT 7: PER-CLASS RECALL

def plot_per_class_recall(results):
    # Grouped bar chart of each model's recall on each of the three classes.
    #
    # WHAT THIS SHOWS THAT THE COMPARISON CHART CANNOT:
    #     Whether a model is genuinely useful, or merely good at the easy
    #     class. A model with high overall scores but near-zero recall on
    #     "major" is useless for the actual job, warning passengers about
    #     serious delays and only this chart makes that obvious.
    #
    # We expect KNN to look bad here, for the structural reason explained in
    # models.py , it has no class_weight setting.
    print("\n  Plot 7: per-class recall")

    model_names = [r["model"] for r in results]
    x_positions = np.arange(len(model_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))


    hatches = ["", "//", "xx"]

    for class_index, class_name in enumerate(config.CLASS_NAMES):
        recalls = [r["recall_per_class"][class_index] for r in results]
        offset = (class_index - 1) * bar_width

        bars = ax.bar(
            x_positions + offset, recalls, bar_width,
            label=class_name,
            color=config.CLASS_COLORS[class_name],
            edgecolor="black", linewidth=0.6,
            hatch=hatches[class_index],
        )

        for bar, value in zip(bars, recalls):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{value:.2f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Recall  (share of that class correctly found)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Recall by Class: Who Actually Detects Delays?\n"
        "A model with near-zero recall on 'major' cannot do the job it was built for",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="Actual class", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, "07_per_class_recall.png")



#FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    # Bar chart of which features the Random Forest relied on most.
    
    #     Random Forest importance scores are BIASED. They systematically
    #     favour features with many distinct values. Our columns each
    #     have only two values (0 and 1), while DISTANCE has hundreds, so
    #     DISTANCE gets an unfair advantage in this ranking.
    #
    #     Requirement Section 6 specifically asks us to discuss
    #     "feature-importance limitations", This is that limitation.
    #
    #     So, this chart as a rough hint about what the model used,
    #     NOT as a ranking of what truly causes delays. Returns None if the model has no feature importances, only tree-based
    # models do.

    if not hasattr(model, "feature_importances_"):
        print(f"    (skipping feature importance: {model_name} does not provide it)")
        return None

    print(f"\n  Plot 8: feature importance for {model_name}")

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)
    top_features = top_features.sort_values(ascending=True)   # for the bar chart

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(top_features) + 2)))

    ax.barh(
        top_features.index, top_features.values,
        color=config.CLASS_COLORS["major"], edgecolor="black", linewidth=0.6,
    )

    for i, value in enumerate(top_features.values):
        ax.text(value + value * 0.02, i, f"{value:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Importance score (higher = used more by the model)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(
        f"Top {len(top_features)} Features Used by {model_name}\n"
        "CAUTION: these scores favour features with many distinct values\n"
        "and do not prove what causes delays",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, top_features.max() * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, f"08_feature_importance_{model_name}.png")



#TO MAKE EVERY RESULT PLOT


def make_all_result_plots(results, comparison_table, models, feature_names):
    """Produces every chart needed for the results section of the report."""
    print("\n" + "=" * 70)
    print("GENERATING RESULT PLOTS  (requirement Section 10)")
    print("=" * 70)

    print("\n  Plot 5: confusion matrices (one per model)")
    for result in results:
        plot_confusion_matrix(result)

    plot_model_comparison(comparison_table)
    plot_per_class_recall(results)

    # Feature importance only exists for tree-based models.
    if "random_forest" in models:
        plot_feature_importance(
            models["random_forest"], feature_names, "random_forest"
        )

    print("\n" + "=" * 70)
    print(f"PLOTS COMPLETE -> {config.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    #Running this file on its own rebuilds whole workflow first, because the plots need real results to draw.
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

    make_all_result_plots(
        results, comparison, trained_models, prepared["feature_names"]
    )


    
