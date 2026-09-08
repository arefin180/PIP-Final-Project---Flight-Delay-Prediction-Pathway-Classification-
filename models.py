from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import config


def get_baselines():
    return {
        "baseline_majority": DummyClassifier(
            strategy="most_frequent",
            random_state=config.RANDOM_STATE,
        ),
        "baseline_random": DummyClassifier(
            strategy="stratified",
            random_state=config.RANDOM_STATE,
        ),
    }


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=25,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


MODEL_DESCRIPTIONS = {
    "baseline_majority": (
        "Always predicts the most common class (on_time). Ignores every "
        "feature. Exists to show what a score looks like when no learning "
        "has happened at all."
    ),
    "baseline_random": (
        "Guesses randomly, in the same proportions as the training data. "
        "Represents knowing the overall delay rate but nothing about the "
        "individual flight."
    ),
    "logistic_regression": (
        "Draws straight boundaries between classes. Fast and interpretable "
        "-- its coefficients show which features push a flight towards "
        "delay. Limited to straight-line boundaries."
    ),
    "knn": (
        "Classifies a flight by majority vote among the 25 most similar "
        "past flights. Assumes nothing about the pattern's shape, but is "
        "slow and weakens as the number of features grows."
    ),
    "random_forest": (
        "Combines 100 decision trees, each trained on a random slice of "
        "the data, and takes their vote. Captures curves and interactions "
        "but is harder to interpret."
    ),
}


MODEL_JUSTIFICATIONS = {
    "logistic_regression": {
        "1_why_this_task": (
            "Multi-class classification is exactly what it does, via a "
            "one-vs-rest scheme across our three classes."
        ),
        "2_why_this_data": (
            "Tabular data with a moderate number of features. Its "
            "coefficients give us a readable statement about which "
            "schedule factors are associated with delay."
        ),
        "3_preprocessing_needed": (
            "Requires scaled numeric features (otherwise coefficients are "
            "not comparable) and one-hot encoded categories. Both done in "
            "preprocessing.py."
        ),
        "4_hyperparameters": (
            "C controls how strongly large coefficients are discouraged. "
            "Selected by cross-validation on the training set only."
        ),
        "5_tradeoffs": (
            "Most interpretable of the three and the fastest to train. "
            "Cannot represent curved boundaries or feature interactions."
        ),
        "6_vs_baseline": (
            "To be filled in from your results. It must clearly beat the "
            "majority baseline on macro F1, or it has earned no place in "
            "the project."
        ),
    },
    "knn": {
        "1_why_this_task": (
            "Similarity-based classification. The underlying intuition is "
            "reasonable: flights on similar routes at similar times "
            "plausibly behave similarly."
        ),
        "2_why_this_data": (
            "All features are numeric after preprocessing, so distances "
            "can be computed. HOWEVER, one-hot encoding produced 47 "
            "features, which is a genuine problem for KNN (see 5)."
        ),
        "3_preprocessing_needed": (
            "Scaling is ESSENTIAL. Without it, DISTANCE (values in the "
            "thousands) would completely dominate DEP_HOUR (values 0-23) "
            "and the model would effectively ignore every other feature."
        ),
        "4_hyperparameters": (
            "k, the number of neighbours who vote. Small k overreacts to "
            "noise; large k blurs real distinctions. Chosen by "
            "cross-validation on the training set only."
        ),
        "5_tradeoffs": (
            "Slowest to predict, since every new flight is compared "
            "against all training flights. Also suffers from the curse of "
            "dimensionality at 47 features: distances between points "
            "become nearly equal, so 'nearest' loses meaning. Has no "
            "class_weight option, so it will under-predict our rare "
            "classes -- a documented model-specific difference under "
            "Section 8."
        ),
        "6_vs_baseline": "To be filled in from your results.",
    },
    "random_forest": {
        "1_why_this_task": (
            "Handles multi-class classification directly and copes well "
            "with imbalanced data when class weighting is applied."
        ),
        "2_why_this_data": (
            "Delay patterns are unlikely to be straight lines. The effect "
            "of departure hour probably differs by airport and season, and "
            "trees capture those interactions without us specifying them."
        ),
        "3_preprocessing_needed": (
            "Trees do not need scaling (they split on thresholds, not "
            "distances). We scale anyway so that all three models receive "
            "identical inputs, which Section 8 requires for a fair "
            "comparison."
        ),
        "4_hyperparameters": (
            "max_depth and min_samples_leaf both control overfitting; "
            "n_estimators controls stability. Selected by cross-validation "
            "on the training set only."
        ),
        "5_tradeoffs": (
            "Usually the strongest performer, but the least interpretable. "
            "Its feature-importance scores are biased towards features "
            "with many distinct values, so we report them with that caveat "
            "rather than treating them as a ranking of true importance."
        ),
        "6_vs_baseline": "To be filled in from your results.",
    },
}


if __name__ == "__main__":
    print("=" * 70)
    print("MODELS DEFINED IN THIS PROJECT")
    print("=" * 70)

    print("\nBASELINES (the scores every real model must beat):")
    for name, model in get_baselines().items():
        print(f"\n  {name}")
        print(f"    {MODEL_DESCRIPTIONS[name]}")
        print(f"    sklearn object: {model}")

    print("\n\nMODELS:")
    for name, model in get_models().items():
        print(f"\n  {name}")
        print(f"    {MODEL_DESCRIPTIONS[name]}")
        print(f"    sklearn object: {model}")

    print("\n\nSETTINGS THAT TUNING WILL TRY:")
    for name, grid in config.PARAM_GRIDS.items():
        print(f"  {name}: {grid}")
