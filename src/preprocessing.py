"""
preprocessing.py
================
Turns the raw table into numbers the models can actually use.

THE GOLDEN RULE OF THIS FILE
----------------------------
    Every transformation LEARNS its settings from the TRAINING data only,
    and then APPLIES those settings to both training and test data.

    Say that again, because it is the whole point:

        LEARN from train.        APPLY to train and test.

    If we learned anything from the test data -- even something as small as
    the average flight distance -- then information from the test set has
    quietly entered our model, and requirement FR-2 is broken.

WHY WE WROTE THIS BY HAND INSTEAD OF USING sklearn's Pipeline
-------------------------------------------------------------
    scikit-learn has a `Pipeline` object that enforces this rule for you.
    Requirement DW-6 says to use one "where practical".

    We deliberately chose the manual version so that every member can SEE
    the fit/transform split happening, line by line, and explain it in the
    viva. The trade-off is that correctness is now our responsibility
    rather than the library's.

    To cover that trade-off we wrote leakage_check.py, which PROVES with
    automatic tests that the rule was followed. Discuss this decision in
    report section 5.

THE TWO TRANSFORMATIONS WE APPLY
--------------------------------
    1. SCALING (for numeric columns like DISTANCE)
       Distance is in the thousands; DEP_HOUR is 0-23. K-Nearest Neighbours
       measures distance between rows, so without scaling the DISTANCE
       column would drown out every other feature. Scaling puts them all on
       a comparable footing.

    2. ONE-HOT ENCODING (for text columns like ORIGIN)
       A model cannot do arithmetic on the word "JFK". So we replace one
       ORIGIN column with many yes/no columns: IS_IT_JFK, IS_IT_LAX, etc.

WORKFLOW STAGE COVERED: DW-6
OWNER: Member 2

HOW TO RUN:
    python -m src.preprocessing
"""

import numpy as np
import pandas as pd

import config
from src.data_loading import get_prepared_data
from src.split import get_train_test



def learn_scaling(train_df, columns):
    """
    LEARNS the scaling settings. Looks at TRAINING DATA ONLY.

    For each column we record two numbers:
        mean               - the average value
        standard deviation - how spread out the values are

    Later we use them to convert every value with this formula:

        scaled_value = (original_value - mean) / standard_deviation

    After that, a typical column has an average of 0 and a spread of 1,
    so no single column can dominate just because its numbers are bigger.

    Parameters
    ----------
    train_df : pandas.DataFrame
        TRAINING data only. Never pass test data to this function.
    columns : list of str
        Which numeric columns to scale.

    Returns
    -------
    dict
        The learned settings, e.g. {"DISTANCE": {"mean": 812.4, "std": 601.2}}
    """
    print("\n[SCALING] Learning mean and standard deviation from TRAINING data only...")

    settings = {}
    for column in columns:
        mean = train_df[column].mean()
        std = train_df[column].std()

        if std == 0 or pd.isna(std):
            print(f"    WARNING: {column} never changes. Skipping its scaling.")
            std = 1.0

        settings[column] = {"mean": mean, "std": std}
        print(f"    {column:<20} mean={mean:>10.2f}   std={std:>10.2f}")

    print("    These numbers came from TRAINING rows only. The test set was not touched.")
    return settings


def apply_scaling(df, settings):
    """
    APPLIES previously-learned scaling settings to a table.

    We call this twice: once for training data, once for test data --
    but both times using the SAME settings, learned from training.

    That is the correct behaviour. The test data is transformed using the
    training set's idea of "average", exactly as it would be in real life
    when a brand-new flight arrives and we have no idea what the future
    average will be.
    """
    df = df.copy()

    for column, values in settings.items():
        df[column] = (df[column] - values["mean"]) / values["std"]

    return df



def learn_categories(train_df, columns, top_n_settings):
    """
    LEARNS which category values are common enough to keep.
    Looks at TRAINING DATA ONLY.

    THE PROBLEM: there are hundreds of airports. Giving each one its own
    column would create a huge, slow, almost-empty table, and the rare
    airports would have far too few flights to learn anything from.

    THE SOLUTION: keep the busiest N of each column and lump the rest into
    a single bucket called "OTHER".

    WHY THIS MUST USE TRAINING DATA ONLY: deciding "the 30 busiest
    airports" by looking at all the data would let the test set influence
    the shape of our features. That is leakage, even though it feels
    harmless.
    """
    print("\n[ENCODING] Learning which categories to keep, from TRAINING data only...")

    vocabularies = {}
    for column in columns:
        top_n = top_n_settings.get(column, 20)

        most_common = train_df[column].value_counts().head(top_n).index.tolist()

        vocabularies[column] = most_common

        n_total = train_df[column].nunique()
        n_kept = len(most_common)
        coverage = 100 * train_df[column].isin(most_common).mean()

        print(f"    {column:<12} keeping {n_kept} of {n_total} values "
              f"-> covers {coverage:.1f}% of training rows")

    print("    Anything outside these lists becomes 'OTHER'.")
    return vocabularies


def apply_encoding(df, vocabularies):
    """
    APPLIES the learned category lists and creates the yes/no columns.

    EXAMPLE. If the learned vocabulary for ORIGIN is ["JFK", "LAX"], then:

        original row      becomes
        ------------      ------------------------------------------
        ORIGIN = "JFK"    ORIGIN_JFK=1, ORIGIN_LAX=0, ORIGIN_OTHER=0
        ORIGIN = "LAX"    ORIGIN_JFK=0, ORIGIN_LAX=1, ORIGIN_OTHER=0
        ORIGIN = "BOS"    ORIGIN_JFK=0, ORIGIN_LAX=0, ORIGIN_OTHER=1

    Notice the last one. "BOS" was not in the training vocabulary, so it
    safely lands in OTHER instead of crashing. This is important: the test
    set will contain airports the training set never saw, and our code must
    survive that.

    WHY WE BUILD THE COLUMNS BY HAND rather than using pd.get_dummies():
        pd.get_dummies() creates columns based on whatever values happen to
        be in the table it is given. Run it on train and test separately and
        you get different columns, and the model breaks. Building from a
        FIXED learned vocabulary guarantees both tables come out identical
        in shape.
    """
    df = df.copy()
    new_columns = {}

    for column, vocabulary in vocabularies.items():
        for category in vocabulary:
            new_name = f"{column}_{category}"
            new_columns[new_name] = (df[column] == category).astype(int)

        other_name = f"{column}_{config.OTHER_CATEGORY_LABEL}"
        new_columns[other_name] = (~df[column].isin(vocabulary)).astype(int)

    encoded = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, encoded], axis=1)

    df = df.drop(columns=list(vocabularies.keys()))

    return df



def learn_imputation(train_df, columns):
    """
    LEARNS the value to use for filling gaps. TRAINING DATA ONLY.

    We use the MEDIAN (the middle value when sorted) rather than the mean
    (the average), because the median is not dragged around by a few
    extreme values. One 14-hour flight would pull the mean upward; the
    median barely notices.
    """
    print("\n[MISSING VALUES] Learning fill values from TRAINING data only...")

    fill_values = {}
    for column in columns:
        median = train_df[column].median()
        fill_values[column] = median

        n_missing_train = train_df[column].isna().sum()
        print(f"    {column:<20} median={median:>10.2f}   "
              f"({n_missing_train:,} missing in training data)")

    return fill_values


def apply_imputation(df, fill_values):
    """Fills gaps using the previously-learned median values."""
    df = df.copy()
    for column, value in fill_values.items():
        df[column] = df[column].fillna(value)
    return df



def prepare_features(train_df, test_df):
    """
    Runs the full preprocessing, in the correct order, on both tables.

    THE ORDER MATTERS AND IT IS ALWAYS THE SAME:

        1. LEARN everything from training data      (fit)
        2. APPLY it to training data                (transform)
        3. APPLY the SAME settings to test data     (transform)

    Step 3 never learns anything new. That is what keeps us honest.

    Returns
    -------
    dict
        X_train, y_train, X_test, y_test, the learned settings, and the
        feature names -- everything the modelling files need.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING  (requirement DW-6)")
    print("=" * 70)
    print("\nRULE: learn from TRAIN only, then apply to BOTH.")

    numeric = config.NUMERIC_FEATURES
    categorical = config.CATEGORICAL_FEATURES

    print("\n" + "-" * 70)
    print("STEP 1: LEARN the settings  --  looking at TRAINING data only")
    print("-" * 70)

    fill_values = learn_imputation(train_df, numeric)
    vocabularies = learn_categories(train_df, categorical, config.TOP_N_CATEGORIES)

    train_filled = apply_imputation(train_df, fill_values)
    scaling = learn_scaling(train_filled, numeric)

    print("\n" + "-" * 70)
    print("STEP 2: APPLY the settings to TRAINING data")
    print("-" * 70)

    train_processed = apply_imputation(train_df, fill_values)
    train_processed = apply_scaling(train_processed, scaling)
    train_processed = apply_encoding(train_processed, vocabularies)
    print("    Training data transformed.")

    print("\n" + "-" * 70)
    print("STEP 3: APPLY THE SAME settings to TEST data")
    print("-" * 70)
    print("    Nothing new is learned here. We only reuse what step 1 learned.")

    test_processed = apply_imputation(test_df, fill_values)
    test_processed = apply_scaling(test_processed, scaling)
    test_processed = apply_encoding(test_processed, vocabularies)
    print("    Test data transformed using the TRAINING settings.")

    onehot_features = []
    for column in categorical:
        for category in vocabularies[column]:
            onehot_features.append(f"{column}_{category}")
        onehot_features.append(f"{column}_{config.OTHER_CATEGORY_LABEL}")

    feature_names = list(numeric) + onehot_features

    missing = [c for c in feature_names if c not in train_processed.columns]
    if missing:
        raise AssertionError(
            f"These features are named in config.py but are not in the "
            f"processed table: {missing}"
        )

    ignored = [
        c for c in train_processed.columns
        if c not in feature_names and c != config.TARGET_COLUMN
    ]
    print()
    print(f"    Features admitted : {len(feature_names)} "
          f"({len(numeric)} numeric + {len(onehot_features)} one-hot)")
    print(f"    Columns ignored   : {len(ignored)}")
    if ignored:
        print(f"      {ignored}")

    X_train = train_processed[feature_names]
    y_train = train_processed[config.TARGET_COLUMN]

    X_test = test_processed[feature_names]
    y_test = test_processed[config.TARGET_COLUMN]

    print("\n" + "-" * 70)
    print("STEP 5: Sanity checks")
    print("-" * 70)

    if list(X_train.columns) != list(X_test.columns):
        raise AssertionError(
            "X_train and X_test have different columns. The model cannot "
            "handle that. Check the encoding step."
        )
    print("    PASS: training and test tables have identical columns.")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise AssertionError("There are still missing values after preprocessing.")
    print("    PASS: no missing values remain.")

    if config.SOURCE_DELAY_COLUMN in feature_names:
        raise AssertionError(
            f"LEAKAGE: {config.SOURCE_DELAY_COLUMN} is still a feature. "
            f"That column IS the answer."
        )
    print(f"    PASS: {config.SOURCE_DELAY_COLUMN} is not among the features.")

    for column in config.NUMERIC_FEATURES:
        column_mean = abs(X_train[column].mean())
        if column_mean > 1.0:
            raise AssertionError(
                f"'{column}' has a mean of {column_mean:.2f} after scaling. "
                f"Scaled columns should average near 0. It was probably missed "
                f"by the scaling step."
            )
    print("    PASS: all numeric features are scaled (means near 0).")

    print(f"\n    Final shape -> X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"    That is {X_train.shape[1]} features per flight.")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "settings": {
            "fill_values": fill_values,
            "scaling": scaling,
            "vocabularies": vocabularies,
        },
    }


if __name__ == "__main__":
    config.print_config()

    data, _ = get_prepared_data()
    train, test, _ = get_train_test(data)
    result = prepare_features(train, test)

    print("\nFirst 5 feature names:")
    for name in result["feature_names"][:5]:
        print(f"  {name}")
    print(f"  ... and {len(result['feature_names']) - 5} more")
