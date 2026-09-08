import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from src.data_loading import get_prepared_data
from src.split import get_train_test
from src.preprocessing import prepare_features
from src.leakage_check import check_no_leakage
from src.models import get_baselines, get_models, MODEL_DESCRIPTIONS


def train_baselines(X_train, y_train):
    print("\n" + "-" * 70)
    print("TRAINING BASELINES  (requirement DW-7)")
    print("-" * 70)

    trained = {}
    for name, model in get_baselines().items():
        print(f"\n  {name}")
        print(f"    {MODEL_DESCRIPTIONS[name]}")

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        trained[name] = model
        print(f"    Trained in {elapsed:.3f} seconds.")

    return trained


def tune_model(name, model, param_grid, X_train, y_train):
    print(f"\n  {name}")
    print(f"    {MODEL_DESCRIPTIONS[name]}")
    print(f"    Settings to try: {param_grid}")

    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    n_fits = n_combinations * config.CV_FOLDS

    print(f"    That is {n_combinations} combination(s) x {config.CV_FOLDS} folds "
          f"= {n_fits} training runs.")

    cv_strategy = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=config.TUNING_SCORE,
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Best settings : {search.best_params_}")
    print(f"    Best CV score : {search.best_score_:.4f} ({config.TUNING_SCORE})")
    print(f"    Search took   : {elapsed:.1f} seconds")

    best_index = search.best_index_
    train_score = search.cv_results_["mean_train_score"][best_index]
    val_score = search.cv_results_["mean_test_score"][best_index]
    gap = train_score - val_score

    print(f"    Train score   : {train_score:.4f}")
    print(f"    Validation    : {val_score:.4f}")
    print(f"    Gap           : {gap:.4f}", end="")

    if gap > 0.15:
        print("   <-- LARGE GAP: this model is overfitting.")
        print("                     Consider reducing max_depth or raising")
        print("                     min_samples_leaf. Report this either way.")
    else:
        print("   (small gap: generalising acceptably)")

    return search, elapsed


def train_all_models(X_train, y_train):
    print("\n" + "-" * 70)
    print("TUNING AND TRAINING MODELS  (requirement DW-8)")
    print("-" * 70)
    print(f"\n  Tuning by {config.CV_FOLDS}-fold cross-validation on TRAINING data.")
    print(f"  Scoring metric: {config.TUNING_SCORE}")
    print("  The test set is NOT touched anywhere in this file.")

    trained = {}
    log_rows = []

    for name, model in get_models().items():
        param_grid = config.PARAM_GRIDS.get(name, {})

        search, elapsed = tune_model(name, model, param_grid, X_train, y_train)

        trained[name] = search.best_estimator_

        results = search.cv_results_
        for i in range(len(results["params"])):
            log_rows.append({
                "model": name,
                "params": str(results["params"][i]),
                "mean_cv_score": round(results["mean_test_score"][i], 4),
                "std_cv_score": round(results["std_test_score"][i], 4),
                "mean_train_score": round(results["mean_train_score"][i], 4),
                "fit_time_seconds": round(results["mean_fit_time"][i], 3),
                "is_best": (i == search.best_index_),
            })

    tuning_log = pd.DataFrame(log_rows)
    return trained, tuning_log


def run_training(X_train, y_train, save=True):
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Features     : {X_train.shape[1]}")
    print(f"Random seed  : {config.RANDOM_STATE}")

    baselines = train_baselines(X_train, y_train)
    models, tuning_log = train_all_models(X_train, y_train)

    all_models = {**baselines, **models}

    if save:
        config.make_folders()
        out = config.METRICS_DIR / "tuning_log.csv"
        tuning_log.to_csv(out, index=False)
        print(f"\n[SAVED] Tuning log: {out}")
        print("        Every combination tried, including the ones that did")
        print("        badly. Section 14 forbids hiding failed experiments.")

    print("\n" + "-" * 70)
    print("EVERY COMBINATION TRIED (cross-validation scores)")
    print("-" * 70)
    print(tuning_log.to_string(index=False))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  {len(all_models)} models ready: {list(all_models.keys())}")
    print("  The test set has still not been touched.")
    print("=" * 70)

    return all_models, tuning_log


if __name__ == "__main__":
    config.print_config()

    data, _ = get_prepared_data()
    train, test, _ = get_train_test(data)
    prepared = prepare_features(train, test)
    check_no_leakage(train, test, prepared)

    run_training(prepared["X_train"], prepared["y_train"])
