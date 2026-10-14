
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from src.modeling import (
    DATA_PATH,
    SPLIT_PATH,
    REPORT_DIR,
    SEED,
    create_pipeline,
    get_baseline_models,
)


TUNING_PATH = REPORT_DIR / "hyperparameter_tuning_results.csv"
PARAMS_PATH = REPORT_DIR / "best_hyperparameters.json"
N_ITER = 8


def top20_metrics(estimator, X, y):
    """
    Evaluate the highest-risk 20% of a validation fold.

    Returns recall, precision and lift at 20%.
    """
    probabilities = estimator.predict_proba(X)[:, 1]
    actual = np.asarray(y, dtype=int)

    n = len(actual)
    k = max(1, int(np.ceil(n * 0.20)))

    # Stable order in case probabilities are tied.
    ranking = np.argsort(-probabilities, kind="stable")
    selected = ranking[:k]

    captured = actual[selected].sum()
    total_churners = actual.sum()

    precision = captured / k
    recall = (
        captured / total_churners
        if total_churners > 0 else 0.0
    )

    prevalence = actual.mean()
    lift = (
        precision / prevalence
        if prevalence > 0 else 0.0
    )

    return {
        "top20_precision": float(precision),
        "top20_recall": float(recall),
        "top20_lift": float(lift),
    }


def top20_precision(estimator, X, y):
    return top20_metrics(estimator, X, y)[
        "top20_precision"
    ]


def top20_recall(estimator, X, y):
    return top20_metrics(estimator, X, y)[
        "top20_recall"
    ]


def top20_lift(estimator, X, y):
    return top20_metrics(estimator, X, y)[
        "top20_lift"
    ]


def get_search_spaces():
    return {
        "Logistic Regression": {
            "classifier__C": [
                0.01, 0.1, 0.5, 1, 2, 5, 10
            ],
            "classifier__class_weight": [
                None, "balanced"
            ],
            "classifier__solver": ["liblinear"],
        },
        "Random Forest": {
            "classifier__n_estimators": [
                100, 200, 300
            ],
            "classifier__max_depth": [
                None, 5, 10, 20
            ],
            "classifier__min_samples_split": [
                2, 5, 10
            ],
            "classifier__min_samples_leaf": [
                1, 2, 4
            ],
            "classifier__class_weight": [
                None, "balanced"
            ],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [
                50, 100, 150, 200
            ],
            "classifier__learning_rate": [
                0.01, 0.05, 0.1, 0.2
            ],
            "classifier__max_depth": [
                2, 3, 4
            ],
            "classifier__subsample": [
                0.8, 1.0
            ],
            "classifier__min_samples_leaf": [
                1, 3, 5
            ],
        },
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Run python -m src.modeling first."
        )

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["random_seed"] != SEED:
        raise ValueError("Unexpected split random seed.")

    train_idx = manifest["train_indices"]
    test_idx = manifest["test_indices"]

    if (
        set(train_idx) & set(test_idx)
        or len(train_idx) + len(test_idx) != len(df)
        or len(set(train_idx + test_idx)) != len(df)
    ):
        raise ValueError("Invalid saved train/test partition.")

    # The final test set is never used here.
    train_df = df.iloc[train_idx]

    X_train = train_df.drop(
        columns=["customer_id", "churn"]
    )
    y_train = train_df["churn"].astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "top20_precision": top20_precision,
        "top20_recall": top20_recall,
        "top20_lift": top20_lift,
    }

    results = []
    best_parameters = {}

    print("=" * 65)
    print("HYPERPARAMETER TUNING — CUSTOMER CHURN")
    print("=" * 65)
    print(f"Training records: {len(X_train)}")
    print(f"Reserved test records: {len(test_idx)}")
    print(f"Cross-validation folds: {cv.n_splits}")
    print(f"Random search iterations per model: {N_ITER}")

    models = get_baseline_models()
    search_spaces = get_search_spaces()

    for name, model in models.items():
        print(f"\nTuning {name}...")

        pipeline = create_pipeline(clone(model))

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_spaces[name],
            n_iter=N_ITER,
            scoring="average_precision",
            n_jobs=1,
            cv=cv,
            refit=True,
            random_state=SEED,
            error_score="raise",
            verbose=0,
        )

        search.fit(X_train, y_train)

        # Diagnostic cross-validation of selected settings.
        # These are NOT independent nested-CV estimates.
        selected_scores = cross_validate(
            clone(search.best_estimator_),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "model": name,
            "search_best_pr_auc": float(
                search.best_score_
            ),
        }

        for metric in scoring:
            values = selected_scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(
                np.mean(values)
            )
            row[f"{metric}_std"] = float(
                np.std(values)
            )

        results.append(row)
        best_parameters[name] = search.best_params_

        print(
            f"Best search PR-AUC: "
            f"{search.best_score_:.4f}"
        )
        print(
            f"Selected CV ROC-AUC: "
            f"{row['roc_auc_mean']:.4f}"
        )
        print(
            f"Selected CV top-20% recall: "
            f"{row['top20_recall_mean']:.4f}"
        )
        print(
            f"Selected CV top-20% lift: "
            f"{row['top20_lift_mean']:.4f}"
        )
        print(f"Best parameters: {search.best_params_}")

    comparison = pd.DataFrame(results).sort_values(
        "search_best_pr_auc",
        ascending=False,
    )

    comparison.to_csv(
        TUNING_PATH, index=False
    )

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_seed": SEED,
                "n_iter": N_ITER,
                "cv_folds": 5,
                "selection_metric": "average_precision",
                "best_parameters": best_parameters,
                "interpretation": (
                    "Scores are training-data CV diagnostics. "
                    "Parameter selection uses the same training "
                    "data, so they are not unbiased nested-CV "
                    "performance estimates. The final test set "
                    "remains reserved."
                ),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("TUNED MODEL COMPARISON")
    print("=" * 65)
    print(
        comparison[
            [
                "model",
                "search_best_pr_auc",
                "roc_auc_mean",
                "top20_precision_mean",
                "top20_recall_mean",
                "top20_lift_mean",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved: {TUNING_PATH}")
    print(f"Saved: {PARAMS_PATH}")
    print("\nFinal test set remains untouched.")
    print("Hyperparameter tuning completed.")


if __name__ == "__main__":
    main()
