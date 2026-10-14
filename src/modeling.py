
from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from src.feature_engineering import (
    ChurnFeatureEngineer,
    ENGINEERED_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "02_cleaned_dataset.csv"
REPORT_DIR = ROOT / "reports"
SPLIT_PATH = ROOT / "data" / "processed" / "split_indices.json"
SEED = 42


def create_pipeline(estimator):
    """Create leakage-safe feature and model pipeline."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_COLUMNS + ENGINEERED_COLUMNS,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS,
        ),
    ])

    return Pipeline([
        ("features", ChurnFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def get_baseline_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=SEED,
        ),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Never include ID or target in predictors
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"].astype(int)

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Reserve final test set for later milestones
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "random_seed": SEED,
            "test_size": 0.20,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }, f, indent=2)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    print("=" * 60)
    print("CUSTOMER CHURN — BASELINE MODEL COMPARISON")
    print("=" * 60)
    print(f"Training records: {len(X_train)}")
    print(f"Reserved test records: {len(test_idx)}")
    print(f"Training churn rate: {y_train.mean():.2%}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = []

    for name, estimator in get_baseline_models().items():
        print(f"\nEvaluating: {name}")

        pipeline = create_pipeline(estimator)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(
                float(values.mean()), 4
            )
            row[f"{metric}_std"] = round(
                float(values.std()), 4
            )

        results.append(row)

        print(f"PR-AUC: {row['pr_auc_mean']:.4f}")
        print(f"ROC-AUC: {row['roc_auc_mean']:.4f}")
        print(f"F1: {row['f1_mean']:.4f}")
        print(f"Precision: {row['precision_mean']:.4f}")
        print(f"Recall: {row['recall_mean']:.4f}")

    comparison = pd.DataFrame(results).sort_values(
        "pr_auc_mean", ascending=False
    )

    output = REPORT_DIR / "baseline_model_comparison.csv"
    comparison.to_csv(output, index=False)

    print("\nBASELINE MODEL COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {output}")
    print("\nMilestone 4 execution completed.")


if __name__ == "__main__":
    main()
