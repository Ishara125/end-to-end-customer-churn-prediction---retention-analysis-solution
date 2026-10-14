
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from src.modeling import DATA_PATH, SPLIT_PATH, SEED


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "03_model_artifact.joblib"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

IMPORTANCE_PATH = REPORT_DIR / "global_feature_importance.csv"
EXPLANATIONS_PATH = REPORT_DIR / "customer_explanations.json"


def python_value(value):
    """Convert NumPy values for JSON output."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_reference_values(X_train):
    """
    Typical training value for each feature.

    Numerical: training median.
    Categorical: training mode.

    These references are not causal interventions.
    """
    references = {}

    for column in X_train.columns:
        series = X_train[column]

        if pd.api.types.is_numeric_dtype(series):
            references[column] = float(series.median())
        else:
            mode = series.mode(dropna=True)
            if mode.empty:
                raise ValueError(
                    f"No reference value for {column}"
                )
            references[column] = mode.iloc[0]

    return references


def explain_customer(pipeline, customer, references):
    """
    Model sensitivity analysis:

    Replace one feature at a time with its training
    reference and measure change in predicted risk.

    This is not SHAP or a causal explanation.
    """
    original_probability = float(
        pipeline.predict_proba(customer)[0, 1]
    )

    drivers = []

    for column in customer.columns:
        modified = customer.copy()
        modified[column] = references[column]

        alternative_probability = float(
            pipeline.predict_proba(modified)[0, 1]
        )

        drivers.append({
            "feature": column,
            "original_value": python_value(
                customer.iloc[0][column]
            ),
            "reference_value": python_value(
                references[column]
            ),
            "original_probability": original_probability,
            "reference_probability": alternative_probability,
            "risk_difference": (
                original_probability
                - alternative_probability
            ),
        })

    drivers.sort(
        key=lambda item: abs(item["risk_difference"]),
        reverse=True,
    )

    return drivers


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, encoding="utf-8") as file:
        split = json.load(file)

    if split["random_seed"] != SEED:
        raise ValueError("Split seed does not match.")

    train_idx = split["train_indices"]
    test_idx = split["test_indices"]

    if (
        set(train_idx) & set(test_idx)
        or len(set(train_idx + test_idx)) != len(df)
    ):
        raise ValueError("Invalid train/test split.")

    X = df.drop(columns=["customer_id", "churn"])

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = df.iloc[test_idx]["churn"]

    print("=" * 60)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    print(f"Model: {artifact['model_name']}")

    # GLOBAL EXPLANATIONS
    # Holdout permutation importance is descriptive only.
    # Do not use it to reselect or retune the model.
    print("\nCalculating permutation importance...")

    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=SEED,
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values(
        "importance_mean", ascending=False
    )

    importance_df.to_csv(
        IMPORTANCE_PATH, index=False
    )

    top = importance_df.head(12).sort_values(
        "importance_mean"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=3,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Decrease in Average Precision When Permuted")
    plt.title("Global Feature Importance")
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "14_global_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nTop global features:")
    print(importance_df.head(10).to_string(index=False))

    # INDIVIDUAL EXPLANATIONS
    print("\nExplaining individual customer predictions...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    ranking = np.argsort(probabilities, kind="stable")

    # Relative low, medium and high risks on this dataset.
    positions = {
        "low_risk": int(ranking[0]),
        "medium_risk": int(ranking[len(ranking) // 2]),
        "high_risk": int(ranking[-1]),
    }

    references = get_reference_values(X_train)
    explanations = {}

    for category, position in positions.items():
        customer = X_test.iloc[[position]]
        customer_id = df.iloc[test_idx[position]][
            "customer_id"
        ]

        probability = float(probabilities[position])
        actual = int(y_test.iloc[position])

        drivers = explain_customer(
            pipeline, customer, references
        )

        explanations[category] = {
            "customer_id": python_value(customer_id),
            "predicted_churn_probability": probability,
            "actual_churn": actual,
            "explanation_method": (
                "One-feature-at-a-time replacement "
                "with training reference values"
            ),
            "top_drivers": drivers[:8],
        }

        print(f"\n{category.upper()}")
        print(f"Customer ID: {customer_id}")
        print(f"Churn probability: {probability:.4f}")
        print(f"Actual churn: {actual}")
        print("Top sensitivity factors:")

        for driver in drivers[:5]:
            print(
                f"  {driver['feature']}: "
                f"{driver['risk_difference']:+.4f}"
            )

        # Individual sensitivity chart
        top_drivers = sorted(
            drivers[:8],
            key=lambda x: x["risk_difference"]
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            [d["feature"] for d in top_drivers],
            [d["risk_difference"] for d in top_drivers],
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel(
            "Predicted Risk Difference vs Training Reference"
        )
        plt.title(
            f"{category.replace('_', ' ').title()} "
            f"Customer — Probability {probability:.2%}"
        )
        plt.tight_layout()

        plt.savefig(
            FIG_DIR / f"15_{category}_explanation.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    output = {
        "model": artifact["model_name"],
        "global_method": "Permutation importance",
        "global_scoring": "average_precision",
        "individual_method": (
            "Single-feature replacement sensitivity"
        ),
        "limitations": [
            "These analyses describe predictive behavior, "
            "not causal effects.",
            "Correlated variables can affect permutation "
            "importance.",
            "Replacing one feature can create combinations "
            "that are uncommon in real customers.",
            "Engineered features may change together with "
            "their source variables inside the pipeline.",
            "Risk categories are relative to test predictions "
            "and do not imply clinical or financial thresholds.",
        ],
        "customers": explanations,
    }

    with open(
        EXPLANATIONS_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=2)

    print("\nSaved:", IMPORTANCE_PATH)
    print("Saved:", EXPLANATIONS_PATH)
    print("\nExplainability analysis completed.")


if __name__ == "__main__":
    main()
