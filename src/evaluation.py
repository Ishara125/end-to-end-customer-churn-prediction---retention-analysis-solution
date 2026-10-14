
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src.modeling import (
    DATA_PATH,
    SPLIT_PATH,
    REPORT_DIR,
    SEED,
    create_pipeline,
    get_baseline_models,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "03_model_artifact.joblib"
FIG_DIR = REPORT_DIR / "figures"

PARAMS_PATH = REPORT_DIR / "best_hyperparameters.json"
METRICS_PATH = REPORT_DIR / "final_evaluation_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def choose_threshold(y_true, probabilities):
    """
    Business assumption:
    Require precision >= 0.90 on out-of-fold predictions,
    then maximize recall.

    Tie-break toward higher F1 and then higher threshold.

    This is separate from the top-20% contact budget.
    """
    records = []

    for threshold in np.round(
        np.arange(0.05, 0.951, 0.01), 2
    ):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        records.append({
            "threshold": float(threshold),
            "precision": float(
                precision_score(
                    y_true, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true, predictions, zero_division=0
                )
            ),
        })

    results = pd.DataFrame(records)
    eligible = results[results["precision"] >= 0.90]

    if eligible.empty:
        raise ValueError(
            "No threshold meets the 90% precision requirement."
        )

    selected = eligible.sort_values(
        ["recall", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(selected["threshold"]), results


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    with open(SPLIT_PATH, encoding="utf-8") as file:
        split = json.load(file)

    with open(PARAMS_PATH, encoding="utf-8") as file:
        tuning = json.load(file)

    train_idx = split["train_indices"]
    test_idx = split["test_indices"]

    if (
        split["random_seed"] != SEED
        or set(train_idx) & set(test_idx)
        or len(set(train_idx + test_idx)) != len(df)
    ):
        raise ValueError("Invalid train/test split manifest.")

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"].astype(int)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print("=" * 65)
    print("CUSTOMER CHURN — FINAL MODEL EVALUATION")
    print("=" * 65)

    print(f"Training size: {len(X_train)}")
    print(f"Reserved test size: {len(X_test)}")
    print(f"Training churn prevalence: {y_train.mean():.4f}")

    selected_model_name = "Gradient Boosting"

    params = tuning["best_parameters"][selected_model_name]

    estimator = get_baseline_models()[selected_model_name]
    pipeline = create_pipeline(estimator)
    pipeline.set_params(**params)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    # Generate out-of-fold probabilities using training data only.
    print("\nGenerating training out-of-fold predictions...")

    oof_probabilities = cross_val_predict(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    selected_threshold, threshold_table = choose_threshold(
        y_train, oof_probabilities
    )

    threshold_table.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False,
    )

    print(f"Selected threshold: {selected_threshold:.2f}")

    # Threshold performance plot
    plt.figure(figsize=(9, 5))
    plt.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        label="Precision",
    )
    plt.plot(
        threshold_table["threshold"],
        threshold_table["recall"],
        label="Recall",
    )
    plt.plot(
        threshold_table["threshold"],
        threshold_table["f1"],
        label="F1",
    )
    plt.axvline(
        selected_threshold,
        linestyle="--",
        color="black",
        label="Selected threshold",
    )
    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Training CV: Precision, Recall and F1")
    plt.legend()
    save_figure("09_threshold_analysis.png")

    # Probability calibration on OOF predictions
    observed, predicted = calibration_curve(
        y_train,
        oof_probabilities,
        n_bins=8,
        strategy="quantile",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.plot(
        predicted,
        observed,
        marker="o",
        label="Gradient Boosting (OOF)",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Churn Frequency")
    plt.title("Out-of-Fold Calibration Curve")
    plt.legend()
    save_figure("10_calibration_curve.png")

    # Fold-to-fold robustness
    cv_scores = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "pr_auc": "average_precision",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
    )

    robustness = {
        metric: {
            "mean": float(
                np.mean(cv_scores[f"test_{metric}"])
            ),
            "std": float(
                np.std(cv_scores[f"test_{metric}"])
            ),
        }
        for metric in ["pr_auc", "roc_auc"]
    }

    training_metrics = calculate_metrics(
        y_train,
        oof_probabilities,
        selected_threshold,
    )

    print("\nTraining OOF performance:")
    print(f"PR-AUC: {training_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {training_metrics['roc_auc']:.4f}")
    print(f"F1: {training_metrics['f1']:.4f}")

    # Only now fit the chosen pipeline on all training data.
    print("\nFitting selected model on full training set...")
    pipeline.fit(X_train, y_train)

    # Final untouched test set: evaluate only after decisions.
    print("Evaluating reserved test set...")

    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print("\nFINAL TEST RESULTS")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        print(f"{key}: {test_metrics[key]:.4f}")

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test, test_probabilities
    )
    plt.title("Final Test ROC Curve")
    save_figure("11_test_roc_curve.png")

    # Precision-recall curve
    PrecisionRecallDisplay.from_predictions(
        y_test, test_probabilities
    )
    plt.title("Final Test Precision-Recall Curve")
    save_figure("12_test_precision_recall.png")

    # Confusion matrix
    predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Retained", "Churned"],
        cmap="Blues",
    )
    plt.title("Final Test Confusion Matrix")
    save_figure("13_test_confusion_matrix.png")

    # Keep these predictions for later retention simulation.
    test_results = pd.DataFrame({
        "customer_id": df.iloc[test_idx][
            "customer_id"
        ].to_numpy(),
        "actual_churn": y_test.to_numpy(),
        "predicted_probability": test_probabilities,
        "predicted_churn": predictions,
    })

    test_results.to_csv(
        PREDICTIONS_PATH, index=False
    )

    # Save the complete fitted pipeline and operating threshold.
    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": selected_threshold,
            "model_name": selected_model_name,
            "feature_columns": X_train.columns.tolist(),
            "random_seed": SEED,
        },
        MODEL_PATH,
    )

    output_metrics = {
        "model_name": selected_model_name,
        "selection_basis": (
            "Highest PR-AUC in the previous "
            "training-only hyperparameter search"
        ),
        "threshold_policy": (
            "Maximize training OOF recall while "
            "requiring precision >= 0.90"
        ),
        "selected_threshold": selected_threshold,
        "train_oof_metrics": training_metrics,
        "test_metrics": test_metrics,
        "robustness": robustness,
        "imbalance_strategy": (
            "Stratified CV; compared class-weight options "
            "during tuning. Selected Gradient Boosting "
            "without resampling."
        ),
        "limitations": [
            "Model selection uses non-nested cross-validation.",
            "OOF threshold performance is optimistic because "
            "the model configuration was previously selected "
            "using the same training records.",
            "Calibration is evaluated, not automatically corrected.",
            "Final test metrics describe discrimination and "
            "prediction, not causal retention effects.",
        ],
    }

    with open(
        METRICS_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(output_metrics, file, indent=2)

    print("\nSaved final model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved test predictions:", PREDICTIONS_PATH)
    print("\nMilestone 6 evaluation completed.")


if __name__ == "__main__":
    main()
