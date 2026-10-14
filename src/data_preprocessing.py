
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "customer_churn_advanced_dataset.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "02_cleaned_dataset.csv"
AUDIT_PATH = ROOT / "reports" / "data_audit.json"

TARGET = "churn"
IDENTIFIER = "customer_id"

NUMERIC_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "complaints_last_12m",
    "late_payments_last_12m",
    "satisfaction_score",
    "monthly_usage_gb",
    "discount_pct",
    "senior_citizen",
]

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
    "region",
    "payment_method",
    "paperless_billing",
    "dependents",
    "support_plan",
    "tech_support",
]


def load_data(path=RAW_PATH):
    """Load the original customer dataset."""
    return pd.read_csv(path)


def validate_data(df):
    """Validate essential dataset rules before modeling."""
    required = (
        set(NUMERIC_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | {TARGET, IDENTIFIER}
    )

    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    if df[IDENTIFIER].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if df[IDENTIFIER].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df[TARGET].isna().any():
        raise ValueError("Missing target values detected.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Churn target must contain only 0 and 1.")

    numeric = df[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Detect unexpected nonnumeric entries.
    for column in NUMERIC_COLUMNS:
        invalid = df[column].notna() & numeric[column].isna()
        if invalid.any():
            raise ValueError(
                f"Non-numeric values found in {column}."
            )

    if (numeric["age"].dropna() < 0).any():
        raise ValueError("Negative ages detected.")

    if (numeric["tenure_months"].dropna() < 0).any():
        raise ValueError("Negative tenure detected.")

    nonnegative_columns = [
        "monthly_charges",
        "total_charges",
        "complaints_last_12m",
        "late_payments_last_12m",
        "monthly_usage_gb",
    ]

    for column in nonnegative_columns:
        if (numeric[column].dropna() < 0).any():
            raise ValueError(
                f"Negative values detected in {column}."
            )

    satisfaction = numeric["satisfaction_score"].dropna()
    if not satisfaction.between(1, 10).all():
        raise ValueError(
            "Satisfaction scores must be between 1 and 10."
        )

    discounts = numeric["discount_pct"].dropna()
    if not discounts.between(0, 100).all():
        raise ValueError("Discount percentage out of range.")

    if not numeric["senior_citizen"].dropna().isin([0, 1]).all():
        raise ValueError("Invalid senior_citizen values.")


def audit_data(df):
    """Produce a reproducible data-quality report."""
    numeric = df.select_dtypes(include="number")

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_types": {
            k: str(v) for k, v in df.dtypes.items()
        },
        "missing_by_column": {
            k: int(v) for k, v in df.isna().sum().items()
        },
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(
            df[IDENTIFIER].duplicated().sum()
        ),
        "target_distribution": {
            str(k): int(v)
            for k, v in df[TARGET].value_counts().items()
        },
        "numeric_summary": (
            numeric.describe()
            .replace({np.nan: None})
            .to_dict()
        ),
        "cleaning_policy": [
            "Preserve the raw CSV unchanged.",
            "Do not remove customers solely for missing predictors.",
            "Reject invalid identifiers and target labels.",
            "Preserve legitimate outliers for investigation.",
            "Fit imputation and encoding on training data only.",
            "Exclude customer_id from predictive features.",
        ],
    }


def clean_data(df):
    """
    Apply deterministic cleaning only.

    Do not fit imputers, scalers or encoders here.
    """
    result = df.copy()

    validate_data(result)

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="raise"
        )

    return result


def build_preprocessor():
    """
    Return an unfitted preprocessing transformer.

    It must be fitted only within the training pipeline.
    """
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

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def main():
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_data()
    audit = audit_data(raw_df)
    cleaned_df = clean_data(raw_df)

    cleaned_df.to_csv(CLEAN_PATH, index=False)

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, allow_nan=False)

    print("=" * 55)
    print("CUSTOMER CHURN — DATA QUALITY AUDIT")
    print("=" * 55)
    print(f"Records: {audit['rows']}")
    print(f"Columns: {audit['columns']}")
    print(f"Missing values: {audit['total_missing_cells']}")
    print(f"Duplicate rows: {audit['duplicate_rows']}")
    print(
        "Duplicate IDs:",
        audit["duplicate_customer_ids"],
    )
    print("\nMissing values by column:")
    print(raw_df.isna().sum().to_string())
    print("\nChurn distribution:")
    print(raw_df[TARGET].value_counts().to_string())
    print(f"\nCleaned dataset: {CLEAN_PATH}")
    print(f"Audit report: {AUDIT_PATH}")
    print("\nData audit completed successfully.")


if __name__ == "__main__":
    main()
