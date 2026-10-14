
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_COLUMNS = [
    "tenure_years",
    "complaints_per_year",
    "late_payment_risk",
    "charge_usage_ratio",
    "low_satisfaction_flag",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create leakage-safe customer churn features."""

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def transform(self, X):
        self._validate(X)
        result = X.copy()

        tenure = pd.to_numeric(
            result["tenure_months"], errors="coerce"
        )
        complaints = pd.to_numeric(
            result["complaints_last_12m"], errors="coerce"
        )
        late_payments = pd.to_numeric(
            result["late_payments_last_12m"], errors="coerce"
        )
        charges = pd.to_numeric(
            result["monthly_charges"], errors="coerce"
        )
        usage = pd.to_numeric(
            result["monthly_usage_gb"], errors="coerce"
        )
        satisfaction = pd.to_numeric(
            result["satisfaction_score"], errors="coerce"
        )

        # Feature 1: Tenure expressed in years
        result["tenure_years"] = tenure / 12.0

        # Feature 2: Annualized complaint intensity
        # Avoid division by zero for new customers
        active_years = (tenure / 12.0).clip(lower=1.0)
        result["complaints_per_year"] = (
            complaints / active_years
        )

        # Feature 3: Repeated late payments
        result["late_payment_risk"] = (
            late_payments >= 2
        ).astype(float)
        result.loc[
            late_payments.isna(), "late_payment_risk"
        ] = np.nan

        # Feature 4: Monthly charge per usage unit
        result["charge_usage_ratio"] = (
            charges / (usage + 1.0)
        )

        # Feature 5: Low satisfaction indicator
        result["low_satisfaction_flag"] = (
            satisfaction <= 4
        ).astype(float)
        result.loc[
            satisfaction.isna(), "low_satisfaction_flag"
        ] = np.nan

        return result

    @staticmethod
    def _validate(X):
        required = [
            "tenure_months",
            "complaints_last_12m",
            "late_payments_last_12m",
            "monthly_charges",
            "monthly_usage_gb",
            "satisfaction_score",
        ]

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )
