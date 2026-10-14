
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import GradientBoostingClassifier

from src.data_preprocessing import (
    load_data,
    clean_data,
    validate_data,
)
from src.feature_engineering import (
    ChurnFeatureEngineer,
    ENGINEERED_COLUMNS,
)
from src.modeling import create_pipeline
from src.retention_analysis import (
    top_k,
    financial_simulation,
    validate_predictions,
)


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    ROOT / "data/raw/customer_churn_advanced_dataset.csv"
)

CLEAN_PATH = (
    ROOT / "data/processed/02_cleaned_dataset.csv"
)

MODEL_PATH = ROOT / "models/03_model_artifact.joblib"


@pytest.fixture(scope="module")
def data():
    return load_data(RAW_PATH)


@pytest.fixture(scope="module")
def cleaned():
    return pd.read_csv(CLEAN_PATH)


def test_dataset_loads(data):
    assert not data.empty
    assert "churn" in data.columns
    assert "customer_id" in data.columns


def test_cleaning_preserves_records(data):
    result = clean_data(data)
    assert len(result) == len(data)
    assert set(result.columns) == set(data.columns)


def test_customer_ids_unique(data):
    assert data["customer_id"].is_unique
    assert not data["customer_id"].isna().any()


def test_target_is_binary(data):
    assert data["churn"].isin([0, 1]).all()


def test_invalid_target_rejected(data):
    invalid = data.copy()
    invalid.loc[invalid.index[0], "churn"] = 3

    with pytest.raises(ValueError):
        validate_data(invalid)


def test_duplicate_customer_id_rejected(data):
    invalid = data.copy()
    invalid.loc[
        invalid.index[1], "customer_id"
    ] = invalid.iloc[0]["customer_id"]

    with pytest.raises(ValueError):
        validate_data(invalid)


def test_cleaned_data_exists(cleaned):
    assert len(cleaned) > 0
    assert "churn" in cleaned.columns


def test_feature_engineering(cleaned):
    X = cleaned.drop(
        columns=["customer_id", "churn"]
    ).head(10)

    transformer = ChurnFeatureEngineer()
    result = transformer.fit_transform(X)

    for column in ENGINEERED_COLUMNS:
        assert column in result.columns

    assert len(result) == len(X)


def test_feature_engineering_zero_tenure(cleaned):
    X = cleaned.drop(
        columns=["customer_id", "churn"]
    ).head(1).copy()

    X["tenure_months"] = 0
    X["monthly_usage_gb"] = 0

    result = ChurnFeatureEngineer().fit_transform(X)

    values = result[ENGINEERED_COLUMNS].to_numpy(
        dtype=float
    )

    assert not np.isinf(values).any()


def test_feature_engineering_does_not_require_target(cleaned):
    X = cleaned.drop(
        columns=["customer_id", "churn"]
    ).head(5)

    result = ChurnFeatureEngineer().fit_transform(X)

    assert "churn" not in result.columns
    assert "customer_id" not in result.columns


def test_pipeline_trains_and_predicts(cleaned):
    X = cleaned.drop(
        columns=["customer_id", "churn"]
    )

    y = cleaned["churn"]

    # Small reproducible subset containing both classes.
    sample_indices = (
        cleaned.groupby("churn")
        .head(15)
        .index
    )

    X_sample = X.loc[sample_indices]
    y_sample = y.loc[sample_indices]

    model = create_pipeline(
        GradientBoostingClassifier(
            n_estimators=10,
            random_state=42,
        )
    )

    model.fit(X_sample, y_sample)

    predictions = model.predict_proba(
        X_sample.head(3)
    )

    assert predictions.shape == (3, 2)
    assert np.isfinite(predictions).all()
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    )


def test_saved_model_loads(cleaned):
    assert MODEL_PATH.exists()

    artifact = joblib.load(MODEL_PATH)

    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "model_name" in artifact

    assert 0 <= artifact["threshold"] <= 1

    X = cleaned.drop(
        columns=["customer_id", "churn"]
    ).head(3)

    probabilities = artifact[
        "pipeline"
    ].predict_proba(X)[:, 1]

    assert len(probabilities) == 3
    assert np.isfinite(probabilities).all()
    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )


def test_top20_capacity():
    assert top_k(240) == 48
    assert top_k(101) == 21
    assert top_k(5) == 1


def test_invalid_contact_fraction():
    with pytest.raises(ValueError):
        top_k(100, fraction=0)

    with pytest.raises(ValueError):
        top_k(100, fraction=1.5)


def test_financial_calculations():
    result = financial_simulation(
        contacted=48,
        actual_churners_reached=48,
        contact_cost=5,
        retained_value=200,
        success_rate=0.15,
    )

    assert result["expected_retained_customers"] == pytest.approx(7.2)
    assert result["campaign_cost"] == pytest.approx(240)
    assert result["expected_benefit"] == pytest.approx(1440)
    assert result["expected_net_benefit"] == pytest.approx(1200)


def test_invalid_financial_assumptions():
    with pytest.raises(ValueError):
        financial_simulation(
            contacted=10,
            actual_churners_reached=11,
        )

    with pytest.raises(ValueError):
        financial_simulation(
            contacted=10,
            actual_churners_reached=5,
            success_rate=1.5,
        )


def test_prediction_data_validation():
    predictions = pd.read_csv(
        ROOT / "reports/test_predictions.csv"
    )

    validate_predictions(predictions)

    assert not predictions.empty


def test_retention_results_consistent():
    predictions = pd.read_csv(
        ROOT / "reports/test_predictions.csv"
    )

    with open(
        ROOT / "reports/retention_simulation.json",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    k = top_k(len(predictions))

    selected = predictions.sort_values(
        ["predicted_probability", "customer_id"],
        ascending=[False, True],
        kind="stable",
    ).head(k)

    captured = int(selected["actual_churn"].sum())

    assert summary["contact_capacity"] == k
    assert (
        summary["targeting"]["actual_churners_captured"]
        == captured
    )

    assert summary["targeting"][
        "precision_at_20"
    ] == pytest.approx(captured / k)


def test_required_deliverables_exist():
    required = [
        ROOT / "notebooks/01_data_science_notebook.ipynb",
        ROOT / "data/processed/02_cleaned_dataset.csv",
        MODEL_PATH,
        ROOT / "reports/04_business_report.pdf",
        ROOT / "05_README.md",
        ROOT / "README.md",
        ROOT / "requirements.txt",
    ]

    for path in required:
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0


def test_notebook_executed():
    import nbformat

    path = ROOT / "notebooks/01_data_science_notebook.ipynb"
    notebook = nbformat.read(path, as_version=4)

    code_cells = [
        cell for cell in notebook.cells
        if cell.cell_type == "code"
    ]

    assert code_cells
    assert all(
        cell.execution_count is not None
        for cell in code_cells
    )

    for cell in code_cells:
        assert not any(
            output.output_type == "error"
            for output in cell.get("outputs", [])
        )


def test_pdf_page_count():
    from pypdf import PdfReader

    path = ROOT / "reports/04_business_report.pdf"

    reader = PdfReader(str(path))

    assert 5 <= len(reader.pages) <= 8
