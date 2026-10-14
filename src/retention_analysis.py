
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "retention_simulation.json"
RANKING_PATH = REPORT_DIR / "retention_targeting.csv"
SENSITIVITY_PATH = REPORT_DIR / "retention_sensitivity.csv"

CONTACT_FRACTION = 0.20

# All financial assumptions are hypothetical.
# Currency-neutral monetary units, not observed revenue.
CONTACT_COST = 5.0
RETAINED_CUSTOMER_VALUE = 200.0
INTERVENTION_SUCCESS_RATE = 0.15


def top_k(n, fraction=CONTACT_FRACTION):
    """Round up the contact capacity to a whole customer."""
    if n <= 0:
        raise ValueError("Customer count must be positive.")
    if not 0 < fraction <= 1:
        raise ValueError("Contact fraction must be in (0, 1].")
    return math.ceil(n * fraction)


def validate_predictions(df):
    required = {
        "customer_id",
        "actual_churn",
        "predicted_probability",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Prediction file is empty.")

    if df["customer_id"].isna().any():
        raise ValueError("Missing customer identifiers.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer identifiers.")

    if (
        df["actual_churn"].isna().any()
        or not df["actual_churn"].isin([0, 1]).all()
    ):
        raise ValueError("Invalid churn labels.")

    probabilities = df["predicted_probability"]

    if (
        probabilities.isna().any()
        or not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError("Invalid churn probabilities.")


def financial_simulation(
    contacted,
    actual_churners_reached,
    contact_cost=CONTACT_COST,
    retained_value=RETAINED_CUSTOMER_VALUE,
    success_rate=INTERVENTION_SUCCESS_RATE,
):
    """
    Expected financial outcomes under hypothetical assumptions.

    Assumes each reached actual churner has an independent,
    equal intervention success probability.

    These results are not observed intervention outcomes.
    """
    if min(contacted, actual_churners_reached) < 0:
        raise ValueError("Counts cannot be negative.")

    if actual_churners_reached > contacted:
        raise ValueError("Churners reached exceed contacts.")

    if contact_cost < 0 or retained_value < 0:
        raise ValueError("Costs and values cannot be negative.")

    if not 0 <= success_rate <= 1:
        raise ValueError("Success rate must be between 0 and 1.")

    expected_retained = actual_churners_reached * success_rate
    campaign_cost = contacted * contact_cost
    expected_benefit = expected_retained * retained_value
    net_benefit = expected_benefit - campaign_cost

    return {
        "contacts": int(contacted),
        "churners_reached": float(actual_churners_reached),
        "success_rate": float(success_rate),
        "expected_retained_customers": float(expected_retained),
        "campaign_cost": float(campaign_cost),
        "expected_benefit": float(expected_benefit),
        "expected_net_benefit": float(net_benefit),
        "roi": (
            float(net_benefit / campaign_cost)
            if campaign_cost > 0 else None
        ),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_PATH)
    validate_predictions(df)

    n = len(df)
    k = top_k(n)
    total_churners = int(df["actual_churn"].sum())

    # Stable, reproducible ranking with ID as tie-breaker.
    ranked = df.sort_values(
        ["predicted_probability", "customer_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    ranked["risk_rank"] = np.arange(1, n + 1)
    ranked["selected_for_contact"] = ranked["risk_rank"] <= k

    selected = ranked.head(k)
    churners_captured = int(selected["actual_churn"].sum())

    precision_at_20 = churners_captured / k
    recall_at_20 = (
        churners_captured / total_churners
        if total_churners else 0.0
    )

    prevalence = total_churners / n

    # Expected performance from uniformly random selection,
    # without replacement, under the same contact budget.
    expected_random_churners = k * prevalence
    random_precision = prevalence
    random_recall = k / n if total_churners else 0.0

    lift = (
        precision_at_20 / prevalence
        if prevalence else 0.0
    )

    additional_churners = (
        churners_captured - expected_random_churners
    )

    targeting = financial_simulation(
        k, churners_captured
    )
    random_strategy = financial_simulation(
        k, expected_random_churners
    )

    # Save ranking for reproducibility and future inspection.
    ranked.to_csv(RANKING_PATH, index=False)

    summary = {
        "evaluation_cohort": n,
        "contact_fraction_requested": CONTACT_FRACTION,
        "contact_capacity": k,
        "actual_contact_fraction": k / n,
        "rounding_rule": "ceil(n * contact_fraction)",
        "total_actual_churners": total_churners,
        "churn_prevalence": prevalence,
        "targeting": {
            "actual_churners_captured": churners_captured,
            "precision_at_20": precision_at_20,
            "recall_at_20": recall_at_20,
            "lift_at_20": lift,
        },
        "random_baseline": {
            "method": "Expected uniform random sample",
            "expected_churners_captured": expected_random_churners,
            "expected_precision": random_precision,
            "expected_recall": random_recall,
        },
        "additional_churners_vs_random_expected": (
            additional_churners
        ),
        "hypothetical_financial_assumptions": {
            "contact_cost": CONTACT_COST,
            "retained_customer_value": RETAINED_CUSTOMER_VALUE,
            "intervention_success_rate": (
                INTERVENTION_SUCCESS_RATE
            ),
            "currency": "Hypothetical monetary units",
        },
        "financial_outcomes": {
            "model_targeting": targeting,
            "random_targeting_expected": random_strategy,
            "no_outreach_incremental_net_benefit": 0.0,
            "incremental_net_benefit_vs_random": (
                targeting["expected_net_benefit"]
                - random_strategy["expected_net_benefit"]
            ),
        },
        "limitations": [
            "Test labels provide retrospective targeting results.",
            "Random baseline is an expectation, not a sampled campaign.",
            "Financial values and success rate are assumptions.",
            "Prediction accuracy does not establish treatment effects.",
            "Customers might differ in intervention responsiveness.",
            "No-outreach incremental benefit is assumed to be zero.",
            "No real retention intervention was conducted.",
        ],
    }

    # Figure 16: Churners reached
    plt.figure(figsize=(8, 5))
    labels = ["Model Targeting", "Random Targeting\n(Expected)"]
    values = [
        churners_captured,
        expected_random_churners,
    ]

    bars = plt.bar(labels, values)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Actual / Expected Churners Reached")
    plt.title("Top-20% Targeting vs Random Outreach")
    plt.ylim(0, k * 1.15)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "16_targeting_vs_random.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 17: Cumulative gain
    cumulative = ranked["actual_churn"].cumsum()
    contact_percent = np.arange(1, n + 1) / n

    model_gain = (
        cumulative / total_churners
        if total_churners
        else np.zeros(n)
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.r_[0, contact_percent],
        np.r_[0, model_gain],
        label="Model ranking",
    )
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Expected random targeting",
    )
    plt.axvline(
        k / n,
        linestyle=":",
        color="black",
        label="20% contact budget",
    )
    plt.xlabel("Fraction of Customers Contacted")
    plt.ylabel("Fraction of Actual Churners Captured")
    plt.title("Cumulative Gains Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "17_cumulative_gains.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Sensitivity analysis: vary intervention success rate.
    sensitivity_rows = []

    for rate in np.arange(0.05, 0.31, 0.05):
        targeted = financial_simulation(
            k, churners_captured,
            success_rate=float(rate),
        )
        random_result = financial_simulation(
            k, expected_random_churners,
            success_rate=float(rate),
        )

        sensitivity_rows.append({
            "success_rate": float(round(rate, 2)),
            "model_net_benefit": (
                targeted["expected_net_benefit"]
            ),
            "random_net_benefit": (
                random_result["expected_net_benefit"]
            ),
            "incremental_advantage": (
                targeted["expected_net_benefit"]
                - random_result["expected_net_benefit"]
            ),
        })

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        sensitivity["success_rate"] * 100,
        sensitivity["model_net_benefit"],
        marker="o",
        label="Model targeting",
    )
    plt.plot(
        sensitivity["success_rate"] * 100,
        sensitivity["random_net_benefit"],
        marker="s",
        label="Random targeting (expected)",
    )
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Assumed Intervention Success Rate (%)")
    plt.ylabel("Expected Net Benefit (Monetary Units)")
    plt.title("Financial Sensitivity to Intervention Success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "18_financial_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print("=" * 65)
    print("CUSTOMER RETENTION INTERVENTION SIMULATION")
    print("=" * 65)

    print(f"Total customers: {n}")
    print(f"Contact capacity (20%): {k}")
    print(f"Total actual churners: {total_churners}")
    print(f"Churn prevalence: {prevalence:.2%}")

    print("\nMODEL TARGETING")
    print(f"Actual churners captured: {churners_captured}")
    print(f"Precision @20%: {precision_at_20:.2%}")
    print(f"Recall @20%: {recall_at_20:.2%}")
    print(f"Lift @20%: {lift:.4f}")

    print("\nRANDOM BASELINE")
    print(
        "Expected churners captured:",
        f"{expected_random_churners:.2f}"
    )
    print(
        "Additional churners vs random expectation:",
        f"{additional_churners:.2f}"
    )

    print("\nHYPOTHETICAL FINANCIAL SIMULATION")
    print(
        f"Model expected retained customers: "
        f"{targeting['expected_retained_customers']:.2f}"
    )
    print(
        f"Model expected campaign cost: "
        f"{targeting['campaign_cost']:.2f}"
    )
    print(
        f"Model expected net benefit: "
        f"{targeting['expected_net_benefit']:.2f}"
    )
    print(
        f"Random expected net benefit: "
        f"{random_strategy['expected_net_benefit']:.2f}"
    )

    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Saved ranking: {RANKING_PATH}")
    print(f"Saved sensitivity: {SENSITIVITY_PATH}")
    print("\nRetention simulation completed successfully.")


if __name__ == "__main__":
    main()
