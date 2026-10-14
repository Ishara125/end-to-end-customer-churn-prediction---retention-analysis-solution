
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "02_cleaned_dataset.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.json"

sns.set_theme(style="whitegrid", palette="deep")


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def run_eda():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn target must be binary.")

    summary = {
        "total_customers": int(len(df)),
        "churned_customers": int(df["churn"].sum()),
        "overall_churn_rate": round(float(df["churn"].mean()), 4),
        "missing_values": df.isna().sum().to_dict(),
    }

    # 1. Churn distribution
    counts = df["churn"].value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=["Retained", "Churned"],
        y=counts.values
    )
    ax.set(
        title="Customer Churn Distribution",
        xlabel="Customer Status",
        ylabel="Number of Customers"
    )
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    save_chart("01_churn_distribution.png")

    # 2. Churn rate by contract type
    contract = (
        df.groupby("contract_type")["churn"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    summary["contract_churn"] = contract.round(4).to_dict("index")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=contract.index,
        y=contract["mean"] * 100
    )
    ax.set(
        title="Churn Rate by Contract Type",
        xlabel="Contract Type",
        ylabel="Churn Rate (%)",
        ylim=(0, 100)
    )
    plt.xticks(rotation=15)
    save_chart("02_contract_churn.png")

    # 3. Tenure versus churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="churn",
        y="tenure_months",
        order=[0, 1]
    )
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.title("Customer Tenure by Churn Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Tenure (Months)")
    save_chart("03_tenure_churn.png")

    # 4. Monthly charges versus churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="churn",
        y="monthly_charges",
        order=[0, 1]
    )
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Monthly Charges")
    save_chart("04_monthly_charges.png")

    # 5. Satisfaction and complaints
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        data=df,
        x="churn",
        y="satisfaction_score",
        order=[0, 1],
        ax=axes[0]
    )
    axes[0].set(
        title="Satisfaction vs Churn",
        xlabel="Churn (0 = No, 1 = Yes)",
        ylabel="Satisfaction Score"
    )

    sns.boxplot(
        data=df,
        x="churn",
        y="complaints_last_12m",
        order=[0, 1],
        ax=axes[1]
    )
    axes[1].set(
        title="Complaints vs Churn",
        xlabel="Churn (0 = No, 1 = Yes)",
        ylabel="Complaints in Last 12 Months"
    )
    save_chart("05_satisfaction_complaints.png")

    # 6. Late payments
    late_payment = (
        df.groupby("late_payments_last_12m")["churn"]
        .agg(["mean", "count"])
        .sort_index()
    )
    summary["late_payment_churn"] = (
        late_payment.round(4).to_dict("index")
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=late_payment.index,
        y=late_payment["mean"] * 100
    )
    plt.title("Churn Rate by Late Payment Count")
    plt.xlabel("Late Payments in Last 12 Months")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 100)
    save_chart("06_late_payments.png")

    # 7. Monthly usage
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="churn",
        y="monthly_usage_gb",
        order=[0, 1]
    )
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.title("Monthly Usage by Churn Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Monthly Usage (GB)")
    save_chart("07_monthly_usage.png")

    # 8. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4
    )
    plt.title("Numeric Feature Correlation Heatmap")
    save_chart("08_correlation_heatmap.png")

    # Descriptive statistics for business interpretation
    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "satisfaction_score",
        "complaints_last_12m",
        "late_payments_last_12m",
        "monthly_usage_gb",
    ]

    comparison = df.groupby("churn")[numeric_features].mean()
    summary["feature_means_by_churn"] = (
        comparison.round(3).to_dict("index")
    )

    # Include sample sizes to contextualize group comparisons
    print("\nOverall churn rate:",
          f"{summary['overall_churn_rate']:.2%}")
    print("\nContract churn summary:\n", contract)
    print("\nLate-payment churn summary:\n", late_payment)
    print("\nFeature means by churn:\n", comparison)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nEDA completed successfully.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_eda()
