from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "transactions_combined.pkl"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# UTILITY
# =============================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_plot(filename):

    path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Saved figure: {path}")

    plt.close()


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print_section(
    "PHASE 2 — EXPLORATORY DATA ANALYSIS"
)

print("\nLoading dataset...")

start = time.time()

df = pd.read_pickle(DATA_FILE)

print(
    f"Dataset loaded in "
    f"{time.time() - start:.2f} seconds"
)

print(f"Shape: {df.shape}")


# =============================================================================
# STEP 2 — CREATE TEMPORAL FEATURES
# =============================================================================

print_section(
    "STEP 2 — TEMPORAL FEATURE EXTRACTION"
)

df["hour"] = df["TX_DATETIME"].dt.hour

df["day_of_week"] = (
    df["TX_DATETIME"].dt.dayofweek
)

df["day"] = (
    df["TX_DATETIME"].dt.day
)

df["month"] = (
    df["TX_DATETIME"].dt.month
)

df["date"] = (
    df["TX_DATETIME"].dt.date
)

print("\nTemporal features created:")

print(
    [
        "hour",
        "day_of_week",
        "day",
        "month",
        "date"
    ]
)


# =============================================================================
# STEP 3 — CLASS DISTRIBUTION
# =============================================================================

print_section(
    "STEP 3 — FRAUD DISTRIBUTION"
)

class_distribution = (
    df["TX_FRAUD"]
    .value_counts()
    .sort_index()
)

print("\nClass distribution:")

for label, count in class_distribution.items():

    name = (
        "Legitimate"
        if label == 0
        else "Fraud"
    )

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{name:<15}: "
        f"{count:>10,} "
        f"({percentage:.4f}%)"
    )


plt.figure(figsize=(8, 5))

plt.bar(
    ["Legitimate", "Fraud"],
    [
        class_distribution.get(0, 0),
        class_distribution.get(1, 0)
    ]
)

plt.title(
    "Transaction Class Distribution"
)

plt.ylabel(
    "Number of Transactions"
)

save_plot(
    "01_class_distribution.png"
)


# =============================================================================
# STEP 4 — TRANSACTION AMOUNTS
# =============================================================================

print_section(
    "STEP 4 — TRANSACTION AMOUNT ANALYSIS"
)

print("\nAmount statistics by class:")

amount_stats = (
    df.groupby("TX_FRAUD")["TX_AMOUNT"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
)

print(
    amount_stats.to_string()
)


plt.figure(figsize=(9, 6))

plt.hist(
    df.loc[
        df["TX_FRAUD"] == 0,
        "TX_AMOUNT"
    ],
    bins=100,
    alpha=0.6,
    label="Legitimate"
)

plt.hist(
    df.loc[
        df["TX_FRAUD"] == 1,
        "TX_AMOUNT"
    ],
    bins=100,
    alpha=0.6,
    label="Fraud"
)

plt.title(
    "Transaction Amount Distribution"
)

plt.xlabel(
    "Transaction Amount"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

save_plot(
    "02_transaction_amount_distribution.png"
)


# =============================================================================
# STEP 5 — FRAUD BY HOUR
# =============================================================================

print_section(
    "STEP 5 — FRAUD RATE BY HOUR"
)

hour_stats = (
    df.groupby("hour")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        frauds=(
            "TX_FRAUD",
            "sum"
        )
    )
)

hour_stats["fraud_rate"] = (
    hour_stats["frauds"]
    / hour_stats["transactions"]
    * 100
)

print(
    "\nHourly fraud statistics:"
)

print(
    hour_stats.to_string()
)


plt.figure(figsize=(10, 5))

plt.plot(
    hour_stats.index,
    hour_stats["fraud_rate"],
    marker="o"
)

plt.title(
    "Fraud Rate by Hour of Day"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Fraud Rate (%)"
)

plt.xticks(
    range(24)
)

save_plot(
    "03_fraud_rate_by_hour.png"
)


# =============================================================================
# STEP 6 — FRAUD BY DAY OF WEEK
# =============================================================================

print_section(
    "STEP 6 — FRAUD RATE BY DAY OF WEEK"
)

weekday_stats = (
    df.groupby("day_of_week")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        frauds=(
            "TX_FRAUD",
            "sum"
        )
    )
)

weekday_stats["fraud_rate"] = (
    weekday_stats["frauds"]
    / weekday_stats["transactions"]
    * 100
)

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_stats["day_name"] = [
    weekday_names[i]
    for i in weekday_stats.index
]

print(
    weekday_stats.to_string()
)


plt.figure(figsize=(10, 5))

plt.bar(
    weekday_stats["day_name"],
    weekday_stats["fraud_rate"]
)

plt.title(
    "Fraud Rate by Day of Week"
)

plt.xlabel(
    "Day"
)

plt.ylabel(
    "Fraud Rate (%)"
)

plt.xticks(rotation=30)

save_plot(
    "04_fraud_rate_by_day_of_week.png"
)


# =============================================================================
# STEP 7 — DAILY FRAUD TREND
# =============================================================================

print_section(
    "STEP 7 — FRAUD TREND OVER TIME"
)

daily_stats = (
    df.set_index("TX_DATETIME")
    .resample("D")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        frauds=(
            "TX_FRAUD",
            "sum"
        )
    )
)

daily_stats["fraud_rate"] = (
    daily_stats["frauds"]
    / daily_stats["transactions"]
    * 100
)

print(
    "\nDaily fraud summary:"
)

print(
    daily_stats.describe().to_string()
)


plt.figure(figsize=(14, 5))

plt.plot(
    daily_stats.index,
    daily_stats["fraud_rate"]
)

plt.title(
    "Daily Fraud Rate Over Time"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Fraud Rate (%)"
)

save_plot(
    "05_daily_fraud_rate.png"
)


# =============================================================================
# STEP 8 — FRAUD SCENARIOS
# =============================================================================

print_section(
    "STEP 8 — FRAUD SCENARIO ANALYSIS"
)

scenario_stats = (
    df.groupby("TX_FRAUD_SCENARIO")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        frauds=(
            "TX_FRAUD",
            "sum"
        )
    )
)

scenario_stats["fraud_rate"] = (
    scenario_stats["frauds"]
    / scenario_stats["transactions"]
    * 100
)

print(
    scenario_stats.to_string()
)


# =============================================================================
# STEP 9 — CUSTOMER ANALYSIS
# =============================================================================

print_section(
    "STEP 9 — CUSTOMER BEHAVIOR"
)

customer_stats = (
    df.groupby("CUSTOMER_ID")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        total_amount=(
            "TX_AMOUNT",
            "sum"
        ),
        mean_amount=(
            "TX_AMOUNT",
            "mean"
        ),
        fraud_count=(
            "TX_FRAUD",
            "sum"
        )
    )
)

customer_stats["fraud_rate"] = (
    customer_stats["fraud_count"]
    / customer_stats["transactions"]
    * 100
)

print(
    f"\nUnique customers: "
    f"{len(customer_stats):,}"
)

print(
    "\nCustomer statistics:"
)

print(
    customer_stats.describe().to_string()
)


# =============================================================================
# STEP 10 — TERMINAL ANALYSIS
# =============================================================================

print_section(
    "STEP 10 — TERMINAL BEHAVIOR"
)

terminal_stats = (
    df.groupby("TERMINAL_ID")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),
        total_amount=(
            "TX_AMOUNT",
            "sum"
        ),
        mean_amount=(
            "TX_AMOUNT",
            "mean"
        ),
        fraud_count=(
            "TX_FRAUD",
            "sum"
        )
    )
)

terminal_stats["fraud_rate"] = (
    terminal_stats["fraud_count"]
    / terminal_stats["transactions"]
    * 100
)

print(
    f"\nUnique terminals: "
    f"{len(terminal_stats):,}"
)

print(
    "\nTerminal statistics:"
)

print(
    terminal_stats.describe().to_string()
)


# =============================================================================
# STEP 11 — FRAUD BY SCENARIO
# =============================================================================

print_section(
    "STEP 11 — FRAUD SCENARIO COUNTS"
)

fraud_scenarios = (
    df.loc[
        df["TX_FRAUD"] == 1
    ]
    ["TX_FRAUD_SCENARIO"]
    .value_counts()
    .sort_index()
)

print(
    "\nFraud cases by scenario:"
)

print(
    fraud_scenarios.to_string()
)


# =============================================================================
# STEP 12 — SAVE TABLES
# =============================================================================

print_section(
    "STEP 12 — SAVING EDA TABLES"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

hour_stats.to_csv(
    TABLE_DIR / "fraud_by_hour.csv"
)

weekday_stats.to_csv(
    TABLE_DIR / "fraud_by_day_of_week.csv"
)

daily_stats.to_csv(
    TABLE_DIR / "daily_fraud_statistics.csv"
)

scenario_stats.to_csv(
    TABLE_DIR / "fraud_scenarios.csv"
)

customer_stats.describe().to_csv(
    TABLE_DIR / "customer_statistics.csv"
)

terminal_stats.describe().to_csv(
    TABLE_DIR / "terminal_statistics.csv"
)

print(
    f"\nTables saved to:"
)

print(TABLE_DIR)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section(
    "PHASE 2 — EDA COMPLETE"
)

print(
    f"""
Dataset
-------
Transactions : {len(df):,}
Fraud cases   : {df["TX_FRAUD"].sum():,}
Fraud rate    : {df["TX_FRAUD"].mean() * 100:.4f}%
Customers     : {df["CUSTOMER_ID"].nunique():,}
Terminals     : {df["TERMINAL_ID"].nunique():,}

EDA outputs
-----------
Figures : {FIGURE_DIR}
Tables  : {TABLE_DIR}
"""
)