from pathlib import Path
import time

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "transactions_combined.pkl"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "features"
)

FEATURE_DIR.mkdir(
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


# =============================================================================
# LOAD DATA
# =============================================================================

print_section(
    "PHASE 2 — LEAKAGE-SAFE FEATURE ENGINEERING"
)

print("\nLoading complete chronological dataset...")

start_time = time.time()

df = pd.read_pickle(DATA_FILE)

df = (
    df.sort_values(
        ["TX_DATETIME", "TRANSACTION_ID"]
    )
    .reset_index(drop=True)
)

print(
    f"Dataset loaded in "
    f"{time.time() - start_time:.2f} seconds"
)

print(
    f"Total transactions: "
    f"{len(df):,}"
)


# =============================================================================
# STEP 1 — TEMPORAL FEATURES
# =============================================================================

print_section(
    "STEP 1 — TEMPORAL FEATURES"
)

df["hour"] = (
    df["TX_DATETIME"].dt.hour
)

df["day_of_week"] = (
    df["TX_DATETIME"].dt.dayofweek
)

df["day_of_month"] = (
    df["TX_DATETIME"].dt.day
)

df["month"] = (
    df["TX_DATETIME"].dt.month
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype("int8")


# Cyclical encoding

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

print(
    "\nCreated:"
)

print(
    "  hour"
)

print(
    "  day_of_week"
)

print(
    "  day_of_month"
)

print(
    "  month"
)

print(
    "  is_weekend"
)

print(
    "  cyclical time features"
)


# =============================================================================
# STEP 2 — TRANSACTION AMOUNT FEATURES
# =============================================================================

print_section(
    "STEP 2 — TRANSACTION AMOUNT FEATURES"
)

df["log_amount"] = np.log1p(
    df["TX_AMOUNT"]
)

print(
    "\nCreated:"
)

print(
    "  log_amount"
)


# =============================================================================
# STEP 3 — CUSTOMER HISTORICAL FEATURES
# =============================================================================

print_section(
    "STEP 3 — CUSTOMER HISTORICAL FEATURES"
)

print(
    "\nCalculating customer history..."
)

customer_group = df.groupby(
    "CUSTOMER_ID",
    sort=False
)


# Number of previous transactions

df["customer_tx_count"] = (
    customer_group.cumcount()
)


# Previous transaction amount

df["customer_previous_amount"] = (
    customer_group["TX_AMOUNT"]
    .shift(1)
)


# Previous transaction time

customer_previous_time = (
    customer_group["TX_DATETIME"]
    .shift(1)
)


df["customer_time_since_previous"] = (
    df["TX_DATETIME"]
    - customer_previous_time
).dt.total_seconds()


# Running historical mean amount

df["customer_mean_amount"] = (
    customer_group["TX_AMOUNT"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# Running historical standard deviation

df["customer_amount_std"] = (
    customer_group["TX_AMOUNT"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .std()
    )
)


print(
    "\nCustomer historical features created."
)


# =============================================================================
# STEP 4 — TERMINAL HISTORICAL FEATURES
# =============================================================================

print_section(
    "STEP 4 — TERMINAL HISTORICAL FEATURES"
)

print(
    "\nCalculating terminal history..."
)

terminal_group = df.groupby(
    "TERMINAL_ID",
    sort=False
)


# Previous transaction count

df["terminal_tx_count"] = (
    terminal_group.cumcount()
)


# Previous transaction amount

df["terminal_previous_amount"] = (
    terminal_group["TX_AMOUNT"]
    .shift(1)
)


# Previous transaction time

terminal_previous_time = (
    terminal_group["TX_DATETIME"]
    .shift(1)
)


df["terminal_time_since_previous"] = (
    df["TX_DATETIME"]
    - terminal_previous_time
).dt.total_seconds()


# Running historical mean

df["terminal_mean_amount"] = (
    terminal_group["TX_AMOUNT"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


print(
    "\nTerminal historical features created."
)


# =============================================================================
# STEP 5 — BEHAVIORAL DEVIATION FEATURES
# =============================================================================

print_section(
    "STEP 5 — BEHAVIORAL DEVIATION FEATURES"
)

df["amount_vs_customer_mean"] = (
    df["TX_AMOUNT"]
    / (
        df["customer_mean_amount"]
        + 1e-6
    )
)

df["amount_vs_terminal_mean"] = (
    df["TX_AMOUNT"]
    / (
        df["terminal_mean_amount"]
        + 1e-6
    )
)


# =============================================================================
# STEP 6 — CLEAN INFINITE VALUES
# =============================================================================

print_section(
    "STEP 6 — NUMERICAL SANITY CHECK"
)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

infinite_count = np.isinf(
    df[numeric_columns]
    .to_numpy()
).sum()

print(
    f"\nInfinite values found: "
    f"{infinite_count:,}"
)

if infinite_count > 0:

    print(
        "\nReplacing infinite values with NaN..."
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


# =============================================================================
# STEP 7 — DEFINE MODEL FEATURES
# =============================================================================

print_section(
    "STEP 7 — MODEL FEATURE DEFINITION"
)

excluded_columns = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO"
]

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


print(
    f"\nTotal model features: "
    f"{len(feature_columns)}"
)

print("\nModel features:")

for feature in feature_columns:

    print(
        f"  - {feature}"
    )


# =============================================================================
# STEP 8 — EXPLICIT LEAKAGE CHECK
# =============================================================================

print_section(
    "STEP 8 — LEAKAGE CHECK"
)

for column in excluded_columns:

    if column in feature_columns:

        raise RuntimeError(
            f"LEAKAGE ERROR: "
            f"{column} is present in model features."
        )


# Specifically ensure fraud scenario is excluded

assert (
    "TX_FRAUD_SCENARIO"
    not in feature_columns
)

assert (
    "TX_FRAUD"
    not in feature_columns
)


print(
    "\nLeakage checks passed."
)

print(
    "Target columns are excluded."
)


# =============================================================================
# STEP 9 — VERIFY HISTORICAL FEATURES
# =============================================================================

print_section(
    "STEP 9 — HISTORICAL FEATURE VALIDATION"
)

history_features = [
    "customer_tx_count",
    "customer_previous_amount",
    "customer_time_since_previous",
    "customer_mean_amount",
    "customer_amount_std",
    "terminal_tx_count",
    "terminal_previous_amount",
    "terminal_time_since_previous",
    "terminal_mean_amount"
]

print(
    "\nHistorical features:"
)

for feature in history_features:

    null_count = df[feature].isna().sum()

    print(
        f"{feature:<35} "
        f"initial/history NaNs: "
        f"{null_count:,}"
    )


# =============================================================================
# STEP 10 — SPLIT AFTER FEATURE CREATION
# =============================================================================

print_section(
    "STEP 10 — TEMPORAL SPLIT"
)

n = len(df)

train_end = int(
    n * 0.70
)

validation_end = int(
    n * 0.85
)


train_features = df.iloc[
    :train_end
].copy()

validation_features = df.iloc[
    train_end:validation_end
].copy()

test_features = df.iloc[
    validation_end:
].copy()


print(
    f"""
TRAIN
Rows : {len(train_features):,}
Start: {train_features["TX_DATETIME"].min()}
End  : {train_features["TX_DATETIME"].max()}

VALIDATION
Rows : {len(validation_features):,}
Start: {validation_features["TX_DATETIME"].min()}
End  : {validation_features["TX_DATETIME"].max()}

TEST
Rows : {len(test_features):,}
Start: {test_features["TX_DATETIME"].min()}
End  : {test_features["TX_DATETIME"].max()}
"""
)


# =============================================================================
# STEP 11 — SAVE
# =============================================================================

print_section(
    "STEP 11 — SAVING FEATURE DATASETS"
)

train_file = (
    FEATURE_DIR
    / "train_features.pkl"
)

validation_file = (
    FEATURE_DIR
    / "validation_features.pkl"
)

test_file = (
    FEATURE_DIR
    / "test_features.pkl"
)


train_features.to_pickle(
    train_file
)

validation_features.to_pickle(
    validation_file
)

test_features.to_pickle(
    test_file
)


pd.Series(
    feature_columns
).to_csv(
    FEATURE_DIR
    / "feature_columns.csv",
    index=False,
    header=["feature"]
)


print(
    f"\nTrain features      : {train_file}"
)

print(
    f"Validation features : {validation_file}"
)

print(
    f"Test features       : {test_file}"
)

print(
    f"Feature list        : "
    f"{FEATURE_DIR / 'feature_columns.csv'}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section(
    "PHASE 2 — FEATURE ENGINEERING COMPLETE"
)

print(
    f"""
Dataset
-------
Total transactions : {len(df):,}

Features
--------
Model features     : {len(feature_columns)}

Splits
-------
Train              : {len(train_features):,}
Validation         : {len(validation_features):,}
Test               : {len(test_features):,}

Important
---------
Historical features were calculated on the
complete chronological transaction stream.

Each transaction only uses information from
transactions occurring before it.

Target leakage columns excluded:
- TX_FRAUD
- TX_FRAUD_SCENARIO
- TRANSACTION_ID

"""
)