import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "artifacts/transactions_combined.pkl"

FEATURE_DIR = "artifacts/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

TARGET = "TX_FRAUD"


# ============================================================
# Utility
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load data
# ============================================================

def load_data():

    print_section("LOADING DATA")

    df = pd.read_pickle(INPUT_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"]
    )

    df = df.sort_values(
        ["TX_DATETIME", "TRANSACTION_ID"]
    ).reset_index(drop=True)

    print(
        f"Date range: "
        f"{df['TX_DATETIME'].min()} "
        f"to "
        f"{df['TX_DATETIME'].max()}"
    )

    return df


# ============================================================
# Temporal features
# ============================================================

def create_temporal_features(df):

    print_section("CREATING TEMPORAL FEATURES")

    dt = df["TX_DATETIME"]

    df["hour"] = dt.dt.hour

    df["day_of_week"] = dt.dt.dayofweek

    df["day_of_month"] = dt.dt.day

    df["month"] = dt.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(np.int8)

    # Cyclic encoding
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

    print("Temporal features created.")

    return df


# ============================================================
# Transaction features
# ============================================================

def create_transaction_features(df):

    print_section("CREATING TRANSACTION FEATURES")

    df["log_amount"] = np.log1p(
        df["TX_AMOUNT"].clip(lower=0)
    )

    print("Transaction features created.")

    return df


# ============================================================
# Customer historical features
# ============================================================

def create_customer_features(df):

    print_section("CREATING CUSTOMER BEHAVIOR FEATURES")

    customer = df.groupby(
        "CUSTOMER_ID",
        sort=False
    )

    # Number of previous transactions
    df["customer_tx_count"] = (
        customer.cumcount()
    )

    # Previous transaction amount
    df["customer_previous_amount"] = (
        customer["TX_AMOUNT"]
        .shift(1)
    )

    # Previous transaction timestamp
    previous_customer_time = (
        customer["TX_DATETIME"]
        .shift(1)
    )

    df["customer_time_since_previous"] = (
        (
            df["TX_DATETIME"]
            - previous_customer_time
        )
        .dt.total_seconds()
    )

    # Running sum
    customer_amount_sum = (
        customer["TX_AMOUNT"]
        .cumsum()
        .shift(1)
    )

    df["customer_mean_amount"] = (
        customer_amount_sum
        / df["customer_tx_count"].replace(
            0,
            np.nan
        )
    )

    # Running squared sum
    customer_squared = (
        df["TX_AMOUNT"] ** 2
    )

    running_squared = (
        customer_squared
        .groupby(
            df["CUSTOMER_ID"],
            sort=False
        )
        .cumsum()
        .shift(1)
    )

    count = df[
        "customer_tx_count"
    ].replace(
        0,
        np.nan
    )

    variance = (
        running_squared / count
        - df["customer_mean_amount"] ** 2
    )

    df["customer_amount_std"] = np.sqrt(
        variance.clip(lower=0)
    )

    # Amount deviation
    df["amount_vs_customer_mean"] = (
        df["TX_AMOUNT"]
        / df["customer_mean_amount"].replace(
            0,
            np.nan
        )
    )

    df["customer_amount_zscore"] = (
        (
            df["TX_AMOUNT"]
            - df["customer_mean_amount"]
        )
        /
        df["customer_amount_std"].replace(
            0,
            np.nan
        )
    )

    # First-seen time
    first_customer_time = (
        df.groupby(
            "CUSTOMER_ID",
            sort=False
        )["TX_DATETIME"]
        .transform("min")
    )

    df["customer_age_seconds"] = (
        (
            df["TX_DATETIME"]
            - first_customer_time
        )
        .dt.total_seconds()
    )

    print("Customer behavioral features created.")

    return df


# ============================================================
# Terminal historical features
# ============================================================

def create_terminal_features(df):

    print_section("CREATING TERMINAL BEHAVIOR FEATURES")

    terminal = df.groupby(
        "TERMINAL_ID",
        sort=False
    )

    df["terminal_tx_count"] = (
        terminal.cumcount()
    )

    df["terminal_previous_amount"] = (
        terminal["TX_AMOUNT"]
        .shift(1)
    )

    previous_terminal_time = (
        terminal["TX_DATETIME"]
        .shift(1)
    )

    df["terminal_time_since_previous"] = (
        (
            df["TX_DATETIME"]
            - previous_terminal_time
        )
        .dt.total_seconds()
    )

    terminal_sum = (
        terminal["TX_AMOUNT"]
        .cumsum()
        .shift(1)
    )

    df["terminal_mean_amount"] = (
        terminal_sum
        / df["terminal_tx_count"].replace(
            0,
            np.nan
        )
    )

    df["amount_vs_terminal_mean"] = (
        df["TX_AMOUNT"]
        / df["terminal_mean_amount"].replace(
            0,
            np.nan
        )
    )

    terminal_first_time = (
        df.groupby(
            "TERMINAL_ID",
            sort=False
        )["TX_DATETIME"]
        .transform("min")
    )

    df["terminal_age_seconds"] = (
        (
            df["TX_DATETIME"]
            - terminal_first_time
        )
        .dt.total_seconds()
    )

    print("Terminal behavioral features created.")

    return df


# ============================================================
# Velocity features
# ============================================================

def create_velocity_features(df):

    print_section("CREATING VELOCITY FEATURES")

    # --------------------------------------------------------
    # Customer rolling transaction counts
    # --------------------------------------------------------

    customer_time = df.set_index(
        "TX_DATETIME"
    )

    for window in ["1h", "6h", "24h", "7d"]:

        values = (
            customer_time
            .groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .rolling(window)
            .count()
            .reset_index(
                level=0,
                drop=True
            )
        )

        # Remove current transaction
        values = values - 1

        df[
            f"customer_tx_count_{window}"
        ] = values.values

    # --------------------------------------------------------
    # Customer rolling amount
    # --------------------------------------------------------

    for window in ["1h", "24h", "7d"]:

        values = (
            customer_time
            .groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .rolling(window)
            .sum()
            .reset_index(
                level=0,
                drop=True
            )
        )

        values = values - df[
            "TX_AMOUNT"
        ].values

        df[
            f"customer_amount_sum_{window}"
        ] = values.values

    # --------------------------------------------------------
    # Terminal rolling transaction counts
    # --------------------------------------------------------

    for window in ["1h", "6h", "24h", "7d"]:

        values = (
            customer_time
            .groupby("TERMINAL_ID")["TX_AMOUNT"]
            .rolling(window)
            .count()
            .reset_index(
                level=0,
                drop=True
            )
        )

        values = values - 1

        df[
            f"terminal_tx_count_{window}"
        ] = values.values

    # --------------------------------------------------------
    # Terminal rolling amounts
    # --------------------------------------------------------

    for window in ["1h", "24h", "7d"]:

        values = (
            customer_time
            .groupby("TERMINAL_ID")["TX_AMOUNT"]
            .rolling(window)
            .sum()
            .reset_index(
                level=0,
                drop=True
            )
        )

        values = values - df[
            "TX_AMOUNT"
        ].values

        df[
            f"terminal_amount_sum_{window}"
        ] = values.values

    print("Velocity features created.")

    return df


# ============================================================
# Customer-terminal interaction
# ============================================================

def create_interaction_features(df):

    print_section(
        "CREATING CUSTOMER-TERMINAL FEATURES"
    )

    pair = (
        df.groupby(
            ["CUSTOMER_ID", "TERMINAL_ID"],
            sort=False
        )
    )

    df["customer_terminal_tx_count"] = (
        pair.cumcount()
    )

    previous_pair_time = (
        pair["TX_DATETIME"]
        .shift(1)
    )

    df["customer_terminal_time_since_previous"] = (
        (
            df["TX_DATETIME"]
            - previous_pair_time
        )
        .dt.total_seconds()
    )

    df["customer_terminal_seen_before"] = (
        (
            df["customer_terminal_tx_count"]
            > 0
        )
        .astype(np.int8)
    )

    df["customer_seen_before"] = (
        (
            df["customer_tx_count"]
            > 0
        )
        .astype(np.int8)
    )

    df["terminal_seen_before"] = (
        (
            df["terminal_tx_count"]
            > 0
        )
        .astype(np.int8)
    )

    print("Interaction features created.")

    return df


# ============================================================
# Historical fraud features
# ============================================================

def create_historical_fraud_features(df):

    print_section(
        "CREATING HISTORICAL FRAUD FEATURES"
    )

    global_fraud_rate = (
        df[TARGET].mean()
    )

    alpha = 10.0

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer = df.groupby(
        "CUSTOMER_ID",
        sort=False
    )

    previous_fraud = (
        customer[TARGET]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    previous_count = (
        df["customer_tx_count"]
    )

    df["customer_previous_fraud_count"] = (
        previous_fraud
    )

    df["customer_historical_fraud_rate"] = (
        previous_fraud
        + alpha * global_fraud_rate
    ) / (
        previous_count + alpha
    )

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------

    terminal = df.groupby(
        "TERMINAL_ID",
        sort=False
    )

    previous_terminal_fraud = (
        terminal[TARGET]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    previous_terminal_count = (
        df["terminal_tx_count"]
    )

    df["terminal_previous_fraud_count"] = (
        previous_terminal_fraud
    )

    df["terminal_historical_fraud_rate"] = (
        previous_terminal_fraud
        + alpha * global_fraud_rate
    ) / (
        previous_terminal_count + alpha
    )

    print(
        f"Global fraud rate: "
        f"{global_fraud_rate:.6%}"
    )

    print("Historical fraud features created.")

    return df


# ============================================================
# Final cleanup
# ============================================================

def prepare_features(df):

    print_section("PREPARING MODEL FEATURES")

    excluded = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_FRAUD_SCENARIO",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded
        and col != TARGET
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].copy()

    # Replace infinities
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing historical values
    # with zero because they represent
    # absence of previous history.
    X = X.fillna(0)

    # Numeric validation
    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:

        raise TypeError(
            "Non-numeric features detected: "
            f"{non_numeric}"
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            "Infinite values remain."
        )

    result = X.copy()

    result[TARGET] = y.values

    print(
        f"Final feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Final dataset shape: "
        f"{result.shape}"
    )

    return result, feature_columns


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    df = create_temporal_features(df)

    df = create_transaction_features(df)

    df = create_customer_features(df)

    df = create_terminal_features(df)

    df = create_velocity_features(df)

    df = create_interaction_features(df)

    df = create_historical_fraud_features(df)

    features, feature_columns = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(features)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train = features.iloc[
        :train_end
    ].copy()

    validation = features.iloc[
        train_end:validation_end
    ].copy()

    test = features.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train.to_pickle(
        f"{FEATURE_DIR}/train_features.pkl"
    )

    validation.to_pickle(
        f"{FEATURE_DIR}/validation_features.pkl"
    )

    test.to_pickle(
        f"{FEATURE_DIR}/test_features.pkl"
    )

    pd.Series(
        feature_columns
    ).to_csv(
        f"{FEATURE_DIR}/feature_columns.csv",
        index=False,
        header=["feature"]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_section(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    print(
        f"Features   : {len(feature_columns)}"
    )

    print(
        f"\nSaved to: {FEATURE_DIR}"
    )


if __name__ == "__main__":
    main()