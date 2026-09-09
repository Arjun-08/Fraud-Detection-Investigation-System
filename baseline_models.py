def load_data():

    print("=" * 70)
    print("LOADING FEATURE DATA")
    print("=" * 70)

    train = pd.read_pickle(TRAIN_PATH)
    validation = pd.read_pickle(VAL_PATH)

    target = "TX_FRAUD"

    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    excluded_columns = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_FRAUD_SCENARIO",
    ]

    excluded_columns = [
        col
        for col in excluded_columns
        if col in train.columns
    ]

    print("\nExcluded columns:")
    for col in excluded_columns:
        print(f"  - {col}")

    X_train = train.drop(
        columns=[target] + excluded_columns
    )

    y_train = train[target].astype(int)

    X_val = validation.drop(
        columns=[target] + excluded_columns
    )

    y_val = validation[target].astype(int)

    # --------------------------------------------------------
    # Feature consistency
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Train and validation feature columns do not match."
        )

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    non_numeric = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            "Non-numeric model features detected: "
            f"{non_numeric}"
        )

    # --------------------------------------------------------
    # Missing / infinite values
    # --------------------------------------------------------

    if X_train.isna().any().any():
        raise ValueError(
            "Missing values detected in training features."
        )

    if X_val.isna().any().any():
        raise ValueError(
            "Missing values detected in validation features."
        )

    if not np.isfinite(
        X_train.to_numpy()
    ).all():
        raise ValueError(
            "Infinite values detected in training features."
        )

    if not np.isfinite(
        X_val.to_numpy()
    ).all():
        raise ValueError(
            "Infinite values detected in validation features."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Train shape      : {X_train.shape}"
    )

    print(
        f"Validation shape : {X_val.shape}"
    )

    print(
        f"Train fraud rate : {y_train.mean():.4%}"
    )

    print(
        f"Val fraud rate   : {y_val.mean():.4%}"
    )

    print(
        f"Number of features: {X_train.shape[1]}"
    )

    return X_train, y_train, X_val, y_val