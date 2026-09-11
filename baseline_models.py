def load_data():

    print("=" * 70)
    print("LOADING FEATURE DATA")
    print("=" * 70)

    train = pd.read_pickle(TRAIN_PATH)
    validation = pd.read_pickle(VAL_PATH)

    target = "TX_FRAUD"

    X_train = train.drop(
        columns=[target]
    )

    y_train = train[target].astype(int)

    X_val = validation.drop(
        columns=[target]
    )

    y_val = validation[target].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Train and validation feature columns do not match."
        )

    # Ensure everything is numeric
    non_numeric = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"Non-numeric features detected: {non_numeric}"
        )

    # Missing values
    if X_train.isna().any().any():
        raise ValueError(
            "Missing values detected in training data."
        )

    if X_val.isna().any().any():
        raise ValueError(
            "Missing values detected in validation data."
        )

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