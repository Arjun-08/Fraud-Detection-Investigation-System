import json

from .config import ARTIFACT_DIR, TARGET, TRAIN_FRAC, VALID_FRAC
from .data_loader import load_transactions
from .split import chronological_split, save_splits
from .features import build_features
from .model import train_model, save_model
from .evaluate import (
    score_model, evaluate_at_threshold, optimize_threshold,
    save_threshold, save_metrics
)


def run():
    print("=" * 80)
    print("FRAUD DETECTION INVESTIGATION SYSTEM")
    print("END-TO-END TRAINING PIPELINE")
    print("=" * 80)

    print("\n[STAGE 1/6] Loading data")
    df = load_transactions()

    print("\n[STAGE 2/6] Creating chronological splits")
    train_df, valid_df, test_df = chronological_split(df)
    save_splits(train_df, valid_df, test_df)

    print("\n[STAGE 3/6] Building leakage-safe features")
    X_all = build_features(df)
    y_all = df[TARGET].astype("int8").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * (TRAIN_FRAC + VALID_FRAC))

    X_train = X_all.iloc[:train_end]
    X_valid = X_all.iloc[train_end:valid_end]
    X_test = X_all.iloc[valid_end:]

    y_train = y_all.iloc[:train_end]
    y_valid = y_all.iloc[train_end:valid_end]
    y_test = y_all.iloc[valid_end:]

    X_train.to_parquet(ARTIFACT_DIR / "X_train.parquet", index=False)
    X_valid.to_parquet(ARTIFACT_DIR / "X_valid.parquet", index=False)
    X_test.to_parquet(ARTIFACT_DIR / "X_test.parquet", index=False)

    print(f"[FEATURES] Train={X_train.shape}; Valid={X_valid.shape}; Test={X_test.shape}")

    print("\n[STAGE 4/6] Training model")
    model = train_model(X_train, y_train, X_valid, y_valid)
    save_model(model, X_train.columns.tolist())

    print("\n[STAGE 5/6] Selecting operating threshold")
    p_valid = score_model(model, X_valid)
    threshold_info = optimize_threshold(y_valid.to_numpy(), p_valid)
    save_threshold(threshold_info)
    threshold = threshold_info["threshold"]

    print("\n[STAGE 6/6] Final evaluation on untouched test set")
    p_test = score_model(model, X_test)
    test_metrics = evaluate_at_threshold(
        y_test.to_numpy(), p_test, threshold
    )
    test_metrics["threshold_source"] = "validation_only"
    test_metrics["n_test"] = int(len(y_test))
    test_metrics["fraud_rate_test"] = float(y_test.mean())

    print(json.dumps(test_metrics, indent=2))
    save_metrics(test_metrics)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run()
