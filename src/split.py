from pathlib import Path
import pandas as pd

from .config import TRAIN_FRAC, VALID_FRAC, TEST_FRAC, ARTIFACT_DIR, TARGET, TIMESTAMP


def chronological_split(df: pd.DataFrame):
    """Split the stream chronologically so future transactions never train the model."""
    assert abs(TRAIN_FRAC + VALID_FRAC + TEST_FRAC - 1.0) < 1e-9

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    valid_end = int(n * (TRAIN_FRAC + VALID_FRAC))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    print("[SPLIT] Chronological split created.")
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        print(
            f"[SPLIT] {name:10s}: rows={len(part):,}, "
            f"fraud={int(part[TARGET].sum()):,}, "
            f"fraud_rate={part[TARGET].mean():.4%}, "
            f"time={part[TIMESTAMP].min()} -> {part[TIMESTAMP].max()}"
        )

    if train[TIMESTAMP].max() > valid[TIMESTAMP].min():
        raise AssertionError("Temporal leakage detected between train and validation.")
    if valid[TIMESTAMP].max() > test[TIMESTAMP].min():
        raise AssertionError("Temporal leakage detected between validation and test.")

    return train, valid, test


def save_splits(train, valid, test):
    out = ARTIFACT_DIR / "splits"
    out.mkdir(parents=True, exist_ok=True)
    train.to_pickle(out / "train.pkl", protocol=4)
    valid.to_pickle(out / "validation.pkl", protocol=4)
    test.to_pickle(out / "test.pkl", protocol=4)
    print(f"[SPLIT] Saved splits to {out}")
