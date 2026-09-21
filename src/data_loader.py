from pathlib import Path
import pandas as pd

from .config import DATA_PATH, TARGET, TIMESTAMP


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the user's local transaction table."""
    print(f"[DATA] Loading dataset from: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the transaction dataset there or change DATA_PATH in src/config.py."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    print(f"[DATA] Loaded {len(df):,} rows x {df.shape[1]:,} columns.")

    required = {TARGET, TIMESTAMP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    if df[TIMESTAMP].isna().any():
        raise ValueError(f"{TIMESTAMP} contains invalid timestamps.")

    df = df.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True)

    print(f"[DATA] Time range: {df[TIMESTAMP].min()} -> {df[TIMESTAMP].max()}")
    print(f"[DATA] Fraud count: {int(df[TARGET].sum()):,}")
    print(f"[DATA] Fraud rate: {df[TARGET].mean():.4%}")
    return df
