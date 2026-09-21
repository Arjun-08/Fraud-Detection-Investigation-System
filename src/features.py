import numpy as np
import pandas as pd

from .config import (
    CUSTOMER_ID, TERMINAL_ID, TIMESTAMP, AMOUNT, TARGET,
    VELOCITY_WINDOWS_SECONDS, AMOUNT_WINDOWS_SECONDS
)


def _entity_features(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Create features using only events before the current transaction.

    Count:
        N_t(W) = sum_j I(t-W <= t_j < t)

    Previous fraud rate:
        F_t = previous fraud count / previous transaction count

    The current row is excluded from all historical calculations.
    """
    out = pd.DataFrame(index=df.index)

    work = df[[entity_col, TIMESTAMP, AMOUNT, TARGET]].copy()
    work["_row_order"] = np.arange(len(work))

    g = work.groupby(entity_col, sort=False, observed=True)

    work["_prev_count"] = g.cumcount()
    work["_prev_amount_sum"] = g[AMOUNT].cumsum().groupby(work[entity_col]).shift(1)
    work["_prev_fraud_sum"] = g[TARGET].cumsum().groupby(work[entity_col]).shift(1)

    first = work["_prev_count"].eq(0)
    work.loc[first, "_prev_amount_sum"] = 0.0
    work.loc[first, "_prev_fraud_sum"] = 0.0

    out[f"{entity_col}_prev_count"] = work["_prev_count"].astype("int32")
    out[f"{entity_col}_prev_amount_mean"] = (
        work["_prev_amount_sum"] /
        work["_prev_count"].replace(0, np.nan)
    ).fillna(0.0)

    out[f"{entity_col}_prev_fraud_rate"] = (
        work["_prev_fraud_sum"] /
        work["_prev_count"].replace(0, np.nan)
    ).fillna(0.0)

    prev_time = g[TIMESTAMP].shift(1)
    out[f"{entity_col}_seconds_since_prev"] = (
        work[TIMESTAMP] - prev_time
    ).dt.total_seconds().fillna(-1.0)

    # Efficient online-compatible velocity features.
    # We use rolling windows on each entity and close='left' so the current
    # transaction is not counted.
    tmp = work.sort_values([entity_col, TIMESTAMP, "_row_order"], kind="stable")
    tmp_index = tmp.set_index(TIMESTAMP)

    for window in VELOCITY_WINDOWS_SECONDS:
        key = f"{entity_col}_count_{window}s"
        rolled = (
            tmp_index.groupby(entity_col, sort=False)[TARGET]
            .rolling(f"{window}s", closed="left")
            .count()
            .to_numpy()
        )
        if len(rolled) == len(tmp):
            # tmp is ordered by entity/time, which is also the order emitted by
            # groupby(..., sort=False). Map values back using the stable row key.
            temp = pd.Series(rolled, index=tmp["_row_order"].to_numpy())
            out[key] = temp.reindex(work["_row_order"]).fillna(0.0).to_numpy()
        else:
            out[key] = 0.0

    for window in AMOUNT_WINDOWS_SECONDS:
        key = f"{entity_col}_amount_mean_{window}s"
        rolled = (
            tmp_index.groupby(entity_col, sort=False)[AMOUNT]
            .rolling(f"{window}s", closed="left")
            .mean()
            .to_numpy()
        )
        if len(rolled) == len(tmp):
            temp = pd.Series(rolled, index=tmp["_row_order"].to_numpy())
            out[key] = temp.reindex(work["_row_order"]).fillna(0.0).to_numpy()
        else:
            out[key] = 0.0

    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[FEATURES] Starting leakage-safe feature engineering...")

    work = df.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True).copy()
    features = pd.DataFrame(index=work.index)

    amount = pd.to_numeric(work[AMOUNT], errors="coerce").fillna(0.0).clip(lower=0.0)
    features["amount"] = amount
    features["log_amount"] = np.log1p(amount)

    ts = work[TIMESTAMP]
    features["hour"] = ts.dt.hour.astype("int8")
    features["day_of_week"] = ts.dt.dayofweek.astype("int8")
    features["day_of_month"] = ts.dt.day.astype("int8")
    features["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")

    for entity in [CUSTOMER_ID, TERMINAL_ID]:
        if entity not in work.columns:
            raise ValueError(f"Required entity column missing: {entity}")
        print(f"[FEATURES] Building history for {entity}...")
        hist = _entity_features(work, entity)
        features = pd.concat([features.reset_index(drop=True), hist], axis=1)

    customer_prev = features[f"{CUSTOMER_ID}_prev_count"]
    terminal_prev = features[f"{TERMINAL_ID}_prev_count"]

    features["customer_terminal_seen_before"] = (
        (customer_prev > 0) & (terminal_prev > 0)
    ).astype("int8")
    features["customer_newness"] = (customer_prev == 0).astype("int8")
    features["terminal_newness"] = (terminal_prev == 0).astype("int8")

    customer_mean = features[f"{CUSTOMER_ID}_prev_amount_mean"]
    features["amount_vs_customer_mean"] = amount / (customer_mean + 1e-6)
    features["log_amount_vs_customer_mean"] = np.log1p(
        features["amount_vs_customer_mean"].clip(lower=0)
    )

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[FEATURES] Completed: {features.shape[1]} features.")
    return features
