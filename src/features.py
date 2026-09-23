import numpy as np
import pandas as pd
from .config import CUSTOMER_ID, TERMINAL_ID, TIMESTAMP, AMOUNT, TARGET, VELOCITY_WINDOWS_SECONDS, AMOUNT_WINDOWS_SECONDS

def _entity_features(df, entity_col):
    out = pd.DataFrame(index=df.index)
    work = df[[entity_col, TIMESTAMP, AMOUNT, TARGET]].copy()
    work["_row"] = np.arange(len(work))
    g = work.groupby(entity_col, sort=False, observed=True)
    work["_prev_count"] = g.cumcount()
    work["_prev_amount_sum"] = g[AMOUNT].cumsum().groupby(work[entity_col]).shift(1).fillna(0.0)
    work["_prev_fraud_sum"] = g[TARGET].cumsum().groupby(work[entity_col]).shift(1).fillna(0.0)
    out[f"{entity_col}_prev_count"] = work["_prev_count"].astype("int32")
    out[f"{entity_col}_prev_amount_mean"] = (work["_prev_amount_sum"] / work["_prev_count"].replace(0, np.nan)).fillna(0.0)
    out[f"{entity_col}_prev_fraud_rate"] = (work["_prev_fraud_sum"] / work["_prev_count"].replace(0, np.nan)).fillna(0.0)
    prev_time = g[TIMESTAMP].shift(1)
    out[f"{entity_col}_seconds_since_prev"] = (work[TIMESTAMP] - prev_time).dt.total_seconds().fillna(-1.0)
    # Grouped rolling counts are calculated on a sorted copy and mapped back by a stable row key.
    tmp = work.sort_values([entity_col, TIMESTAMP, "_row"], kind="stable").copy()
    for window in VELOCITY_WINDOWS_SECONDS:
        key = f"{entity_col}_count_{window}s"
        vals = tmp.groupby(entity_col, sort=False, observed=True).rolling(f"{window}s", on=TIMESTAMP, closed="left")[TARGET].count().reset_index()
        vals = vals.sort_values([entity_col, TIMESTAMP], kind="stable").reset_index(drop=True)
        vals["_row"] = tmp["_row"].to_numpy()
        out[key] = vals.set_index("_row")[TARGET].reindex(work["_row"]).fillna(0.0).to_numpy()
    for window in AMOUNT_WINDOWS_SECONDS:
        key = f"{entity_col}_amount_mean_{window}s"
        vals = tmp.groupby(entity_col, sort=False, observed=True).rolling(f"{window}s", on=TIMESTAMP, closed="left")[AMOUNT].mean().reset_index()
        vals = vals.sort_values([entity_col, TIMESTAMP], kind="stable").reset_index(drop=True)
        vals["_row"] = tmp["_row"].to_numpy()
        out[key] = vals.set_index("_row")[AMOUNT].reindex(work["_row"]).fillna(0.0).to_numpy()
    return out.reset_index(drop=True)

def build_features(df):
    print("[FEATURES] Starting leakage-safe feature engineering...")
    work = df.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True).copy()
    features = pd.DataFrame(index=work.index)
    amount = pd.to_numeric(work[AMOUNT], errors="coerce").fillna(0.0).clip(lower=0.0)
    features["amount"] = amount; features["log_amount"] = np.log1p(amount)
    ts = work[TIMESTAMP]
    features["hour"] = ts.dt.hour.astype("int8"); features["day_of_week"] = ts.dt.dayofweek.astype("int8"); features["day_of_month"] = ts.dt.day.astype("int8"); features["is_weekend"] = ts.dt.dayofweek.isin([5,6]).astype("int8")
    for entity in [CUSTOMER_ID, TERMINAL_ID]:
        if entity not in work.columns: raise ValueError(f"Required entity column missing: {entity}")
        print(f"[FEATURES] Building history for {entity}...")
        features = pd.concat([features.reset_index(drop=True), _entity_features(work, entity)], axis=1)
    cp = features[f"{CUSTOMER_ID}_prev_count"]; tp = features[f"{TERMINAL_ID}_prev_count"]
    features["customer_terminal_seen_before"] = ((cp > 0) & (tp > 0)).astype("int8")
    features["customer_newness"] = (cp == 0).astype("int8"); features["terminal_newness"] = (tp == 0).astype("int8")
    cm = features[f"{CUSTOMER_ID}_prev_amount_mean"]
    features["amount_vs_customer_mean"] = amount / (cm + 1e-6)
    features["log_amount_vs_customer_mean"] = np.log1p(features["amount_vs_customer_mean"].clip(lower=0))
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"[FEATURES] Completed: {features.shape[1]} features.")
    return features
