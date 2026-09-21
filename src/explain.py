import pandas as pd
import shap


def explain_rows(model, X_rows: pd.DataFrame, max_rows: int = 25):
    """Calculate local SHAP contributions.

    f(x) = E[f(X)] + sum_i phi_i

    phi_i is the contribution attributed to feature i.
    """
    rows = X_rows.head(max_rows).copy()
    print(f"[SHAP] Explaining {len(rows)} row(s)...")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(rows)

    if isinstance(values, list):
        values = values[-1]

    return pd.DataFrame(values, columns=rows.columns, index=rows.index)


def top_reason(shap_row: pd.Series, top_k: int = 5):
    ranked = shap_row.abs().sort_values(ascending=False).head(top_k)
    return [
        f"{feature}: {'increases' if shap_row[feature] > 0 else 'decreases'} risk"
        for feature in ranked.index
    ]


def generate_investigation_summary(
    probability: float,
    threshold: float,
    shap_values: pd.Series,
    feature_values: pd.Series,
):
    decision = "REVIEW" if probability >= threshold else "ALLOW"

    return {
        "decision": decision,
        "fraud_probability": round(float(probability), 6),
        "threshold": round(float(threshold), 6),
        "top_model_reasons": top_reason(shap_values),
        "amount": float(feature_values.get("amount", 0.0)),
        "customer_newness": int(feature_values.get("customer_newness", 0)),
        "terminal_newness": int(feature_values.get("terminal_newness", 0)),
    }
