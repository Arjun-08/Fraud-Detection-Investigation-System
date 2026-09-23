import pandas as pd
import shap

def explain_rows(model, X_rows, max_rows=25):
    rows=X_rows.head(max_rows).copy(); print(f"[SHAP] Explaining {len(rows)} row(s)..."); values=shap.TreeExplainer(model).shap_values(rows)
    if isinstance(values,list): values=values[-1]
    return pd.DataFrame(values,columns=rows.columns,index=rows.index)

def top_reason(shap_row, top_k=5):
    ranked=shap_row.abs().sort_values(ascending=False).head(top_k)
    return [f"{f}: {'increases' if shap_row[f]>0 else 'decreases'} risk" for f in ranked.index]
