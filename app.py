import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACT_DIR
from src.data_loader import load_transactions
from src.features import build_features
from src.model import load_model
from src.explain import explain_rows, generate_investigation_summary

st.set_page_config(
    page_title="Fraud Detection Investigation System",
    layout="wide",
)

st.title("Fraud Detection Investigation System")
st.caption(
    "Chronological fraud scoring, behavioral risk signals, "
    "and deterministic investigation support."
)

MODEL_PATH = ARTIFACT_DIR / "fraud_lgbm.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Model artifacts are missing. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def get_model():
    print("[STREAMLIT] Loading model...")
    return load_model()


@st.cache_data
def get_data():
    print("[STREAMLIT] Loading transaction data...")
    return load_transactions()


@st.cache_data
def get_features(df):
    print("[STREAMLIT] Building features...")
    return build_features(df)


model = get_model()
df = get_data()
X = get_features(df)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold = float(json.load(f)["threshold"])

metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

tab1, tab2, tab3 = st.tabs(
    ["Dashboard", "Transaction Investigation", "Model"]
)

with tab1:
    st.subheader("System overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Observed fraud rate", f"{df['TX_FRAUD'].mean():.3%}")
    c3.metric("Decision threshold", f"{threshold:.3f}")
    c4.metric("Test PR-AUC", f"{metrics.get('pr_auc', float('nan')):.4f}")

    print("[STREAMLIT] Computing dashboard predictions...")
    sample_n = min(200_000, len(X))
    dashboard_X = X.tail(sample_n)
    dashboard_df = df.tail(sample_n).copy()
    dashboard_df["fraud_probability"] = model.predict_proba(dashboard_X)[:, 1]
    dashboard_df["decision"] = (
        dashboard_df["fraud_probability"] >= threshold
    ).map({True: "REVIEW", False: "ALLOW"})

    fig = px.histogram(
        dashboard_df,
        x="fraud_probability",
        color="decision",
        nbins=50,
        title="Predicted fraud probability distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent high-risk transactions")
    st.dataframe(
        dashboard_df.sort_values("fraud_probability", ascending=False)
        .head(100),
        use_container_width=True,
    )

with tab2:
    st.subheader("Investigate a transaction")

    transaction_ids = df["TRANSACTION_ID"].astype(str).tolist()
    selected = st.selectbox("Transaction ID", transaction_ids)

    idx = df["TRANSACTION_ID"].astype(str).eq(selected).idxmax()
    row_X = X.iloc[[idx]]
    probability = float(model.predict_proba(row_X)[0, 1])
    decision = "REVIEW" if probability >= threshold else "ALLOW"

    a, b, c = st.columns(3)
    a.metric("Fraud probability", f"{probability:.4f}")
    b.metric("Decision", decision)
    c.metric("Threshold", f"{threshold:.4f}")

    shap_df = explain_rows(model, row_X, max_rows=1)
    shap_values = shap_df.iloc[0]

    summary = generate_investigation_summary(
        probability=probability,
        threshold=threshold,
        shap_values=shap_values,
        feature_values=row_X.iloc[0],
    )

    st.subheader("Investigation summary")
    st.json(summary)

    explanation = (
        shap_values.abs()
        .sort_values(ascending=False)
        .head(12)
        .rename("absolute_shap")
        .to_frame()
    )
    explanation["shap_value"] = shap_values[explanation.index]

    fig = px.bar(
        explanation.sort_values("shap_value"),
        x="shap_value",
        y=explanation.sort_values("shap_value").index,
        orientation="h",
        title="Local model explanation",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Evaluation")
    st.json(metrics)

    st.markdown(
        """
        The classifier produces a fraud probability from chronological
        behavioral features. The operating threshold is selected on
        validation data only; the test set is evaluated afterward using
        that frozen threshold.
        """
    )
