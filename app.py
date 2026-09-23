import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import ARTIFACT_DIR
from src.data_loader import load_transactions
from src.features import build_features
from src.model import load_model
from src.rag import LocalRAG
from src.llm import LocalInvestigationLLM, SYSTEM_PROMPT
from src.agents import SupervisorAgent
st.set_page_config(page_title="Fraud Detection Investigation System",layout="wide")
st.title("Fraud Detection Investigation System")
st.caption("Machine learning, RAG, local LLM reasoning, explicit agents, and Streamlit.")
if not (ARTIFACT_DIR/"fraud_lgbm.joblib").exists() or not (ARTIFACT_DIR/"threshold.json").exists(): st.error("Run python run_full_system.py first."); st.stop()
@st.cache_resource
def get_model(): print("[STREAMLIT] Loading model..."); return load_model()
@st.cache_data
def get_data(): print("[STREAMLIT] Loading data..."); return load_transactions()
@st.cache_data
def get_features(df): print("[STREAMLIT] Building features..."); return build_features(df)
@st.cache_resource
def get_rag(): print("[STREAMLIT] Loading RAG..."); return LocalRAG()
@st.cache_resource
def get_llm(): print("[STREAMLIT] Loading local LLM..."); return LocalInvestigationLLM()
model=get_model(); df=get_data(); X=get_features(df); threshold=float(json.loads((ARTIFACT_DIR/"threshold.json").read_text())['threshold'])
metrics=json.loads((ARTIFACT_DIR/"metrics.json").read_text()) if (ARTIFACT_DIR/"metrics.json").exists() else {}; ragm=json.loads((ARTIFACT_DIR/"rag_metrics.json").read_text()) if (ARTIFACT_DIR/"rag_metrics.json").exists() else {}; agm=json.loads((ARTIFACT_DIR/"agent_metrics.json").read_text()) if (ARTIFACT_DIR/"agent_metrics.json").exists() else {}
t1,t2,t3,t4=st.tabs(["Dashboard","Transaction Investigation","RAG Assistant","System Evaluation"])
with t1:
    a,b,c,d=st.columns(4); a.metric("Transactions",f"{len(df):,}"); b.metric("Observed fraud rate",f"{df['TX_FRAUD'].mean():.3%}"); c.metric("Threshold",f"{threshold:.3f}"); d.metric("Test PR-AUC",f"{metrics.get('pr_auc',0):.4f}")
    n=min(200000,len(X)); z=df.tail(n).copy(); z["fraud_probability"]=model.predict_proba(X.tail(n))[:,1]; z["decision"]=(z.fraud_probability>=threshold).map({True:"REVIEW",False:"ALLOW"}); st.plotly_chart(px.histogram(z,x="fraud_probability",color="decision",nbins=50,title="Predicted fraud probability distribution"),use_container_width=True); st.dataframe(z.sort_values("fraud_probability",ascending=False).head(100),use_container_width=True)
with t2:
    ids=df["TRANSACTION_ID"].astype(str).tolist(); selected=st.selectbox("Transaction ID",ids); idx=df["TRANSACTION_ID"].astype(str).eq(selected).idxmax(); p=float(model.predict_proba(X.iloc[[idx]])[0,1]); a,b,c=st.columns(3); a.metric("Fraud probability",f"{p:.4f}"); b.metric("Decision","REVIEW" if p>=threshold else "ALLOW"); c.metric("Threshold",f"{threshold:.4f}")
    if st.button("Run investigation",type="primary"):
        with st.spinner("Running agents, SHAP, RAG and local LLM..."):
            report=SupervisorAgent(model,X,get_rag(),get_llm()).run(df.iloc[idx],threshold)
        st.subheader("Investigation report"); st.markdown(report["llm_report"])
        for key in ["transaction","customer","terminal","model","retrieved_knowledge"]:
            with st.expander(key.replace("_"," ").title()): st.json(report[key])
with t3:
    q=st.text_input("Ask about fraud features, thresholds, methodology or agents")
    if q:
        rag=get_rag(); results=rag.retrieve(q,5); context="\n\n".join(f"SOURCE: {r['source']}\n{r['text']}" for r in results); prompt=f"Answer only from this retrieved knowledge. Mention supporting source filenames.\nQUESTION: {q}\nKNOWLEDGE:\n{context}"; st.markdown(get_llm().generate(SYSTEM_PROMPT,prompt));
        with st.expander("Retrieved sources"):
            for r in results: st.markdown(f"**{r['source']}** | similarity={r['score']:.3f}"); st.write(r['text'])
with t4:
    st.subheader("Fraud model"); st.json(metrics); st.subheader("RAG retrieval"); st.json(ragm); st.subheader("Agent workflow"); st.json(agm)
