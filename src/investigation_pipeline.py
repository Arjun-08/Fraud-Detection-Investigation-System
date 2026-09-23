import json
import numpy as np
from .config import DATA_PATH, THRESHOLD_PATH, INVESTIGATION_CASES_PATH
from .data_loader import load_transactions
from .features import build_features
from .model import load_model
from .rag import LocalRAG
from .llm import LocalInvestigationLLM
from .agents import SupervisorAgent

def run_cases(case_count=5):
    print("="*80); print("AUTONOMOUS FRAUD INVESTIGATION WORKFLOW"); print("="*80)
    df=load_transactions(DATA_PATH); X=build_features(df); model=load_model(); threshold=float(json.loads(THRESHOLD_PATH.read_text())['threshold']); probs=model.predict_proba(X)[:,1]
    start=int(len(df)*0.85); positions=np.arange(start,len(df)); ranked=positions[np.argsort(probs[start:])[::-1]][:case_count]; print(f"[INVESTIGATION] Selected {len(ranked)} high-risk cases from post-training test period.")
    rag=LocalRAG(); llm=LocalInvestigationLLM(); supervisor=SupervisorAgent(model,X,rag,llm); reports=[supervisor.run(df.iloc[pos],threshold) for pos in ranked]
    INVESTIGATION_CASES_PATH.write_text(json.dumps(reports,indent=2,default=str),encoding="utf-8"); print(f"[INVESTIGATION] Saved {len(reports)} cases to {INVESTIGATION_CASES_PATH}"); return reports
