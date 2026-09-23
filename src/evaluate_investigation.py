import json
from .config import ARTIFACT_DIR, INVESTIGATION_CASES_PATH

def evaluate_agent_cases():
    cases=json.loads(INVESTIGATION_CASES_PATH.read_text(encoding="utf-8")); n=max(len(cases),1); completed=sum(set(["transaction","customer","terminal","model","llm_report"]).issubset(c) for c in cases); evidence=sum(bool(c.get("model",{}).get("shap")) for c in cases); sources=sum(bool(c.get("retrieved_knowledge")) for c in cases); reports=sum(isinstance(c.get("llm_report"),str) and len(c["llm_report"])>40 for c in cases); m={"cases":len(cases),"case_completion_rate":completed/n,"evidence_presence_rate":evidence/n,"retrieved_source_rate":sources/n,"llm_report_presence_rate":reports/n}; (ARTIFACT_DIR/"agent_metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8"); print("[AGENT-EVAL] "+json.dumps(m,indent=2)); return m
