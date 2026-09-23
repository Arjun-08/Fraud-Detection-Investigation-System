import json
from dataclasses import dataclass
import pandas as pd
from .config import CUSTOMER_ID, TERMINAL_ID, TRANSACTION_ID, TIMESTAMP, AMOUNT
from .explain import explain_rows
from .rag import LocalRAG
from .llm import LocalInvestigationLLM, SYSTEM_PROMPT

@dataclass
class AgentContext:
    row: pd.Series; feature_row: pd.Series; probability: float; threshold: float; model: object; X: pd.DataFrame

class TransactionAgent:
    def run(self,ctx):
        r=ctx.row; f=ctx.feature_row; out={"transaction_id":str(r[TRANSACTION_ID]),"timestamp":str(r[TIMESTAMP]),"amount":float(r[AMOUNT]),"fraud_probability":ctx.probability,"threshold":ctx.threshold,"decision":"REVIEW" if ctx.probability>=ctx.threshold else "ALLOW","customer_newness":int(f.get("customer_newness",0)),"terminal_newness":int(f.get("terminal_newness",0))}; print(f"[AGENT:TRANSACTION] {out['transaction_id']} -> {out['decision']} ({out['fraud_probability']:.4f})"); return out

class CustomerAgent:
    def run(self,ctx):
        f=ctx.feature_row; p=CUSTOMER_ID; out={"previous_transactions":int(f.get(f"{p}_prev_count",0)),"previous_amount_mean":float(f.get(f"{p}_prev_amount_mean",0)),"previous_fraud_rate":float(f.get(f"{p}_prev_fraud_rate",0)),"seconds_since_previous":float(f.get(f"{p}_seconds_since_prev",-1)),"count_last_60s":float(f.get(f"{p}_count_60s",0)),"count_last_3600s":float(f.get(f"{p}_count_3600s",0)),"count_last_86400s":float(f.get(f"{p}_count_86400s",0))}; print(f"[AGENT:CUSTOMER] previous_tx={out['previous_transactions']}, fraud_rate={out['previous_fraud_rate']:.4f}"); return out

class TerminalAgent:
    def run(self,ctx):
        f=ctx.feature_row; p=TERMINAL_ID; out={"previous_transactions":int(f.get(f"{p}_prev_count",0)),"previous_amount_mean":float(f.get(f"{p}_prev_amount_mean",0)),"previous_fraud_rate":float(f.get(f"{p}_prev_fraud_rate",0)),"seconds_since_previous":float(f.get(f"{p}_seconds_since_prev",-1)),"count_last_60s":float(f.get(f"{p}_count_60s",0)),"count_last_3600s":float(f.get(f"{p}_count_3600s",0)),"count_last_86400s":float(f.get(f"{p}_count_86400s",0))}; print(f"[AGENT:TERMINAL] previous_tx={out['previous_transactions']}, fraud_rate={out['previous_fraud_rate']:.4f}"); return out

class EvidenceAgent:
    def __init__(self,rag): self.rag=rag
    def run(self,ctx,customer,terminal):
        sv=explain_rows(ctx.model,ctx.X.iloc[[ctx.row.name]]).iloc[0]; top=sv.abs().sort_values(ascending=False).head(8).index; shap_e=[{"feature":f,"contribution":float(sv[f]),"direction":"risk-increasing" if sv[f]>0 else "risk-decreasing"} for f in top]
        query=f"Fraud investigation with probability {ctx.probability:.3f}; customer previous transactions {customer['previous_transactions']}; terminal previous transactions {terminal['previous_transactions']}."; retrieved=self.rag.retrieve(query,5); out={"model_evidence":{"fraud_probability":ctx.probability,"threshold":ctx.threshold,"decision":"REVIEW" if ctx.probability>=ctx.threshold else "ALLOW","shap":shap_e},"retrieved_knowledge":retrieved}; print(f"[AGENT:EVIDENCE] SHAP={len(shap_e)}, RAG={len(retrieved)}"); return out

class ReportAgent:
    def __init__(self,llm): self.llm=llm
    def run(self,transaction,customer,terminal,evidence):
        prompt=f"TRANSACTION AGENT\n{json.dumps(transaction,indent=2)}\n\nCUSTOMER AGENT\n{json.dumps(customer,indent=2)}\n\nTERMINAL AGENT\n{json.dumps(terminal,indent=2)}\n\nMODEL EVIDENCE\n{json.dumps(evidence['model_evidence'],indent=2)}\n\nRETRIEVED KNOWLEDGE\n{json.dumps(evidence['retrieved_knowledge'],indent=2)}\n\nWrite the investigation report using only this evidence."; narrative=self.llm.generate(SYSTEM_PROMPT,prompt); return {"transaction":transaction,"customer":customer,"terminal":terminal,"model":evidence["model_evidence"],"retrieved_knowledge":evidence["retrieved_knowledge"],"llm_report":narrative}

class SupervisorAgent:
    def __init__(self,model,X,rag,llm): self.model=model; self.X=X; self.tx=TransactionAgent(); self.customer=CustomerAgent(); self.terminal=TerminalAgent(); self.evidence=EvidenceAgent(rag); self.report=ReportAgent(llm)
    def run(self,row,threshold):
        print("-"*80); print(f"[SUPERVISOR] Starting case {row[TRANSACTION_ID]}"); pos=row.name; fx=self.X.iloc[pos]; p=float(self.model.predict_proba(self.X.iloc[[pos]])[0,1]); ctx=AgentContext(row,fx,p,threshold,self.model,self.X); tr=self.tx.run(ctx); cr=self.customer.run(ctx); ter=self.terminal.run(ctx); ev=self.evidence.run(ctx,cr,ter); report=self.report.run(tr,cr,ter,ev); print(f"[SUPERVISOR] Completed case {row[TRANSACTION_ID]}"); return report
