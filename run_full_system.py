from src.pipeline import run as run_ml
from src.investigation_pipeline import run_cases
from src.evaluate_investigation import evaluate_agent_cases

def main():
    print("="*100); print("FULL FRAUD DETECTION + RAG + LLM + AGENT INVESTIGATION SYSTEM"); print("="*100)
    print("[PHASES 1-10] ML, explainability and RAG"); run_ml()
    print("[PHASES 11-13] Local LLM and autonomous agents"); run_cases(5); evaluate_agent_cases()
    print("="*100); print("FULL SYSTEM COMPLETE"); print("Run: streamlit run app.py"); print("="*100)
if __name__=="__main__": main()
