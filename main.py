from src.pipeline import run as run_ml
from src.investigation_pipeline import run_cases
from src.evaluate_investigation import evaluate_agent_cases

def main():
    print("="*100); print("FULL FRAUD DETECTION + RAG + LLM + AGENT INVESTIGATION SYSTEM"); print("="*100)
    print(" ML, explainability and RAG"); run_ml()
    print(" Local LLM and autonomous agents"); run_cases(5); evaluate_agent_cases()
    
if __name__=="__main__": main()
