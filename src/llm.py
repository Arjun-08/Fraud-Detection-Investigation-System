import torch
from transformers import pipeline
from .config import LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE

SYSTEM_PROMPT = """You are a fraud investigation assistant. Use only the evidence supplied. Do not invent transaction facts, customer history, terminal history, policies, or external events. The classifier probability is model evidence, not proof of fraud. If evidence is unavailable, say so. Produce: 1) case assessment, 2) evidence supporting review or allow, 3) relevant retrieved knowledge, 4) evidence limitations, 5) recommended next review actions. Do not assign guilt or certainty."""

class LocalInvestigationLLM:
    def __init__(self):
        print(f"[LLM] Loading local model: {LLM_MODEL}"); self.generator=pipeline("text-generation",model=LLM_MODEL,device=0 if torch.cuda.is_available() else -1); print(f"[LLM] Ready on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    def generate(self,system_prompt,user_prompt):
        print("[LLM] Generating investigation narrative..."); messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]; out=self.generator(messages,max_new_tokens=LLM_MAX_NEW_TOKENS,do_sample=False,temperature=LLM_TEMPERATURE,return_full_text=False)[0]["generated_text"]
        if isinstance(out,list): out=out[-1].get("content",str(out))
        print(f"[LLM] Generated {len(str(out))} characters."); return str(out).strip()
