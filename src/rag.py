import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import KNOWLEDGE_DIR, ARTIFACT_DIR, EMBEDDING_MODEL, RAG_INDEX_PATH, RAG_METADATA_PATH

def load_documents():
    print(f"[RAG] Loading knowledge from: {KNOWLEDGE_DIR}"); docs=[]
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        paragraphs=[p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        for i,p in enumerate(paragraphs): docs.append({"source":path.name,"chunk_id":i,"text":p})
    print(f"[RAG] Loaded {len(docs)} knowledge chunks."); return docs

def build_index():
    docs=load_documents();
    if not docs: raise RuntimeError("Knowledge directory is empty.")
    print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}"); enc=SentenceTransformer(EMBEDDING_MODEL); texts=[d["text"] for d in docs]
    emb=enc.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype("float32"); index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb); ARTIFACT_DIR.mkdir(parents=True,exist_ok=True); faiss.write_index(index,str(RAG_INDEX_PATH)); RAG_METADATA_PATH.write_text(json.dumps(docs,indent=2),encoding="utf-8"); print(f"[RAG] Index built: {index.ntotal} vectors, dimension={emb.shape[1]}"); return index,docs

class LocalRAG:
    def __init__(self):
        print("[RAG] Initializing local retriever...")
        if not RAG_INDEX_PATH.exists() or not RAG_METADATA_PATH.exists(): build_index()
        self.index=faiss.read_index(str(RAG_INDEX_PATH)); self.documents=json.loads(RAG_METADATA_PATH.read_text(encoding="utf-8")); self.encoder=SentenceTransformer(EMBEDDING_MODEL); print(f"[RAG] Ready with {self.index.ntotal} chunks.")
    def retrieve(self,query,k=5):
        print(f"[RAG] Query: {query[:120]}"); v=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32"); scores,inds=self.index.search(v,min(k,self.index.ntotal)); out=[]
        for score,idx in zip(scores[0],inds[0]):
            if idx<0: continue
            item=dict(self.documents[int(idx)]); item["score"]=float(score); out.append(item)
        print("[RAG] Retrieved: "+", ".join(f"{x['source']}:{x['score']:.3f}" for x in out)); return out

def evaluate_retrieval():
    r=LocalRAG(); cases=[
        ("How are historical velocity features calculated?","02_behavioral_features.md"),
        ("Why is chronological splitting used?","03_temporal_evaluation.md"),
        ("How does the system handle class imbalance?","04_class_imbalance.md"),
        ("How is the decision threshold selected?","05_threshold_and_cost.md"),
        ("How are SHAP contributions interpreted?","06_shap_explainability.md"),
        ("How does retrieval augmented generation work here?","08_rag_methodology.md"),
        ("What does the customer agent do?","09_agent_roles.md"),
        ("What should the investigation report contain?","10_investigation_report_schema.md")]
    hits=0; rr=[]; k=5
    for q,expected in cases:
        res=r.retrieve(q,k); ranks=[i+1 for i,x in enumerate(res) if x["source"]==expected]
        if ranks: hits+=1; rr.append(1/ranks[0])
        else: rr.append(0)
    m={"queries":len(cases),"k":k,"recall_at_k":hits/len(cases),"mrr":sum(rr)/len(rr)}; p=ARTIFACT_DIR/"rag_metrics.json"; p.write_text(json.dumps(m,indent=2),encoding="utf-8"); print(f"[RAG-EVAL] Recall@{k}={m['recall_at_k']:.4f}; MRR={m['mrr']:.4f}"); return m
