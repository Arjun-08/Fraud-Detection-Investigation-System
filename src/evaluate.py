import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, log_loss
from .config import ARTIFACT_DIR, MAX_REVIEW_RATE, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, COST_REVIEW

def score_model(model, X): return model.predict_proba(X)[:,1]

def evaluate_at_threshold(y_true, proba, threshold):
    pred=(proba>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
    return {"threshold":float(threshold),"roc_auc":float(roc_auc_score(y_true,proba)),"pr_auc":float(average_precision_score(y_true,proba)),"precision":float(precision_score(y_true,pred,zero_division=0)),"recall":float(recall_score(y_true,pred,zero_division=0)),"f1":float(f1_score(y_true,pred,zero_division=0)),"mcc":float(matthews_corrcoef(y_true,pred)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"review_rate":float(pred.mean()),"log_loss":float(log_loss(y_true,proba,labels=[0,1]))}

def business_cost(y_true, proba, threshold):
    pred=(proba>=threshold).astype(int); review=pred==1; fn=(y_true==1)&(pred==0); fp=(y_true==0)&(pred==1)
    return COST_FALSE_NEGATIVE*fn.sum()+COST_FALSE_POSITIVE*fp.sum()+COST_REVIEW*review.sum()

def optimize_threshold(y_valid,p_valid):
    print("[THRESHOLD] Searching validation thresholds..."); candidates=[]
    for t in np.linspace(0.01,0.99,197):
        m=evaluate_at_threshold(y_valid,p_valid,t)
        if m["review_rate"]<=MAX_REVIEW_RATE: m["business_cost"]=float(business_cost(y_valid,p_valid,t)); candidates.append(m)
    if not candidates: raise RuntimeError("No threshold satisfies MAX_REVIEW_RATE.")
    best=min(candidates,key=lambda x:x["business_cost"]); print(f"[THRESHOLD] Selected {best['threshold']:.3f}; F1={best['f1']:.4f}; review_rate={best['review_rate']:.4%}; cost={best['business_cost']:.2f}"); return best

def save_threshold(info):
    p=ARTIFACT_DIR/"threshold.json"; p.write_text(json.dumps(info,indent=2),encoding="utf-8"); print(f"[THRESHOLD] Saved: {p}")

def save_metrics(metrics):
    p=ARTIFACT_DIR/"metrics.json"; p.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(f"[EVAL] Saved: {p}")
