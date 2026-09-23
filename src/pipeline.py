import json
from .config import ARTIFACT_DIR, TARGET, TRAIN_FRAC, VALID_FRAC
from .data_loader import load_transactions
from .split import chronological_split, save_splits
from .features import build_features
from .model import train_model, save_model
from .evaluate import score_model, evaluate_at_threshold, optimize_threshold, save_threshold, save_metrics
from .rag import build_index, evaluate_retrieval

def run():
    print("="*80); print("FRAUD DETECTION CORE PIPELINE"); print("="*80)
    print("[STAGE 1/8] Loading data"); df=load_transactions()
    print("[STAGE 2/8] Chronological split"); train_df,valid_df,test_df=chronological_split(df); save_splits(train_df,valid_df,test_df)
    print("[STAGE 3/8] Leakage-safe features"); X=build_features(df); y=df[TARGET].astype("int8").reset_index(drop=True)
    n=len(df); te=int(n*TRAIN_FRAC); ve=int(n*(TRAIN_FRAC+VALID_FRAC)); Xtr,Xv,Xte=X.iloc[:te],X.iloc[te:ve],X.iloc[ve:]; ytr,yv,yte=y.iloc[:te],y.iloc[te:ve],y.iloc[ve:]
    Xtr.to_parquet(ARTIFACT_DIR/"X_train.parquet",index=False); Xv.to_parquet(ARTIFACT_DIR/"X_valid.parquet",index=False); Xte.to_parquet(ARTIFACT_DIR/"X_test.parquet",index=False)
    print("[STAGE 4/8] Training model"); model=train_model(Xtr,ytr,Xv,yv); save_model(model,Xtr.columns.tolist())
    print("[STAGE 5/8] Validation threshold"); pv=score_model(model,Xv); ti=optimize_threshold(yv.to_numpy(),pv); save_threshold(ti); threshold=ti["threshold"]
    print("[STAGE 6/8] Untouched test evaluation"); pt=score_model(model,Xte); tm=evaluate_at_threshold(yte.to_numpy(),pt,threshold); tm.update({"threshold_source":"validation_only","n_test":int(len(yte)),"fraud_rate_test":float(yte.mean())}); print(json.dumps(tm,indent=2)); save_metrics(tm)
    print("[STAGE 7/8] Building local RAG index"); build_index()
    print("[STAGE 8/8] Evaluating RAG retrieval"); evaluate_retrieval()
    print("="*80); print("CORE ML + RAG PIPELINE COMPLETE"); print("="*80)
