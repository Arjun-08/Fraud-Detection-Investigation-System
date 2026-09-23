import joblib
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from .config import MODEL_PATH, FEATURE_COLUMNS_PATH, RANDOM_STATE

def build_model(scale_pos_weight):
    print(f"[MODEL] Building LightGBM; scale_pos_weight={scale_pos_weight:.3f}")
    return LGBMClassifier(objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, scale_pos_weight=scale_pos_weight)

def train_model(X_train, y_train, X_valid, y_valid):
    fraud = max(int(y_train.sum()), 1); legit = int((y_train == 0).sum()); scale = legit / fraud
    model = build_model(scale); print("[MODEL] Training started...")
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=["auc", "average_precision"], callbacks=[early_stopping(100, verbose=False), log_evaluation(50)])
    print(f"[MODEL] Training complete. Best iteration: {model.best_iteration_}")
    return model

def save_model(model, feature_columns):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, MODEL_PATH); FEATURE_COLUMNS_PATH.write_text("\n".join(feature_columns), encoding="utf-8")
    print(f"[MODEL] Saved model: {MODEL_PATH}")

def load_model(): print(f"[MODEL] Loading: {MODEL_PATH}"); return joblib.load(MODEL_PATH)
