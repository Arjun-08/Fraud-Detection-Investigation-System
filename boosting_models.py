import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "artifacts/features/train_features.pkl"
VAL_PATH = "artifacts/features/validation_features.pkl"

MODEL_DIR = "artifacts/models"
PREDICTION_DIR = "artifacts/predictions"
RESULT_DIR = "artifacts/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


RANDOM_STATE = 42


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(y_true, probabilities, threshold=0.5):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            predictions
        ),

        "FPR": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),

        "FNR": (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        ),

        "Brier": brier_score_loss(
            y_true,
            probabilities
        ),

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ============================================================
# Load data
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train = pd.read_pickle(
        TRAIN_PATH
    )

    validation = pd.read_pickle(
        VAL_PATH
    )

    target = "TX_FRAUD"

    X_train = train.drop(
        columns=[target]
    )

    y_train = train[target].astype(int)

    X_val = validation.drop(
        columns=[target]
    )

    y_val = validation[target].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            "Train and validation features do not match."
        )

    print(
        f"Train shape      : {X_train.shape}"
    )

    print(
        f"Validation shape : {X_val.shape}"
    )

    print(
        f"Train fraud rate : {y_train.mean():.4%}"
    )

    print(
        f"Val fraud rate   : {y_val.mean():.4%}"
    )

    return X_train, y_train, X_val, y_val


# ============================================================
# Main
# ============================================================

def main():

    X_train, y_train, X_val, y_val = load_data()

    # Fraud imbalance
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.2f}"
    )

    all_results = []

    # ========================================================
    # Random Forest
    # ========================================================

    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()

    rf.fit(
        X_train,
        y_train
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    rf_prob = rf.predict_proba(
        X_val
    )[:, 1]

    inference_time = time.perf_counter() - start

    rf_metrics = evaluate_model(
        y_val,
        rf_prob
    )

    rf_metrics["Model"] = "Random Forest"
    rf_metrics["Train_Time_sec"] = train_time
    rf_metrics["Inference_Time_sec"] = inference_time

    all_results.append(
        rf_metrics
    )

    print(
        pd.Series(rf_metrics)
    )

    joblib.dump(
        rf,
        f"{MODEL_DIR}/random_forest.pkl"
    )

    np.save(
        f"{PREDICTION_DIR}/random_forest_validation_prob.npy",
        rf_prob
    )

    # ========================================================
    # XGBoost
    # ========================================================

    print("\n" + "=" * 70)
    print("XGBOOST")
    print("=" * 70)

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,

        objective="binary:logistic",
        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",
        device="cpu",

        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    start = time.perf_counter()

    xgb.fit(
        X_train,
        y_train,

        eval_set=[
            (X_val, y_val)
        ],

        verbose=False
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    xgb_prob = xgb.predict_proba(
        X_val
    )[:, 1]

    inference_time = time.perf_counter() - start

    xgb_metrics = evaluate_model(
        y_val,
        xgb_prob
    )

    xgb_metrics["Model"] = "XGBoost"
    xgb_metrics["Train_Time_sec"] = train_time
    xgb_metrics["Inference_Time_sec"] = inference_time

    all_results.append(
        xgb_metrics
    )

    print(
        pd.Series(xgb_metrics)
    )

    joblib.dump(
        xgb,
        f"{MODEL_DIR}/xgboost.pkl"
    )

    np.save(
        f"{PREDICTION_DIR}/xgboost_validation_prob.npy",
        xgb_prob
    )

    # ========================================================
    # CatBoost
    # ========================================================

    print("\n" + "=" * 70)
    print("CATBOOST")
    print("=" * 70)

    catboost = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,

        loss_function="Logloss",
        eval_metric="PRAUC",

        auto_class_weights="Balanced",

        random_seed=RANDOM_STATE,

        thread_count=-1,

        verbose=False,

        task_type="CPU"
    )

    start = time.perf_counter()

    catboost.fit(
        X_train,
        y_train,
        eval_set=(
            X_val,
            y_val
        ),
        verbose=False
    )

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    cat_prob = catboost.predict_proba(
        X_val
    )[:, 1]

    inference_time = time.perf_counter() - start

    cat_metrics = evaluate_model(
        y_val,
        cat_prob
    )

    cat_metrics["Model"] = "CatBoost"
    cat_metrics["Train_Time_sec"] = train_time
    cat_metrics["Inference_Time_sec"] = inference_time

    all_results.append(
        cat_metrics
    )

    print(
        pd.Series(cat_metrics)
    )

    joblib.dump(
        catboost,
        f"{MODEL_DIR}/catboost.pkl"
    )

    np.save(
        f"{PREDICTION_DIR}/catboost_validation_prob.npy",
        cat_prob
    )

    # ========================================================
    # Save comparison
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    columns = [
        "Model",
        "PR_AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "FPR",
        "FNR",
        "Brier",
        "TP",
        "TN",
        "FP",
        "FN",
        "Train_Time_sec",
        "Inference_Time_sec"
    ]

    results = results[columns]

    results.to_csv(
        f"{RESULT_DIR}/boosting_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("BOOSTING MODEL RESULTS")
    print("=" * 70)

    print(
        results.sort_values(
            "PR_AUC",
            ascending=False
        ).to_string(
            index=False
        )
    )

    print(
        "\nPhase 3 model training completed."
    )


if __name__ == "__main__":
    main()