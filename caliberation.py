import os
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
)


VAL_PATH = (
    "artifacts/features/validation_features.pkl"
)

PREDICTION_DIR = (
    "artifacts/predictions"
)

RESULT_DIR = (
    "artifacts/results"
)

MODEL_DIR = (
    "artifacts/models"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_FILES = {
    "Logistic Regression":
        "logistic_validation_prob.npy",

    "Random Forest":
        "random_forest_validation_prob.npy",

    "XGBoost":
        "xgboost_validation_prob.npy",

    "CatBoost":
        "catboost_validation_prob.npy",
}


def sigmoid_calibration(
    probabilities,
    y_true
):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )

    logits = np.log(
        probabilities /
        (1 - probabilities)
    )

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs"
    )

    calibrator.fit(
        logits.reshape(-1, 1),
        y_true
    )

    calibrated = calibrator.predict_proba(
        logits.reshape(-1, 1)
    )[:, 1]

    return calibrator, calibrated


def main():

    print("=" * 70)
    print("PROBABILITY CALIBRATION")
    print("=" * 70)

    validation = pd.read_pickle(
        VAL_PATH
    )

    y_val = validation[
        "TX_FRAUD"
    ].astype(int).values

    results = []

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(
            PREDICTION_DIR,
            filename
        )

        if not os.path.exists(path):
            continue

        print(
            f"\nCalibrating: {model_name}"
        )

        probabilities = np.load(
            path
        )

        # ----------------------------------------------------
        # Raw model
        # ----------------------------------------------------

        raw_brier = brier_score_loss(
            y_val,
            probabilities
        )

        raw_pr_auc = average_precision_score(
            y_val,
            probabilities
        )

        raw_roc_auc = roc_auc_score(
            y_val,
            probabilities
        )

        # ----------------------------------------------------
        # Sigmoid calibration
        # ----------------------------------------------------

        calibrator, calibrated_prob = (
            sigmoid_calibration(
                probabilities,
                y_val
            )
        )

        calibrated_brier = (
            brier_score_loss(
                y_val,
                calibrated_prob
            )
        )

        calibrated_pr_auc = (
            average_precision_score(
                y_val,
                calibrated_prob
            )
        )

        calibrated_roc_auc = (
            roc_auc_score(
                y_val,
                calibrated_prob
            )
        )

        results.append({

            "Model": model_name,

            "Raw_Brier":
                raw_brier,

            "Calibrated_Brier":
                calibrated_brier,

            "Raw_PR_AUC":
                raw_pr_auc,

            "Calibrated_PR_AUC":
                calibrated_pr_auc,

            "Raw_ROC_AUC":
                raw_roc_auc,

            "Calibrated_ROC_AUC":
                calibrated_roc_auc
        })

        joblib_path = (
            f"{MODEL_DIR}/"
            f"{model_name.lower().replace(' ', '_')}"
            f"_calibrator.pkl"
        )

        import joblib

        joblib.dump(
            calibrator,
            joblib_path
        )

        np.save(
            f"{PREDICTION_DIR}/"
            f"{model_name.lower().replace(' ', '_')}"
            f"_calibrated_validation_prob.npy",
            calibrated_prob
        )

    results = pd.DataFrame(
        results
    )

    results.to_csv(
        f"{RESULT_DIR}/calibration_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nCalibration analysis completed."
    )


if __name__ == "__main__":
    main()