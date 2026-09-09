import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
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

os.makedirs(
    RESULT_DIR,
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


def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        predictions
    )

    return {
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def main():

    print("=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)

    validation = pd.read_pickle(
        VAL_PATH
    )

    y_val = validation[
        "TX_FRAUD"
    ].astype(int).values

    all_results = []

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(
            PREDICTION_DIR,
            filename
        )

        if not os.path.exists(path):

            print(
                f"\nSkipping {model_name}: "
                f"prediction file not found."
            )

            continue

        print(
            f"\nOptimizing threshold: "
            f"{model_name}"
        )

        probabilities = np.load(
            path
        )

        for threshold in thresholds:

            metrics = evaluate_threshold(
                y_val,
                probabilities,
                threshold
            )

            metrics["Model"] = model_name

            all_results.append(
                metrics
            )

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        f"{RESULT_DIR}/threshold_sweep.csv",
        index=False
    )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_f1 = (
        results
        .sort_values(
            ["Model", "F1"],
            ascending=[True, False]
        )
        .groupby("Model")
        .head(1)
    )

    print("\n" + "=" * 70)
    print("BEST THRESHOLD BY F1")
    print("=" * 70)

    print(
        best_f1[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "MCC",
                "FPR",
                "FNR"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best MCC threshold
    # --------------------------------------------------------

    best_mcc = (
        results
        .sort_values(
            ["Model", "MCC"],
            ascending=[True, False]
        )
        .groupby("Model")
        .head(1)
    )

    print("\n" + "=" * 70)
    print("BEST THRESHOLD BY MCC")
    print("=" * 70)

    print(
        best_mcc[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "MCC",
                "FPR",
                "FNR"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save recommended threshold
    # --------------------------------------------------------

    threshold_config = best_f1[
        [
            "Model",
            "Threshold"
        ]
    ].copy()

    threshold_config.to_csv(
        f"{RESULT_DIR}/recommended_thresholds.csv",
        index=False
    )

    print(
        "\nThreshold optimization completed."
    )


if __name__ == "__main__":
    main()