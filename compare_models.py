import os
import pandas as pd


RESULT_DIR = "artifacts/results"


def main():

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    baseline_path = (
        f"{RESULT_DIR}/baseline_results.csv"
    )

    boosting_path = (
        f"{RESULT_DIR}/boosting_results.csv"
    )

    baseline = pd.read_csv(
        baseline_path
    )

    boosting = pd.read_csv(
        boosting_path
    )

    results = pd.concat(
        [
            baseline,
            boosting
        ],
        ignore_index=True
    )

    results = results.sort_values(
        "PR_AUC",
        ascending=False
    )

    results.to_csv(
        f"{RESULT_DIR}/model_comparison.csv",
        index=False
    )

    display_columns = [
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
        "Train_Time_sec"
    ]

    print(
        results[display_columns].to_string(
            index=False
        )
    )

    best_model = results.iloc[0]["Model"]

    print("\n" + "=" * 70)
    print(
        f"BEST MODEL BY VALIDATION PR-AUC: {best_model}"
    )
    print("=" * 70)

    print(
        f"PR-AUC : {results.iloc[0]['PR_AUC']:.6f}"
    )

    print(
        f"ROC-AUC: {results.iloc[0]['ROC_AUC']:.6f}"
    )

    print(
        "\nComparison saved to:"
        f" {RESULT_DIR}/model_comparison.csv"
    )


if __name__ == "__main__":
    main()