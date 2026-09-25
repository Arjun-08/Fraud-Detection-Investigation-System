from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "transactions.pkl"


# ---------------------------------------------------------------------
# Expected columns
# ---------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
}


# ---------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------

def prepare_dataset():

    print("=" * 70)
    print("FRAUD DETECTION DATA PREPARATION")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Raw data    : {RAW_DIR}")
    print(f"Output      : {OUTPUT_FILE}")

    # -------------------------------------------------------------
    # Check raw directory
    # -------------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"\nRaw data directory does not exist:\n{RAW_DIR}\n\n"
            "Download the Fraud Detection Handbook dataset first."
        )

    # -------------------------------------------------------------
    # Find pickle files
    # -------------------------------------------------------------

    files = sorted(RAW_DIR.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(
            f"\nNo .pkl files found in:\n{RAW_DIR}\n\n"
            "Expected files such as:\n"
            "2018-04-01.pkl\n"
            "2018-04-02.pkl\n"
            "..."
        )

    print(f"\nFound {len(files)} raw pickle files.")

    print("\nFirst files:")
    for file in files[:5]:
        print(f"  {file.name}")

    if len(files) > 5:
        print("  ...")

    # -------------------------------------------------------------
    # Load daily files
    # -------------------------------------------------------------

    dataframes = []

    print("\nLoading daily transaction files...")

    for i, file in enumerate(files, start=1):

        print(
            f"[{i:03d}/{len(files):03d}] "
            f"Loading {file.name}"
        )

        df = pd.read_pickle(file)

        # ---------------------------------------------------------
        # Validate columns
        # ---------------------------------------------------------

        missing_columns = EXPECTED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"\nMissing columns in {file.name}: "
                f"{sorted(missing_columns)}"
            )

        dataframes.append(df)

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    print("\nCombining daily datasets...")

    transactions = pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )

    print(
        f"Combined shape: "
        f"{transactions.shape}"
    )

    # -------------------------------------------------------------
    # Convert datetime
    # -------------------------------------------------------------

    transactions["TX_DATETIME"] = pd.to_datetime(
        transactions["TX_DATETIME"]
    )

    # -------------------------------------------------------------
    # Create transaction ID
    # -------------------------------------------------------------

    if "TRANSACTION_ID" not in transactions.columns:

        print(
            "\nTRANSACTION_ID not present. "
            "Creating sequential transaction IDs..."
        )

        transactions.insert(
            0,
            "TRANSACTION_ID",
            range(len(transactions)),
        )

    # -------------------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------------------

    print("\nSorting transactions chronologically...")

    transactions = transactions.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Validate fraud label
    # -------------------------------------------------------------

    if "TX_FRAUD" not in transactions.columns:

        raise ValueError(
            "\nTX_FRAUD column was not found.\n"
            "The downloaded dataset does not appear to be "
            "the labeled Fraud Detection Handbook dataset."
        )

    transactions["TX_FRAUD"] = transactions["TX_FRAUD"].astype(int)

    # -------------------------------------------------------------
    # Dataset statistics
    # -------------------------------------------------------------

    total_rows = len(transactions)

    fraud_count = int(
        transactions["TX_FRAUD"].sum()
    )

    legitimate_count = total_rows - fraud_count

    fraud_rate = fraud_count / total_rows

    start_date = transactions["TX_DATETIME"].min()
    end_date = transactions["TX_DATETIME"].max()

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Rows              : {total_rows:,}")
    print(f"Columns           : {transactions.shape[1]}")
    print(f"Legitimate        : {legitimate_count:,}")
    print(f"Fraud             : {fraud_count:,}")
    print(f"Fraud rate        : {fraud_rate:.4%}")
    print(f"Start             : {start_date}")
    print(f"End               : {end_date}")

    print("\nColumns:")

    for column in transactions.columns:
        print(f"  - {column}")

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nSaving combined dataset to:\n"
        f"{OUTPUT_FILE}"
    )

    transactions.to_pickle(
        OUTPUT_FILE,
        protocol=4,
    )

    # -------------------------------------------------------------
    # Verify output
    # -------------------------------------------------------------

    print("\nVerifying saved file...")

    verification = pd.read_pickle(
        OUTPUT_FILE
    )

    print(
        f"Verified shape: "
        f"{verification.shape}"
    )

    print(
        f"File size: "
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print("\nDataset preparation completed successfully.")

    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    prepare_dataset()