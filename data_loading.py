from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "simulated-data-raw"

OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_memory(bytes_value: float) -> str:

    if bytes_value < 1024:
        return f"{bytes_value:.2f} B"

    if bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"

    if bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.2f} MB"

    return f"{bytes_value / (1024 ** 3):.2f} GB"


# =============================================================================
# STEP 1 — ENVIRONMENT
# =============================================================================

print_section("FRAUD DETECTION PROJECT — DATA LOADING")

print(f"Python version : {sys.version.split()[0]}")
print(f"Pandas version : {pd.__version__}")
print(f"Project root   : {PROJECT_ROOT}")


# =============================================================================
# STEP 2 — FIND DATASET
# =============================================================================

print_section("STEP 2 — LOCATING DATASET")

print(f"\nSearching inside:")
print(DATA_ROOT)

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Data directory does not exist:\n{DATA_ROOT}"
    )


# Search recursively for pickle files.
transaction_files = sorted(
    DATA_ROOT.rglob("*.pkl")
)

if not transaction_files:

    print("\nNo .pkl files were found.")

    print("\nDirectories found:")

    for directory in DATA_ROOT.rglob("*"):
        if directory.is_dir():
            print(f"  - {directory}")

    raise FileNotFoundError(
        "\nNo transaction .pkl files were found under:\n"
        f"{DATA_ROOT}"
    )


print(f"\nTransaction files found : {len(transaction_files)}")

print("\nFirst 5 files:")

for file in transaction_files[:5]:
    print(f"  {file.relative_to(PROJECT_ROOT)}")


print("\nLast 5 files:")

for file in transaction_files[-5:]:
    print(f"  {file.relative_to(PROJECT_ROOT)}")


# =============================================================================
# STEP 3 — LOAD ALL TRANSACTIONS
# =============================================================================

print_section("STEP 3 — LOADING ALL TRANSACTIONS")

start_time = time.time()

dataframes = []

for i, file in enumerate(transaction_files, start=1):

    df = pd.read_pickle(file)

    dataframes.append(df)

    if (
        i == 1
        or i % 20 == 0
        or i == len(transaction_files)
    ):
        print(
            f"[{i:3d}/{len(transaction_files)}] "
            f"{file.name:<20} "
            f"rows={len(df):>8,}"
        )


print("\nCombining daily transaction files...")

train_df = pd.concat(
    dataframes,
    ignore_index=True
)

elapsed = time.time() - start_time

print("\nAll files loaded successfully.")

print(f"Total rows       : {len(train_df):,}")
print(f"Total columns    : {train_df.shape[1]}")
print(f"Loading time     : {elapsed:.2f} seconds")

memory_usage = train_df.memory_usage(
    deep=True
).sum()

print(
    f"Memory usage     : "
    f"{format_memory(memory_usage)}"
)


# =============================================================================
# STEP 4 — DATASET STRUCTURE
# =============================================================================

print_section("STEP 4 — DATASET STRUCTURE")

print(f"\nDataset shape : {train_df.shape}")

print("\nColumns:")

for i, column in enumerate(
    train_df.columns,
    start=1
):
    print(f"{i:2d}. {column}")


print("\nData types:")

print(train_df.dtypes)


# =============================================================================
# STEP 5 — CHRONOLOGICAL ORDER
# =============================================================================

print_section("STEP 5 — CHRONOLOGICAL ORDER CHECK")

print("\nSorting transactions by TX_DATETIME...")

train_df = train_df.sort_values(
    "TX_DATETIME"
).reset_index(drop=True)

is_sorted = train_df[
    "TX_DATETIME"
].is_monotonic_increasing

print(
    f"Chronologically sorted : "
    f"{is_sorted}"
)

print(
    f"\nFirst transaction : "
    f"{train_df['TX_DATETIME'].iloc[0]}"
)

print(
    f"Last transaction  : "
    f"{train_df['TX_DATETIME'].iloc[-1]}"
)


# =============================================================================
# STEP 6 — TARGET ANALYSIS
# =============================================================================

print_section("STEP 6 — FRAUD TARGET ANALYSIS")

target_counts = (
    train_df["TX_FRAUD"]
    .value_counts()
    .sort_index()
)

legitimate = int(
    target_counts.get(0, 0)
)

fraudulent = int(
    target_counts.get(1, 0)
)

total = len(train_df)

fraud_rate = (
    fraudulent / total * 100
)

print(
    f"\nTotal transactions      : "
    f"{total:,}"
)

print(
    f"Legitimate transactions : "
    f"{legitimate:,}"
)

print(
    f"Fraudulent transactions : "
    f"{fraudulent:,}"
)

print(
    f"Fraud rate              : "
    f"{fraud_rate:.4f}%"
)

if fraudulent > 0:

    imbalance_ratio = (
        legitimate / fraudulent
    )

    print(
        f"Class imbalance         : "
        f"{imbalance_ratio:.2f}:1"
    )


# =============================================================================
# STEP 7 — FRAUD SCENARIOS
# =============================================================================

print_section("STEP 7 — FRAUD SCENARIO ANALYSIS")

scenario_counts = (
    train_df[
        "TX_FRAUD_SCENARIO"
    ]
    .value_counts()
    .sort_index()
)

print("\nFraud scenario distribution:")

for scenario, count in scenario_counts.items():

    print(
        f"Scenario {scenario}: "
        f"{count:,}"
    )


# =============================================================================
# STEP 8 — DATA QUALITY
# =============================================================================

print_section("STEP 8 — DATA QUALITY CHECK")

print("\nMissing values:")

missing = train_df.isna().sum()

missing = missing[
    missing > 0
]

if missing.empty:

    print("No missing values found.")

else:

    print(
        missing
        .sort_values(ascending=False)
        .to_string()
    )


print("\nDuplicate rows:")

duplicates = train_df.duplicated().sum()

print(
    f"Duplicate rows : "
    f"{duplicates:,}"
)


print("\nDuplicate transaction IDs:")

duplicate_ids = (
    train_df[
        "TRANSACTION_ID"
    ]
    .duplicated()
    .sum()
)

print(
    f"Duplicate IDs  : "
    f"{duplicate_ids:,}"
)


# =============================================================================
# STEP 9 — TIME RANGE
# =============================================================================

print_section("STEP 9 — DATASET TIME RANGE")

min_date = train_df[
    "TX_DATETIME"
].min()

max_date = train_df[
    "TX_DATETIME"
].max()

duration = (
    max_date - min_date
)

print(
    f"\nStart date : {min_date}"
)

print(
    f"End date   : {max_date}"
)

print(
    f"Duration   : {duration}"
)


# =============================================================================
# STEP 10 — TRANSACTION STATISTICS
# =============================================================================

print_section("STEP 10 — TRANSACTION AMOUNT STATISTICS")

print(
    train_df[
        "TX_AMOUNT"
    ]
    .describe()
    .to_string()
)


# =============================================================================
# STEP 11 — DAILY FRAUD ANALYSIS
# =============================================================================

print_section("STEP 11 — DAILY FRAUD ANALYSIS")

daily_stats = (
    train_df
    .set_index("TX_DATETIME")
    .resample("D")
    .agg(
        transactions=(
            "TRANSACTION_ID",
            "count"
        ),

        frauds=(
            "TX_FRAUD",
            "sum"
        )
    )
)

daily_stats[
    "fraud_rate"
] = (
    daily_stats["frauds"]
    / daily_stats["transactions"]
    * 100
)


print("\nFirst 10 days:")

print(
    daily_stats
    .head(10)
    .to_string()
)


print("\nDaily statistics:")

print(
    daily_stats[
        [
            "transactions",
            "frauds",
            "fraud_rate"
        ]
    ]
    .describe()
    .to_string()
)


# =============================================================================
# STEP 12 — LEAKAGE WARNING
# =============================================================================

print_section("STEP 12 — INITIAL LEAKAGE CHECK")

print(
    "\nWARNING:"
)

print(
    "TX_FRAUD_SCENARIO will NOT be used "
    "as a model feature."
)

print(
    "\nReason:"
)

print(
    "It directly identifies the fraud scenario "
    "and would introduce target leakage."
)

print(
    "\nPrediction target:"
)

print(
    "TX_FRAUD"
)


# =============================================================================
# STEP 13 — SAVE COMBINED DATASET
# =============================================================================

print_section("STEP 13 — SAVING COMBINED DATASET")

output_file = (
    OUTPUT_DIR
    / "transactions_combined.pkl"
)

print(
    f"\nSaving dataset to:\n"
    f"{output_file}"
)

start_time = time.time()

train_df.to_pickle(
    output_file
)

elapsed = time.time() - start_time

print(
    f"\nSaved successfully."
)

print(
    f"File size : "
    f"{format_memory(output_file.stat().st_size)}"
)

print(
    f"Save time : "
    f"{elapsed:.2f} seconds"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section("DATA LOADING COMPLETE")

print(
    f"""
Dataset summary
----------------
Rows              : {len(train_df):,}
Columns           : {train_df.shape[1]}
Fraud cases       : {fraudulent:,}
Fraud rate        : {fraud_rate:.4f}%
Start date        : {min_date}
End date          : {max_date}
Memory usage      : {format_memory(memory_usage)}

Saved dataset:
{output_file}


"""
)