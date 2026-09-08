from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "transactions_combined.pkl"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# UTILITY
# =============================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print_section(
    "PHASE 2 — TEMPORAL DATA SPLIT"
)

print("\nLoading dataset...")

df = pd.read_pickle(DATA_FILE)

print(
    f"Total transactions: "
    f"{len(df):,}"
)


# =============================================================================
# STEP 2 — SORT CHRONOLOGICALLY
# =============================================================================

print_section(
    "STEP 2 — CHRONOLOGICAL SORTING"
)

print(
    "\nSorting transactions by TX_DATETIME..."
)

df = (
    df.sort_values(
        ["TX_DATETIME", "TRANSACTION_ID"]
    )
    .reset_index(drop=True)
)

print(
    f"Chronologically sorted: "
    f"{df['TX_DATETIME'].is_monotonic_increasing}"
)


# =============================================================================
# STEP 3 — CALCULATE SPLIT POINTS
# =============================================================================

print_section(
    "STEP 3 — CALCULATING SPLIT POINTS"
)

n = len(df)

train_end = int(
    n * 0.70
)

validation_end = int(
    n * 0.85
)

print(
    f"\nTotal rows       : {n:,}"
)

print(
    f"Train end index  : {train_end:,}"
)

print(
    f"Validation end   : {validation_end:,}"
)


# =============================================================================
# STEP 4 — CREATE SPLITS
# =============================================================================

train_df = df.iloc[
    :train_end
].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()


# =============================================================================
# STEP 5 — SPLIT SUMMARY
# =============================================================================

print_section(
    "STEP 5 — SPLIT SUMMARY"
)

splits = {
    "TRAIN": train_df,
    "VALIDATION": validation_df,
    "TEST": test_df
}

for name, split in splits.items():

    fraud_count = int(
        split["TX_FRAUD"].sum()
    )

    fraud_rate = (
        fraud_count
        / len(split)
        * 100
    )

    print(
        f"""
{name}
--------------------
Rows       : {len(split):,}
Fraud      : {fraud_count:,}
Fraud rate : {fraud_rate:.4f}%
Start      : {split["TX_DATETIME"].min()}
End        : {split["TX_DATETIME"].max()}
"""
    )


# =============================================================================
# STEP 6 — TEMPORAL VALIDATION
# =============================================================================

print_section(
    "STEP 6 — TEMPORAL ORDER VALIDATION"
)

train_end_time = (
    train_df["TX_DATETIME"].max()
)

validation_start_time = (
    validation_df["TX_DATETIME"].min()
)

validation_end_time = (
    validation_df["TX_DATETIME"].max()
)

test_start_time = (
    test_df["TX_DATETIME"].min()
)

test_end_time = (
    test_df["TX_DATETIME"].max()
)


print(
    f"\nTrain ends        : "
    f"{train_end_time}"
)

print(
    f"Validation starts : "
    f"{validation_start_time}"
)

print(
    f"Validation ends   : "
    f"{validation_end_time}"
)

print(
    f"Test starts       : "
    f"{test_start_time}"
)

print(
    f"Test ends         : "
    f"{test_end_time}"
)


# IMPORTANT:
# Equal timestamps at boundaries are allowed.
#
# Example:
#
# Validation ends : 13:09:12
# Test starts     : 13:09:12
#
# This can happen because multiple transactions occur
# at exactly the same timestamp.


temporal_order_valid = (
    train_end_time
    <= validation_start_time
    <= validation_end_time
    <= test_end_time
)

print(
    f"\nTemporal ordering valid: "
    f"{temporal_order_valid}"
)

assert temporal_order_valid, (
    "Temporal ordering is invalid."
)


# =============================================================================
# STEP 7 — CHECK TRANSACTION ID OVERLAP
# =============================================================================

print_section(
    "STEP 7 — TRANSACTION ID OVERLAP CHECK"
)

train_ids = set(
    train_df["TRANSACTION_ID"]
)

validation_ids = set(
    validation_df["TRANSACTION_ID"]
)

test_ids = set(
    test_df["TRANSACTION_ID"]
)


train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)


print(
    f"\nTrain ∩ Validation : "
    f"{len(train_validation_overlap):,}"
)

print(
    f"Train ∩ Test       : "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation ∩ Test  : "
    f"{len(validation_test_overlap):,}"
)


assert len(train_validation_overlap) == 0
assert len(train_test_overlap) == 0
assert len(validation_test_overlap) == 0


print(
    "\nNo transaction ID overlap detected."
)


# =============================================================================
# STEP 8 — CHECK ROW COUNTS
# =============================================================================

print_section(
    "STEP 8 — ROW COUNT VALIDATION"
)

total_split_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print(
    f"\nOriginal rows : "
    f"{len(df):,}"
)

print(
    f"Split rows    : "
    f"{total_split_rows:,}"
)

print(
    f"Difference    : "
    f"{len(df) - total_split_rows:,}"
)

assert (
    total_split_rows
    == len(df)
)

print(
    "\nAll transactions accounted for."
)


# =============================================================================
# STEP 9 — CLASS DISTRIBUTION
# =============================================================================

print_section(
    "STEP 9 — CLASS DISTRIBUTION ACROSS SPLITS"
)

for name, split in splits.items():

    fraud = int(
        split["TX_FRAUD"].sum()
    )

    legitimate = (
        len(split) - fraud
    )

    print(
        f"""
{name}
    Legitimate : {legitimate:,}
    Fraud      : {fraud:,}
    Fraud rate : {fraud / len(split) * 100:.4f}%
"""
    )


# =============================================================================
# STEP 10 — SAVE SPLITS
# =============================================================================

print_section(
    "STEP 10 — SAVING SPLITS"
)

train_file = (
    SPLIT_DIR
    / "train.pkl"
)

validation_file = (
    SPLIT_DIR
    / "validation.pkl"
)

test_file = (
    SPLIT_DIR
    / "test.pkl"
)


train_df.to_pickle(
    train_file
)

validation_df.to_pickle(
    validation_file
)

test_df.to_pickle(
    test_file
)


print(
    f"\nTrain      : {train_file}"
)

print(
    f"Validation : {validation_file}"
)

print(
    f"Test       : {test_file}"
)


# =============================================================================
# FINAL
# =============================================================================

print_section(
    "TEMPORAL SPLIT COMPLETE"
)

print(
    """
The temporal split passed all validation checks.

Important:
-----------
The test set will remain untouched during:

- Feature engineering decisions
- Model training
- Hyperparameter tuning
- Threshold optimization
- Calibration
- Model selection

"""
)