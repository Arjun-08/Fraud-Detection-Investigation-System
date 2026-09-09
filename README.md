# End-to-End Fraud Detection System

## Work in Progress

<p align="center">
  <img src="https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif" width="350">
</p>

<p align="center">
  <b>🚧 Work in Progress 🚧</b>
</p>

<p align="center">
  Building, breaking, debugging, rebuilding...<br>
  Repeat until it works.
</p>

---

## Building

An end-to-end fraud detection system combining machine learning, behavioral feature engineering, explainable AI, and later-stage generative AI investigation.

## Project Overview

This project develops a fraud detection pipeline capable of identifying suspicious financial transactions and supporting downstream investigation.

The system is being developed incrementally, beginning with rigorous tabular machine learning and progressing toward explainability, retrieval-augmented generation, and agentic investigation.

## Development Progress

| Phase | Status | Focus |
|---|---|---|
| Phase 1 | Completed | Data loading and quality analysis |
| Phase 2 | Completed | EDA, temporal analysis and feature engineering |
| Phase 3 | In progress | Machine learning model development and comparison |
| Phase 4 | Planned | Threshold optimization, calibration and cost-based decisions |
| Phase 5 | Planned | Explainable AI with SHAP |
| Phase 6 | Planned | RAG-based fraud investigation |
| Phase 7 | Planned | Agentic investigation and end-to-end integration |

---

# Phase 1 — Data Loading & Quality Analysis

The initial stage focused on understanding and validating the raw transaction dataset.

### Completed

- Dataset acquisition and validation
- Loading the complete transaction dataset
- Chronological organization of transactions
- Fraud class analysis
- Data quality analysis
- Customer and terminal analysis
- Initial leakage identification

---

# Dataset

The project uses the open simulated transaction dataset provided by the **Fraud Detection Handbook**.

The dataset contains transaction-level information including:

- Transaction ID
- Transaction timestamp
- Customer ID
- Terminal ID
- Transaction amount
- Time information
- Fraud label
- Fraud scenario

The dataset itself is not included in this repository.

---

# Phase 2 — EDA, Temporal Analysis & Feature Engineering

Phase 2 focuses on understanding the transaction data, analyzing fraud patterns over time, preventing data leakage, and creating behavior-based features for downstream machine learning models.

## Exploratory Data Analysis

The dataset was analyzed across several dimensions:

- Class distribution and fraud imbalance
- Transaction amount distributions
- Fraud rate by hour of day
- Fraud rate by day of week
- Daily fraud trends
- Fraud scenario distribution
- Customer transaction behavior
- Terminal transaction behavior

These analyses help identify temporal and behavioral patterns that can be incorporated into the fraud detection pipeline.

## Temporal Data Split

Since fraud detection is a time-dependent problem, the dataset is split chronologically rather than randomly.

| Split | Transactions | Fraud Cases | Fraud Rate |
|---|---:|---:|---:|
| Train | 1,227,908 | 9,996 | 0.8141% |
| Validation | 263,123 | 2,355 | 0.8950% |
| Test | 263,124 | 2,330 | 0.8855% |

The test set represents a future time period relative to the training data, providing a more realistic evaluation of model performance.

## Leakage Prevention

Several fields were excluded from the model features:

- `TX_FRAUD` — prediction target
- `TX_FRAUD_SCENARIO` — contains direct information about the fraud mechanism
- `TRANSACTION_ID` — identifier with no predictive meaning
- `TX_DATETIME` — replaced by derived temporal features
- `CUSTOMER_ID` and `TERMINAL_ID` — used to construct historical behavioral features rather than directly passed to the model

Historical behavioral features are calculated using **previous transactions only**, ensuring that future information does not influence the prediction of an earlier transaction.

---

## Feature Engineering

The feature engineering pipeline creates several groups of predictive features.

### Temporal Features

- Hour
- Day of week
- Day of month
- Month
- Weekend indicator
- Cyclic hour encoding using sine/cosine
- Cyclic day-of-week encoding using sine/cosine

### Transaction Features

- Log-transformed transaction amount

### Customer Behavioral Features

- Historical transaction count
- Previous transaction amount
- Time since previous transaction
- Historical mean transaction amount
- Historical transaction amount standard deviation

### Terminal Behavioral Features

- Historical transaction count
- Previous transaction amount
- Time since previous transaction
- Historical mean transaction amount

### Behavioral Deviation Features

- Transaction amount relative to historical customer average
- Transaction amount relative to historical terminal average

---

## Historical Feature Construction

A key design choice is that behavioral statistics are calculated chronologically using previous transactions only.

For example, the historical customer mean transaction amount is calculated as:

$$
\text{Customer Mean Amount}_t =
\frac{1}{N_{t-1}}\sum_{i<t} Amount_i
$$

Similarly, the time since a customer's previous transaction is calculated as:

$$
\Delta t_t = t_t - t_{t-1}
$$

This prevents the current transaction, future transactions, or future labels from influencing the feature values.

---

# Planned ML Pipeline

```text
Raw Transactions
       |
       v
Data Quality Analysis
       |
       v
Temporal Analysis
       |
       v
Behavioral Feature Engineering
       |
       v
Train / Validation / Test
       |
       +------------------+
       |                  |
       v                  v
Random Forest          XGBoost
       |                  |
       +--------+---------+
                |
                v
             CatBoost
                |
                v
        Model Comparison
                |
                v
       Threshold Optimization
                |
                v
          Calibration
                |
                v
       Cost-Based Decisions
                |
                v
          SHAP Analysis
                |
                v
      Fraud Investigation
                |
                v
          RAG / Agents
