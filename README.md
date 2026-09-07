# Fraud Detection & Investigation System

An end-to-end fraud detection system combining machine learning, behavioral feature engineering, explainable AI, and later-stage generative AI investigation.

## Project Overview

The project develops a fraud detection pipeline capable of identifying suspicious financial transactions and supporting downstream investigation.

The system is being developed incrementally, beginning with rigorous tabular machine learning and progressing toward explainability, retrieval-augmented generation, and agentic investigation.

## Current Stage

Phase 1 — Data Loading and Quality Analysis

The current implementation focuses on:

- Dataset acquisition and validation
- Loading the complete transaction dataset
- Chronological organization of transactions
- Fraud class analysis
- Data quality analysis
- Customer and terminal analysis
- Initial leakage identification

## Dataset

The project uses the open simulated transaction dataset provided by the Fraud Detection Handbook.

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

## Planned ML Pipeline

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