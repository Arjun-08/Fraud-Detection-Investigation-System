# Implementation Notes

This is the engineering story behind the implementation rather than a README.

## Data

The project uses the Fraud Detection Handbook simulated transaction schema:

- `TRANSACTION_ID`
- `TX_DATETIME`
- `CUSTOMER_ID`
- `TERMINAL_ID`
- `TX_AMOUNT`
- `TX_FRAUD`

The simulator is synthetic and intentionally captures difficult fraud-detection properties such as severe class imbalance, high-cardinality entities, and time-dependent fraud scenarios.

The raw dataset stays outside Git.

## Temporal evaluation

Fraud detection is a forecasting problem. The stream is split:

    earliest 70% -> train
    next 15%     -> validation
    latest 15%   -> test

The test set is never used to select the model or threshold.

## Leakage-safe features

For time t and window W:

    N_t(W) = sum_j I(t-W <= t_j < t)

The current transaction is excluded.

Previous fraud rate:

    F_t = previous_fraud_count / previous_transaction_count

This is designed to be computable at scoring time.

## Model

LightGBM gradient-boosted trees are used:

    features -> decision trees -> additive score F(x) -> sigmoid -> probability

    p(fraud|x) = 1 / (1 + exp(-F(x)))

Class imbalance is handled with `scale_pos_weight`, not SMOTE.

## Threshold

The model's probability is converted into an operational decision:

    REVIEW if p >= threshold
    ALLOW  if p < threshold

Illustrative validation cost:

    Cost(t) = C_FN*FN(t) + C_FP*FP(t) + C_REVIEW*Review(t)

These coefficients are configurable and are not presented as real financial assumptions.

## Explainability

For tree models:

    f(x) = E[f(X)] + sum_i phi_i

SHAP values provide local feature contributions.

## Investigation layer

The application intentionally uses deterministic summaries instead of an external LLM. That means no API key, no third-party transaction-data transfer, and no dependency on a hosted model.

The investigation screen combines probability, threshold, behavioral features, and SHAP contributions.

## Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Place the existing dataset at:

    data/transactions.pkl

Then:

    python train.py
    streamlit run app.py
