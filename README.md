# Fraud Detection & Investigation System

Fraud detection starts with a deceptively simple question: **Is this transaction fraudulent?**

A useful investigation system has to answer a harder sequence of questions: why did the model flag it, what changed in the customer's behavior, what does the terminal history show, which knowledge applies, what evidence was retrieved, and how can the evidence be organized into a traceable investigation report?

This project builds that progression from a transaction-level classifier into a local **Fraud Detection & Investigation System**.

No paid API, cloud deployment, or external inference endpoint is required.

```text
Transaction Data
      |
      v
Temporal Feature Engineering
      |
      v
LightGBM Fraud Classifier
      |
      v
Risk Probability
      |
      v
SHAP Evidence
      |
      +-----------------------------+
      |                             |
      v                             v
Investigation Knowledge Base       Behavioral Context
      |                             |
      v                             |
Sentence Embeddings                |
      |                             |
      v                             |
FAISS Retrieval <------------------+
      |
      v
Local Instruction LLM
      |
      v
Agent Orchestration
      |
      v
Investigation Report
      |
      v
Streamlit
```

## 1. The problem

A conventional fraud classifier ends with a probability:

<div align="center">

### **P(Y = 1 | X = x)**

</div>

That probability is useful, but it does not explain what an investigator should examine next. This project therefore treats classification as the beginning of the investigation rather than the end.

## 2. Data

The project is designed around the simulated transaction data from the Fraud Detection Handbook. The expected schema is:

| Column | Meaning |
|---|---|
| `TRANSACTION_ID` | Transaction identifier |
| `TX_DATETIME` | Transaction timestamp |
| `CUSTOMER_ID` | Customer identifier |
| `TERMINAL_ID` | Terminal identifier |
| `TX_AMOUNT` | Transaction amount |
| `TX_FRAUD` | Fraud label |

The raw dataset is intentionally not included in the repository. See `DATA_LICENSE_NOTE.txt` for the upstream source and licensing note.

### Dataset Statistics

- **Total Transactions:** 1,754,155
- **Legitimate Transactions:** 1,739,474
- **Fraudulent Transactions:** 14,681
- **Fraud Rate:** 0.8369%
- **Number of Features:** 9
- **Time Period:** April 1, 2018 – September 30, 2018
- **Dataset Size:** 88.54 MB
- **Source:** Fraud Detection Handbook — Simulated Transaction Dataset
  
## 3. Why time matters

Fraud detection is time-dependent. Randomly mixing future and past transactions can make evaluation unrealistically easy.

The project uses:

```text
Earliest 70% -> Training
Next 15%     -> Validation
Latest 15%   -> Final Test
```

Therefore:

$t_{train} < t_{validation} < t_{test}$

The final test set remains untouched until model and threshold selection are finished.

## 4. From transactions to behavior

For a time $t$ and historical window $W$, transaction velocity is:

$$
N_t(W) = \sum_j I(t-W \leq t_j < t)
$$

The strict condition $t_j < t$ excludes the current transaction. This makes the feature compatible with online scoring.

The project builds customer and terminal history such as previous transaction count, previous mean amount, previous fraud rate, recency, and 60-second, one-hour, and one-day velocity.

Amount deviation is represented as:

$$
R_t = \frac{x_t}{\mu_{\mathrm{customer}} + \epsilon}
$$

Raw customer and terminal IDs are not direct model features. Historical behavior is used instead, reducing direct identifier memorization.


## 5. The fraud classifier

The core classifier is LightGBM gradient-boosted decision trees.

```text
Behavioral Features
        |
        v
Decision Trees
        |
        v
Additive Score F(x)
        |
        v
Sigmoid
        |
        v
Fraud Probability
```

$$
p(x) = \frac{1}{1 + e^{-F(x)}}
$$

## 6. Class imbalance

Fraud is the minority class. The project uses:
**scale_pos_weight** = $N_{negative} / N_{positive}$

This changes training emphasis without generating synthetic transactions.

## 7. Probability becomes a decision

The classifier produces $p = P(Y=1 \mid X=x)$. An operational threshold converts it into a decision:

$$
\mathrm{Decision}(x) =
\begin{cases}
\mathrm{REVIEW}, & p(x) \geq t \\
\mathrm{ALLOW}, & p(x) < t
\end{cases}
$$

The threshold is selected on validation data only.

## 8. Cost-sensitive thresholding

The project demonstrates an illustrative cost function:

$$
\mathrm{Cost}(t) =
C_{\mathrm{FN}}FN(t)
+
C_{\mathrm{FP}}FP(t)
+
C_{\mathrm{Review}}Review(t)
$$

A maximum review-rate constraint prevents the optimizer from simply flagging almost everything. The coefficients are configuration examples, not universal banking costs.

## 9. Evaluation

The fraud model reports ROC-AUC, PR-AUC, precision, recall, F1, MCC, confusion matrix, review rate, and log loss.

PR-AUC is particularly useful when the positive class is rare because it directly reflects the precision-recall behavior of the fraud class.

The project keeps predictive evaluation separate from the investigation-layer evaluation.

## 10. SHAP: moving from score to evidence

A fraud probability still leaves the question: **why?**

For a tree model, SHAP provides an additive explanation:

$$
f(x) = E[f(X)] + \sum_i \phi_i
$$

A positive $\phi_i$ pushes the model toward higher risk; a negative value pushes it toward lower risk.

SHAP is treated as model evidence, not as independent proof of fraud.

## 11. RAG

Investigation requires more than transaction features. The system therefore maintains a project-specific knowledge base under `knowledge/`.

The retrieval flow is:

```text
Question
   |
   v
Embedding Model
   |
   v
Query Vector
   |
   v
FAISS
   |
   +--> Knowledge Chunk 1
   +--> Knowledge Chunk 2
   +--> Knowledge Chunk 3
   |
   v
Evidence Context
```

The query and each document chunk become vectors:

$$
e_q = Embedding(q)
$$

$$
e_i = Embedding(d_i)
$$

With normalized embeddings:

$$
Similarity(q,d_i) = e_q^T e_i
$$

The top-k chunks are passed to the language model.

The project uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and FAISS for local vector search.

## 12. RAG evaluation

Retrieval is evaluated independently of the language model.

Recall@k is:

$$
Recall@k =
\frac{queries\ with\ a\ relevant\ result\ in\ top\text{-}k}
{total\ queries}
$$

Mean Reciprocal Rank is:

$$
MRR =
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{rank_i}
$$

This separation matters: if the evidence is not retrieved, an LLM cannot produce a grounded answer from it.

## 13. Local LLM investigation assistant

The narrative layer uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

through Transformers.

The LLM is deliberately not the fraud classifier. Its role is to organize supplied evidence into an investigator-readable report.

The model receives:

```text
Transaction facts
+
Customer evidence
+
Terminal evidence
+
Model probability
+
SHAP evidence
+
Retrieved knowledge
```

The system instructs the model not to invent transaction facts or treat a model probability as proof of fraud.

## 14. Why agents?

A single LLM call could produce a report, but that would hide the investigation process.

The project therefore uses explicit specialist agents:

```text
                    Supervisor
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    Transaction       Customer       Terminal
       Agent            Agent          Agent
          |              |              |
          +--------------+--------------+
                         |
                         v
                  Evidence Agent
                    /         \\
                  SHAP        RAG
                    \\         /
                     +---+----+
                         |
                         v
                   Report Agent
                         |
                         v
                        LLM
```

| Component | Responsibility |
|---|---|
| **Transaction Agent** | Establishes transaction facts, probability, threshold, decision, and novelty indicators. |
| **Customer Agent** | Examines previous customer transactions, historical amount, fraud rate, recency, and velocity. |
| **Terminal Agent** | Examines terminal history using the same backward-looking behavioral context. |
| **Evidence Agent** | Combines model probability, SHAP evidence, customer context, terminal context, and RAG results. |
| **Report Agent** | Uses the local instruction model to turn the evidence package into a structured narrative. |
| **Supervisor** | Runs the graph in a fixed dependency order: Transaction → (Customer + Terminal) → Evidence → Report. |

This is deliberately inspectable rather than an uncontrolled agent loop.

## 15. End-to-end autonomous investigation

After the predictive model is frozen, the system selects high-risk cases from the final chronological period and runs the complete workflow:

```text
Transaction
    |
    v
Behavioral Context
    |
    v
Fraud Model
    |
    +--> Probability
    +--> Threshold Decision
    +--> SHAP Evidence
    |
    v
Customer + Terminal Agents
    |
    v
Evidence Agent
    |
    +--> RAG Retrieval
    |
    v
Report Agent
    |
    v
Local LLM
    |
    v
Investigation Report
```

The resulting cases are saved under `artifacts/investigation_cases.json`.

## 16. Three evaluation layers

The completed system evaluates three different things rather than forcing them into one score.

### Predictive layer

$$
ROC\text{-}AUC,\quad PR\text{-}AUC,\quad Precision,\quad Recall,\quad F1,\quad MCC
$$

### Retrieval layer

$$
Recall@k,\quad MRR
$$

### Agent layer

```text
Case completion rate
Evidence presence rate
Retrieved source rate
LLM report presence rate
```

These engineering metrics do not claim real-world fraud investigation accuracy.

## 17. What the LLM does and does not do

| LLM Does | LLM Does Not |
|---|---|
| Summarize supplied evidence | Train the fraud classifier |
| Explain retrieved project knowledge | Determine the fraud probability |
| Organize findings | Override the model threshold |
| Identify evidence limitations | Access external banking systems |
| Suggest next review actions | Establish fraud as a fact |

This separation keeps quantitative prediction and narrative investigation distinct.

## 18. Limitations

The dataset is synthetic, so the reported results may not directly reflect real-world production performance. The local language model functions as a narrative and investigation assistant rather than a fraud classifier, while RAG effectiveness depends on the coverage and quality of the available knowledge base. The illustrative cost coefficients would need to be replaced with organization-specific estimates before operational deployment. The system also does not incorporate external intelligence such as identity, merchant, device, geolocation, sanctions, banking, or network data.

This project integrates;
```text
Machine Learning
+
Explainable AI
+
Retrieval
+
LLMs
+
Agent orchestration
```

can be combined into a single fraud-investigation workflow.

## Note:
>Results are yet to be updated. If you want to run the experiments, please follow the instructions provided in commands.txt before running the project.

## Results & Evaluation

The complete pipeline was evaluated on a temporally separated test set. The model was trained on earlier transactions, while validation and test data were kept chronologically later to reduce temporal leakage and better reflect a real fraud-detection setting.

### Dataset Split

| Split      | Transactions | Fraud Cases | Fraud Rate |
| ---------- | -----------: | ----------: | ---------: |
| Train      |    1,227,908 |       9,996 |    0.8141% |
| Validation |      263,123 |       2,355 |    0.8950% |
| Test       |      263,124 |       2,330 |    0.8855% |

The test set was kept untouched during model training and threshold selection.

### Fraud Detection Performance

The LightGBM classifier was evaluated on the untouched test set using the threshold selected exclusively from validation data.

| Metric      | Test Result |
| ----------- | ----------: |
| ROC-AUC     |  **0.9437** |
| PR-AUC      |  **0.3556** |
| Precision   |  **0.7205** |
| Recall      |  **0.2910** |
| F1 Score    |  **0.4146** |
| MCC         |  **0.4551** |
| Review Rate | **0.3576%** |
| Log Loss    |  **0.0346** |

The selected classification threshold was **0.235**. At this operating point, only approximately **0.36% of transactions** were sent for review.

### Confusion Matrix

On the 263,124 transaction test set:

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------: | --------------: |
| **Actual Legitimate** |              260,531 |             263 |
| **Actual Fraud**      |                1,652 |             678 |

The model therefore identified **678 of the 2,330 fraudulent transactions** at the selected operating threshold while generating **263 false positives**.

The distinction between ROC-AUC and threshold-dependent metrics is important here. A ROC-AUC of 0.9437 indicates strong ranking ability across possible thresholds, while the selected threshold reflects a specific operational trade-off between fraud detection, false positives, and investigation workload.

### Threshold Optimization

Instead of using the default probability threshold of 0.5, the system searches for an operating threshold using the validation set.

The objective incorporates:

* False-negative cost
* False-positive cost
* Manual review cost
* Maximum allowable review rate

The selected threshold is then frozen before evaluating the test set.

```text
Training Data
      |
      v
Chronological Validation Set
      |
      v
Threshold / Cost Optimization
      |
      v
Selected Threshold = 0.235
      |
      v
Untouched Test Set
      |
      v
Final Evaluation
```

This prevents the test set from influencing the operating threshold.

## Retrieval-Augmented Generation

The investigation system also contains a fully local Retrieval-Augmented Generation (RAG) component.

The knowledge base currently contains **11 project-authored knowledge chunks** covering topics such as:

* Fraud detection fundamentals
* Behavioral features
* Temporal evaluation
* Class imbalance
* Threshold and cost optimization
* SHAP explainability
* Investigation methodology
* RAG methodology
* Agent responsibilities
* Investigation report structure
* System limitations

The system uses:

```text
Embedding Model:
sentence-transformers/all-MiniLM-L6-v2

Embedding Dimension:
384

Vector Store:
FAISS IndexFlatIP
```

### RAG Retrieval Evaluation

The retrieval component was evaluated using representative questions covering the system's major concepts.

| Metric              |     Result |
| ------------------- | ---------: |
| Recall@5            | **1.0000** |
| MRR                 | **1.0000** |
| Knowledge Chunks    |     **11** |
| Embedding Dimension |    **384** |

A Recall@5 of 1.0 means that the relevant knowledge source was retrieved within the top five results for all evaluation queries.

An MRR of 1.0 indicates that the relevant source appeared at rank 1 for the evaluated queries.

Example retrievals included:

```text
"How are historical velocity features calculated?"
    → behavioral_features.md

"Why is chronological splitting used?"
    → temporal_evaluation.md

"How does the system handle class imbalance?"
    → class_imbalance.md

"How is the decision threshold selected?"
    → threshold_and_cost.md

"How are SHAP contributions interpreted?"
    → shap_explainability.md

"What does the customer agent do?"
    → agent_roles.md
```
> **Note:** The RAG knowledge `.md` files are excluded from GitHub via `.gitignore`. Run `python src/create_knowledge_base.py` to generate them locally before running the RAG pipeline.

## End-to-End Pipeline

The current pipeline successfully executes the following stages:

```text
Raw Fraud Detection Handbook Dataset
                |
                v
        Data Preparation
                |
                v
      Chronological Splitting
                |
                v
    Leakage-Safe Feature Engineering
                |
                v
        LightGBM Classifier
                |
                v
      Validation Threshold Search
                |
                v
       Untouched Test Evaluation
                |
                v
        SHAP Explainability
                |
                v
        Local RAG Knowledge Base
                |
                v
        Retrieval Evaluation
                |
                v
       Local LLM Investigation
                |
                v
        Agent Investigation
                |
                v
          Streamlit Interface
```

The ML and RAG stages currently execute successfully from the command-line pipeline. The investigation agents and local LLM extend the system from simply detecting suspicious transactions toward producing evidence-grounded investigation reports.

## Experiment & Results Checklist

### Completed

* [x] Dataset preparation
* [x] Temporal train/validation/test split
* [x] Leakage-safe feature engineering
* [x] LightGBM classification
* [x] Threshold optimization
* [x] Test-set evaluation
* [x] SHAP explainability
* [x] RAG knowledge base
* [x] FAISS retrieval
* [x] RAG evaluation — Recall@5: **1.0000**, MRR: **1.0000**

### Yet to Do

* [ ] Local LLM
* [ ] Investigation Agents
* [ ] Agent Orchestration
* [ ] End-to-End Investigation
* [ ] Final System Evaluation


> **Note:** These results are based on the Fraud Detection Handbook's simulated transaction dataset. The dataset is synthetic, and therefore these metrics should not be interpreted as production fraud-detection performance. The system is intended as an end-to-end engineering and research project demonstrating temporal evaluation, leakage-safe feature engineering, imbalanced classification, explainability, retrieval-augmented investigation, and agent-based orchestration.
