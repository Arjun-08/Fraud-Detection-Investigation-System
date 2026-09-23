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

\[
P(Y=1\mid X=x)
\]

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

## 3. Why time matters

Fraud detection is time-dependent. Randomly mixing future and past transactions can make evaluation unrealistically easy.

The project uses:

```text
Earliest 70% -> Training
Next 15%     -> Validation
Latest 15%   -> Final Test
```

Therefore:

\[
t_{train}<t_{validation}<t_{test}
\]

The final test set remains untouched until model and threshold selection are finished.

## 4. From transactions to behavior

For a time \(t\) and historical window \(W\), transaction velocity is:

\[
N_t(W)=\sum_j I(t-W\leq t_j<t)
\]

The strict \(t_j<t\) condition excludes the current transaction. This makes the feature compatible with online scoring.

The project builds customer and terminal history such as previous transaction count, previous mean amount, previous fraud rate, recency, and 60-second, one-hour, and one-day velocity.

Amount deviation is represented as:

\[
R_t=\frac{x_t}{\mu_{customer}+\epsilon}
\]

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

\[
p(x)=\frac{1}{1+e^{-F(x)}}
\]

## 6. Class imbalance

Fraud is the minority class. The project uses:

\[
scale\_pos\_weight=\frac{N_{negative}}{N_{positive}}
\]

This changes training emphasis without generating synthetic transactions.

## 7. Probability becomes a decision

The classifier produces \(p=P(Y=1\mid X=x)\). An operational threshold converts it into a decision:

\[
Decision(x)=
\begin{cases}
REVIEW & p(x)\geq t\\
ALLOW & p(x)<t
\end{cases}
\]

The threshold is selected on validation data only.

## 8. Cost-sensitive thresholding

The project demonstrates an illustrative cost function:

\[
Cost(t)=C_{FN}FN(t)+C_{FP}FP(t)+C_{Review}Review(t)
\]

A maximum review-rate constraint prevents the optimizer from simply flagging almost everything. The coefficients are configuration examples, not universal banking costs.

## 9. Evaluation

The fraud model reports ROC-AUC, PR-AUC, precision, recall, F1, MCC, confusion matrix, review rate, and log loss.

PR-AUC is particularly useful when the positive class is rare because it directly reflects the precision-recall behavior of the fraud class.

The project keeps predictive evaluation separate from the investigation-layer evaluation.

## 10. SHAP: moving from score to evidence

A fraud probability still leaves the question: **why?**

For a tree model, SHAP provides an additive explanation:

\[
f(x)=E[f(X)]+\sum_i\phi_i
\]

A positive \(\phi_i\) pushes the model toward higher risk; a negative value pushes it toward lower risk.

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

\[
e_q=Embedding(q)
\]

\[
e_i=Embedding(d_i)
\]

With normalized embeddings:

\[
Similarity(q,d_i)=e_q^Te_i
\]

The top-k chunks are passed to the language model.

The project uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and FAISS for local vector search.

## 12. RAG evaluation

Retrieval is evaluated independently of the language model.

Recall@k is:

\[
Recall@k=\frac{\text{queries with a relevant result in top-k}}{\text{total queries}}
\]

Mean Reciprocal Rank is:

\[
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

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

### Transaction Agent

Establishes transaction facts, probability, threshold, decision, and novelty indicators.

### Customer Agent

Examines previous customer transactions, historical amount, fraud rate, recency, and velocity.

### Terminal Agent

Examines terminal history using the same backward-looking behavioral context.

### Evidence Agent

Combines model probability, SHAP evidence, customer context, terminal context, and RAG results.

### Report Agent

Uses the local instruction model to turn the evidence package into a structured narrative.

### Supervisor

Runs the graph in a fixed dependency order:

\[
Transaction\rightarrow(Customer+Terminal)\rightarrow Evidence\rightarrow Report
\]

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

\[
ROC\text{-}AUC,\quad PR\text{-}AUC,\quad Precision,\quad Recall,\quad F1,\quad MCC
\]

### Retrieval layer

\[
Recall@k,\quad MRR
\]

### Agent layer

```text
Case completion rate
Evidence presence rate
Retrieved source rate
LLM report presence rate
```

These engineering metrics do not claim real-world fraud investigation accuracy.

## 17. Streamlit

The application contains four areas:

1. **Dashboard** — fraud distribution, threshold, test metrics, and high-risk transactions.
2. **Transaction Investigation** — runs the complete agentic workflow for a selected transaction.
3. **RAG Assistant** — retrieves project knowledge and generates a grounded answer.
4. **System Evaluation** — displays ML, RAG, and agent metrics.

## 18. Project structure

```text
fraud_detection_investigation_system/
|
+-- app.py
+-- train.py
+-- run_full_system.py
+-- requirements.txt
+-- RUN_COMMANDS.txt
+-- DATA_LICENSE_NOTE.txt
+-- PROJECT_NOTES.md
+|
+|-- data/
+|   +-- transactions.pkl
+|
+|-- knowledge/
+|   +-- fraud investigation knowledge files
+|
+|-- artifacts/
+|   +-- trained model
+|   +-- threshold
+|   +-- metrics
+|   +-- FAISS index
+|   +-- investigation cases
+|
+`-- src/
+    +-- data_loader.py
+    +-- split.py
+    +-- features.py
+    +-- model.py
+    +-- evaluate.py
+    +-- explain.py
+    +-- rag.py
+    +-- llm.py
+    +-- agents.py
+    +-- investigation_pipeline.py
+    `-- evaluate_investigation.py
```

## 19. Running the project

Create an environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the existing transaction dataset at:

```text
data/transactions.pkl
```

Run the complete system:

```bash
python run_full_system.py
```

Then start Streamlit:

```bash
streamlit run app.py
```

The first run downloads and caches the embedding and local instruction models. No API key or cloud inference endpoint is required.

## 20. What the LLM does and does not do

The LLM does:

- summarize supplied evidence,
- explain retrieved project knowledge,
- organize findings,
- identify evidence limitations,
- suggest next review actions.

The LLM does not:

- train the fraud classifier,
- determine the fraud probability,
- override the model threshold,
- access external banking systems,
- establish fraud as a fact.

This separation keeps quantitative prediction and narrative investigation distinct.

## 21. Limitations

The dataset is synthetic, so synthetic performance is not equivalent to production performance.

The local language model is a narrative assistant rather than a fraud classifier.

RAG quality depends on knowledge-base coverage.

Illustrative cost coefficients must be replaced with organization-specific estimates before operational use.

The system has no external identity, merchant, device, geolocation, sanctions, banking, or network intelligence.

The system is therefore best understood as a complete research and portfolio implementation demonstrating how:

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

## 22. Final architecture

The project evolves through four layers:

```text
LAYER 1: Predictive ML
    Behavioral features -> LightGBM -> threshold -> metrics

LAYER 2: Explainable ML
    Model probability -> SHAP -> local evidence

LAYER 3: Knowledge + Generation
    Knowledge -> embeddings -> FAISS -> local LLM

LAYER 4: Agentic Investigation
    Supervisor -> specialist agents -> evidence -> report -> Streamlit
```

The original question was:

\[
\boxed{\text{Is this transaction fraudulent?}}
\]

The completed system moves toward:

\[
\boxed{\text{What evidence makes this transaction worth investigating?}}
\]

and:

\[
\boxed{\text{How can that evidence be retrieved, analyzed, and presented as a traceable report?}}
\]
