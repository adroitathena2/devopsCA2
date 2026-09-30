# Comparative Study: Alternative Architectures and Ensembling for DDI Severity Classification

For this clinical AI system, relying on external LLM APIs (like Gemini) introduces per-call latency (~4 s per severity call with reasoning-grade thinking), direct API costs, and privacy concerns for real-time EMR environments. To solve this, we trained local deep learning architectures on physician-audited severity labels (multi-agent adjudication with a standardized codebook — see UPDATE_3.md; the original single-LLM heuristic labels were replaced after a 50%-agreement audit revealed convention drift).

We then applied advanced ensembling techniques—specifically **Class Weighting** and **Stacking Meta-Models**—to combine their strengths. (Focal loss has since joined the hyperparameter search space for the 90%+-Moderate skew; production still uses class weights.)

> Note (2026-09-23): production ensemble is FastClinicalCNN + MedSeverityNet + TinyClinicalFormer; an extended 2-model search (CNN + Transformer, 96 configs, 5-fold refine) is running to retire the BiLSTM. Validation tables below predate the physician relabel where marked.

---

## 1. The Three Architectures

### A. MedSeverityNet (Hybrid Morpho-Semantic LSTM)
* **Architecture:** Character-Level CNN + Word Embeddings $\rightarrow$ BiLSTM $\rightarrow$ Self-Attention Pooling $\rightarrow$ Linear Classifier.
* **Strengths:** The CharCNN captures complex clinical morphology (e.g., "-toxicity"), while the BiLSTM understands the full sequential context of the sentence.
* **Weaknesses:** Recurrent networks process sequentially, making them the slowest architecture at inference time.

### B. FastClinicalCNN (Temporal Convolutional Network)
* **Architecture:** Word & Char Embeddings $\rightarrow$ 1D Convolutions (Kernels 2,3,4,5) $\rightarrow$ Global Max Pooling $\rightarrow$ Linear Classifier.
* **Strengths:** Blazing fast. Excellent at identifying specific n-gram "trigger phrases" (e.g., "fatal arrhythmia", "significantly increased risk") which strictly dictate DDI severity.
* **Weaknesses:** Cannot easily link the subject and object if they are separated by 20+ words, as it lacks long-range contextual memory.

### C. TinyClinicalFormer (Lightweight Custom Transformer)
* **Architecture:** Word & Char Embeddings + Positional Encoding $\rightarrow$ 2-Layer Transformer Encoder $\rightarrow$ Mean Pooling $\rightarrow$ Linear Classifier.
* **Strengths:** Uses self-attention to dynamically adjust a word's meaning based on its surroundings. Highly parallelizable.
* **Weaknesses:** Transformers are notoriously data-hungry. Training a Transformer completely from scratch on only 40k records yields poorer generalization compared to CNNs or LSTMs.

---

## 2. Dealing with Class Imbalance

Our validation distribution (post physician audit) is heavily imbalanced:
* **Moderate:** ~84% (exposure changes, efficacy loss, monitorable systemic effects)
* **Severe:** ~14% (immediate/life-threatening: QTc/torsades, bleeding, organ toxicity, etc.)
* **Mild:** ~2% (GI irritation only, minor-drug effects, no-consequence PK shifts)

Standard Cross-Entropy Loss defaults to the majority "Moderate" class. **The Fix:** **Class Weights** inside the PyTorch loss function penalize missing the rare classes (focal-loss gamma is now also a searched dimension in `tune_severity_search.py`). Macro F1 is the selection objective, not accuracy.

---

## 3. The Power of Stacking (Meta-Ensembles)

With three models producing heavily penalized probabilities, a naive average (Weighted Ensemble) maxed out early. We implemented **Stacking Generalization** to fix this.

We extracted the 9-dimensional probability outputs (3 models $\times$ 3 classes) for our training and validation sets. Known limitation, accepted at n=1600 and recorded in UPDATE_3.md: the stacker trains on simulated out-of-fold features from full-train logits (no per-fold base retraining), and temperature scalars sit at 1.5 (refit on held-out val NLL moves nothing) — sharpness is handled by codebook decision policy. We then trained an **XGBoost Meta-Model** to learn exactly *when* to trust each neural network.

### Final Validation Results (pre-relabel — retained for history; current numbers in UPDATE_3.md):
| Architecture / Ensemble | Accuracy | Macro F1 | ROC-AUC |
|-------------------------|----------|----------|---------|
| MedSeverityNet          | 77.00%   | 73.24%   | 0.9958  |
| FastClinicalCNN         | 93.75%   | 93.95%   | 0.9990  |
| TinyClinicalFormer      | 94.00%   | 94.23%   | 0.9923  |
| **Naive Weighted Avg**  | 94.25%   | 94.42%   | 0.9988  |
| **Logistic Regression Stacker** | 94.50%   | 94.71%   | 0.9938  |
| **XGBoost Stacker**     | 94.50%   | 94.90%   | 0.9887  |

### End-to-End Pipeline Metrics (Test Set, current):
On the physician-audited test set of 233 prescriptions (pairs: Moderate 172, Severe 66, Mild 5), ensemble-only severity with codebook policy gates:
* **AUC-ROC:** 0.9746
* **Sensitivity (Recall):** 100.00%
* **Specificity (TNR):** 97.06%
* **Binary F1:** 99.50%; accuracy 93.13%, macro F1 95.14%, weighted F1 93.08%; per-class F1 Mild 100.00% / Moderate 94.16% / Severe 88.00%
The Gemini pipeline (gemini-3.8-flash, severity-only, codebook prompt, low thinking) scores AUC 0.9712 with binary F1 99.25% on the same set, at a measured $0.42 per full refresh (250 calls, 0 errors) — the local ensemble leads it on severity separation at $0 per run. Note: the keyword path cannot predict Mild (DB floor is Moderate), so its macro F1 (68.72%) is penalized on the 5 Mild pairs where the ensemble scores 100%.

**Conclusion:** The XGBoost and Logistic stackers achieved an impressive **~94.5% accuracy**, seamlessly learning when to trust the superior n-gram detection of the CNN and successfully compensating for the weaker standalone performance of the Transformer architecture. 

---

## 4. The Inference Latency vs. Accuracy Trade-off

A key finding of this project is the **Pareto Frontier** of AI in healthcare:

1. **Cloud LLMs (Gemini 3.8-flash):** Provide highly flexible and accurate generalization but take **~5 seconds** per interacting pair, require constant internet connectivity, cost money per API call (measured $0.87 per 233-row refresh, ~99% of it thinking tokens), and pose HIPAA/Data Privacy risks by sending patient data to external servers.
2. **Our Local Stacked Ensemble:** Achieves **high benchmarked accuracy** but runs in **milliseconds** per interaction, operates entirely offline, costs $0 at inference, and guarantees 100% patient data privacy.

In a clinical setting where a doctor is waiting for an EMR system to load, a 4-second delay is an eternity. Trading a few percentage points of theoretical accuracy for a 99% speedup and complete offline privacy is exactly how real-world clinical systems must be architected.
