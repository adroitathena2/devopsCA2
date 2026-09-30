# AI Clinical Copilot: End-to-End Pipeline Summary

This document outlines the architecture and data flow of the ML-enhanced clinical copilot backend, designed to securely and efficiently extract medications from prescriptions and detect Drug-Drug Interactions (DDIs).

## 1. Data Processing & Curation
Before inference, the system relies on a physician-audited dataset to train its local models (see UPDATE_3.md for the audit trail):
- **Multi-agent labeling:** 2,000 DrugBank interaction descriptions adjudicated by independent agents under a standardized severity codebook (Severe = immediate/life-threatening; Moderate = avoid unless necessary; Mild = no long-term effects), 96.3% raw agreement, 73 conflicts resolved by physician rulings. A separate 2000-description codebook sample backs the same criteria.
- **Taxonomy:** strict 3-class integer format:
  * `0`: **Mild** (avoid only if an alternative exists, no long-term effects — a predictable minority class)
  * `1`: **Moderate** (requires monitoring or dosage adjustment)
  * `2`: **Severe** (life-threatening, contraindicated)
  plus `No Interaction` as the prescription-level outcome when no pair is found.
- **Data Splitting:** frozen 1,600 training / 400 validation split (never re-cut).

## 2. Inference Pipeline: Step-by-Step

### Phase 1: Entity Extraction (NER)
When a prescription is submitted (either via raw text or OCR using the Gemini Vision API), the system extracts medical entities:
- **GLiNER Medical NER + Dictionary (FlashText):** Uses the `E3-JSI/gliner-multi-med-ner-synthetic-v1` span model to locate drugs, backed by a FlashText dictionary over ~48k generic/alias/brand keywords as a safety net against NER omission, plus mention grounding to reject hallucinated entities.

### Phase 2: Semantic Matching (Entity Linking)
Extracted entities often contain typos, brand names, or tokenization artifacts.
- **Granite Embeddings:** The system generates semantic embeddings for the extracted strings using `ibm-granite/granite-embedding-278m-multilingual` (selected over SapBERT, WeMM-2B, Qwen3, gemma, and PubMedBERT in an end-to-end comparison — see `EMBEDDING_MODEL_COMPARISON.md`).
- **Cosine Similarity Search:** The extracted embeddings are compared against pre-computed embeddings of a comprehensive DrugBank database. Matches exceeding a strict similarity threshold (e.g., `0.85`) are resolved to their canonical generic names.

### Phase 3: Interaction Detection
With a clean list of canonical generic drugs, the system checks for pairwise interactions.
- **Fast Local Lookup:** The system queries a pre-indexed JSON mapping of the DrugBank database to see if a known interaction exists between any two prescribed drugs.
- **Description Retrieval:** If an interaction exists, its clinical description is fetched for severity classification.

### Phase 4: Severity Classification (Ablation Paths)
To determine if an interaction is actionable, the description is passed to a severity classifier. The pipeline supports two parallel paths for ablation testing:

**Path A: Baseline Pipeline (Cloud LLM)**
- The description is sent to **gemini-3.8-flash** using a strict single-label prompt (severity) plus an optional layman rewrite (reasoning, toggleable), with the final label resolved safety-first as max(Database keywords, Gemini).
- *Pros:* High generalization.
- *Cons:* ~4–5 seconds per interacting pair, requires internet, per-token API cost, violates air-gapped EMR privacy constraints.

**Path B: Optimized Pipeline (Local Custom Ensembles)**
- The description is passed through locally-hosted neural networks (production: FastClinicalCNN + MedSeverityNet + TinyClinicalFormer; extended search running to keep CNN + Transformer only):
  1. **FastClinicalCNN:** A 1D Temporal Convolutional Network that excels at finding localized severe n-gram triggers (e.g., "fatal arrhythmia").
  2. **MedSeverityNet:** A BiLSTM + CharCNN architecture that captures deep sequential clinical morphology (in production, retirement in progress).
  3. **TinyClinicalFormer:** A lightweight custom transformer.
- **Stacking Meta-Model (XGBoost):** base-model probabilities feed an XGBoost classifier (accepted limitation: trained on simulated out-of-fold full-train features at n=1600 — see UPDATE_3.md; temperatures stay 1.5 evidence-backed).
- **Ensemble-only severity with codebook policy:** the stacker is the sole severity signal (DrugBank-keyword fusion removed); physician-confirmed gates keep genuinely-mild descriptions Mild and require a clear margin for Severe. `P(Moderate) + P(Severe)` remains the binary risk score, with Mild counted negative.
- *Pros:* Runs in milliseconds, 100% offline, absolute data privacy.

### Phase 5: Confidence Estimation & Reporting
- A Monte Carlo / Ensemble variance estimator calculates a final **Confidence Score** for the detection.
- A human-readable text report and a structured JSON object are returned, explicitly flagging "Severe" or "Moderate" interactions for the clinician while silently ignoring safe ("No Interaction") pairings.

## 3. Evaluation Criteria
The system is benchmarked on a separate test set of 233 prescriptions with physician-audited ground truth (pairs: Moderate 172, Severe 66, Mild 5).
- **Binary Classification Task:** Moderate/Severe = 1 (alert), Mild/No Interaction = 0.
- **Severity Task (headline):** per-class precision/recall/F1 plus accuracy, macro and weighted F1 — binary alone hides severity errors.
- **Metrics Tracked:** AUC-ROC, Specificity (True Negative Rate), Sensitivity (Recall), per-class F1, Macro F1-Score.