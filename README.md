# AI Clinical Copilot

## Project Overview

The AI Clinical Copilot is a machine learning-based clinical decision support system designed to address the digitization gap in Indian healthcare. It processes unstructured prescription text (from OCR-scanned handwritten prescriptions or voice commands) to extract medication entities and detect potential drug-drug interactions (DDI). A hybrid approach combining deep learning models with traditional rule-based methods ensures robustness in real-world clinical settings.

The system ships as a premium PyQt6 desktop application (`clinical_copilot_ui.py`) backed by an ML/NLP pipeline (`clinical_copilot_backend.py`), with a one-time cache build step (`build_cache.py`) for fast startup.

---

## System Architecture

### High-Level Pipeline

The backend implements a multi-phase processing pipeline:

1. **Entity Extraction** — Hybrid extraction of medication names from unstructured text using GLiNER NER, FlashText dictionary matching, semantic embeddings, and fuzzy matching.
2. **Interaction Detection** — Pairwise drug-drug interaction checks against the DrugBank database.
3. **Severity Classification** — Local ML pipeline utilizing a stacked ensemble of custom deep learning architectures (production: FastClinicalCNN + MedSeverityNet + TinyClinicalFormer through an XGBoost meta-model; extended search running to keep CNN + Transformer only, cutover at promotion) trained on physician-audited severity labels.
4. **Confidence Estimation** — Probabilistic assessment of prediction reliability using ensemble calibration, continuous risk scores, and uncertainty quantification.

### Data Sources

- **DrugBank JSON Database** (`full_database.json`) — International drug interaction database providing DDI information with descriptions and DrugBank IDs.

---

## Deep Learning and Machine Learning Components

### 1. GLiNER Medical Named Entity Recognition (NER)

**Model:** `E3-JSI/gliner-multi-med-ner-synthetic-v1`

GLiNER is a span-based zero-shot NER model used here for medical entity extraction. It runs alongside a FlashText dictionary over ~48k generic/alias/brand keywords, with mention grounding to reject hallucinated entities.

**Entity types extracted:** MEDICATION, DOSAGE, FREQUENCY, DURATION, ROUTE, CONDITION.

**Fallback:** If the ML model is unavailable, the NER module falls back to regex-based extraction for medications, dosages, and frequencies.

### 2. Semantic Embedding Search

**Model:** `ibm-granite/granite-embedding-278m-multilingual` (278M parameters, 768-dim embeddings)

The `SemanticEmbeddingGenerator` encodes prescription text and known medicine names into dense vectors, then performs cosine-similarity search to find the closest matching drugs. Medicine-name embeddings are pre-computed once (via `build_cache.py`) and loaded from a `.safetensors` file at runtime for speed.

**Use cases:** semantic synonym detection (e.g. Paracetamol ↔ Acetaminophen), OCR error recovery, and entity disambiguation.

### 3. Fuzzy String Matching

An always-on fallback layer using `difflib.get_close_matches`. N-grams (1–5 tokens) from the input text are compared against the medicine name list with a configurable similarity cutoff (default: 0.75).

### 4. Severity Classification & Ensemble Stacking

To replace slow Cloud LLMs, we trained localized deep learning models (production: FastClinicalCNN + MedSeverityNet + TinyClinicalFormer; extended search running to retire the BiLSTM and keep CNN + Transformer):
- **FastClinicalCNN** (Temporal Convolutional Network)
- **MedSeverityNet** (CharCNN + BiLSTM — in production, retirement in progress)
- **TinyClinicalFormer** (Lightweight Transformer)

The models output logits which are fused using an **XGBoost Stacking Meta-Model**. This architecture operates in milliseconds. Training labels are physician-audited (multi-agent adjudication with a standardized severity codebook, not single-LLM heuristics). Known limitation, recorded in UPDATE_3.md: the stacker trains on simulated out-of-fold features from the full training set (no per-fold retraining at n=1600), and temperature scalars sit at 1.5 (refit on held-out val NLL moves nothing) — so probability sharpness is handled by codebook decision policy, not calibration.

### 5. Confidence Estimation and Uncertainty Quantification

The `ConfidenceEstimator` computes:

- Mean and standard deviation of prediction score distributions
- 95% and 99% confidence intervals (percentile-based)
- Temperature-scaled calibration (default T = 1.5)
- Ensemble variance across multi-method agreement

Low-confidence entities are flagged for manual review by clinicians.

### Hybrid Ensemble Strategy

All three extraction methods run in sequence and their results are fused with weighted confidence:

| Method | Confidence Weight |
|---|---|
| GLiNER NER | ≈ 0.8 |
| Semantic Embeddings | ≈ 0.85 |
| Fuzzy Matching | ≈ 0.75 |

Entities detected by multiple methods receive higher effective confidence. The `EntityExtractor` also filters out common non-medicine stopwords (e.g. "daily", "tablet", "breakfast") to reduce false positives.

---

## Component Details

### `clinical_copilot_backend.py`

The backend module containing the full ML/NLP pipeline. Key classes:

| Class | Purpose |
|---|---|
| `ConfigLoader` | Loads and manages configuration from `config.yaml` with nested key access and default fallbacks. |
| `BioNERModule` | GLiNER-based medical NER, with regex fallback. |
| `SemanticEmbeddingGenerator` | Sentence-transformer embeddings for semantic similarity search. Supports batch pre-computation, in-memory caching, and top-K retrieval. |
| `ConfidenceEstimator` | Uncertainty quantification via ensemble statistics, confidence intervals, and temperature scaling. |
| `MedicineDataLoader` | Loads the DrugBank JSON database. |
| `EntityExtractor` | Multi-method entity extraction combining NER, embeddings, and fuzzy matching with entity validation and deduplication. |
| `InteractionDetector` | Detects drug-drug interactions from the DrugBank index. Supports fuzzy drug-name lookup and pairwise interaction scanning. |
| `ClinicalCopilotPipeline` | Main orchestrator. Tries cache-first loading for fast startup; falls back to raw data parsing. Exposes `process_input()` and `generate_clinical_report()`. |

### `clinical_copilot_ui.py`

A premium PyQt6 desktop application providing a dark-themed interface for prescription analysis.

**Key features:**

- **Two-panel layout** — Left panel for prescription text input, right panel for analysis results.
- **Asynchronous analysis** — The `AnalysisWorker` runs the backend pipeline in a `QThread` with real-time phase progress updates (4 phases: entity extraction → formatting → interaction detection → confidence scoring).
- **Gemini-powered severity + rephrasing** — If a `GEMINI_API_KEY` environment variable is set, the keyword pipeline grades interaction severity with gemini-3.8-flash and can rephrase DrugBank descriptions into simple patient-friendly language (toggleable via `reasoning_enabled` in `config.yaml`).
- **Rich result rendering** — Identified medicines shown as `MedicineTile` cards with generic name, composition, price, manufacturer, and Rx status. Interactions shown as `InteractionCard` widgets with reasoning and DrugBank references. Confidence gauges with animated progress bars.
- **Safety alerts** — A prominent banner when interactions are detected, with colour-coded confidence indicators (green ≥ 80%, yellow ≥ 50%, red < 50%).

**Running the UI:**

```bash
uv run python clinical_copilot_ui.py
```

### `build_cache.py`

A one-time cache-building script that pre-computes heavy data so the UI loads in seconds instead of minutes.

**What it does (4 steps):**

1. **Load DrugBank JSON** — Reads `full_database.json` and extracts all medicine names.
2. **Compute embeddings** — Encodes all medicine names with the Granite embedding model on GPU (CUDA) if available, otherwise CPU. Saves to `cache/medicine_embeddings.safetensors`.
3. **Save medicine metadata** — Writes the ordered list of medicine names to `cache/medicine_data.json`.
4. **Build DrugBank index** — Constructs a lookup index keyed by drug name and DrugBank ID, storing only interaction data. Saves as `cache/drugbank_index.json.gz` (gzip-compressed, typically ~30–50 MB vs 2.4 GB raw).

**Usage:**

```bash
uv run python build_cache.py
```

**Cache output files:**

| File | Description |
|---|---|
| `cache/medicine_embeddings.safetensors` | Pre-computed 768-dim embeddings for all medicine names |
| `cache/medicine_data.json` | Ordered list of medicine names |
| `cache/drugbank_index.json.gz` | Gzip-compressed DrugBank interaction index |

After building the cache, the UI and backend will automatically load from it on next startup.

---

## Setup and Installation

### Prerequisites

- Python ≥ 3.11
- CUDA-capable GPU (optional, recommended for faster cache building and inference)

### Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install all dependencies with:

```bash
uv sync
```

> If you don't have `uv` installed yet, see: https://docs.astral.sh/uv/getting-started/installation/

### Database File Required

- `full_database.json` — DrugBank interaction database (place in project root).

### Configuration

All settings are in `config.yaml`. Key sections:

```yaml
models:
  ner:
    model_name: "E3-JSI/gliner-multi-med-ner-synthetic-v1"
    device: "cpu"            # "cuda" for GPU
    confidence_threshold: 0.6
  embeddings:
    model_name: "ibm-granite/granite-embedding-278m-multilingual"
    device: "cpu"
    similarity_threshold: 0.85

pipeline:
  use_ml_ner: true
  use_embeddings: true
```

### Environment Variables

Create a `.env` file in the project root (optional):

```
GEMINI_API_KEY=your_api_key_here
```

When set, the keyword pipeline grades interaction severity with gemini-3.8-flash, and the UI can use it to rephrase interaction descriptions into patient-friendly language (toggleable via `reasoning_enabled` in `config.yaml`).

---

## Usage

### Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Build cache (one-time, takes a few minutes)
uv run python build_cache.py

# 3. Launch the desktop UI
uv run python clinical_copilot_ui.py
```

### Programmatic Usage

```python
from clinical_copilot_backend import ClinicalCopilotPipeline

pipeline = ClinicalCopilotPipeline(config_path="config.yaml")

result = pipeline.process_input("Aspirin 75mg + Clopidogrel 75mg + Atorvastatin 10mg")
report = pipeline.generate_clinical_report(result)
print(report)
```

### Pipeline Output Structure

```python
{
    'identified_medicines': ['Aspirin', 'Clopidogrel', 'Atorvastatin'],
    'entity_confidence_scores': [0.85, 0.85, 0.75],
    'extraction_methods': ['flashtext_dictionary', 'gliner_ner', 'semantic_embeddings'],
    'drug_interactions': [
        {
            'drug_a': 'Aspirin',
            'drug_b': 'Clopidogrel',
            'description': 'Increased risk of bleeding...',
            'drugbank_id': 'DB00945',
            'detection_confidence': 0.85
        }
    ],
    'confidence_metrics': {
        'entity_extraction': { 'mean': 0.85, 'std': 0.05, 'confidence': 0.95 },
        'interaction_detection': 0.85,
        'overall_confidence': 0.90
    },
    'metadata': {
        'total_medicines': 3,
        'total_interactions': 1,
        'has_safety_concerns': True,
        'ml_enabled': True,
        'embeddings_enabled': True,
        'low_confidence_entities': 0
    }
}
```

---

## Performance and Validation

The system is validated by end-to-end testing on a physician-audited prescription set plus cross-validated architecture search. Known limitations are recorded in UPDATE_3.md (simulated-OOF stacker, temperature at 1.5, label-prevalence gaps) rather than claimed away.

### End-to-End Pipeline Metrics
On the final test set of 233 prescriptions with physician-audited ground truth (pairs: Moderate 172, Severe 66, Mild 5):
* **Ensemble (local):** AUC-ROC 0.9746, Sensitivity 100.00%, Specificity 97.06%, binary F1 99.75%; accuracy 93.56%, macro F1 95.49%, weighted F1 93.55%; per-class F1 No Interaction 98.41% / Mild 100.00% / Moderate 94.46% / Severe 89.06% — $0 per run. 2-model CNN+Transformer, LogReg stacker, learned Severe-vs-rest codebook gate (config: severity_gate).
* **Keyword (gemini-3.8-flash, severity-only, detailed codebook prompt, low thinking):** AUC-ROC 0.9712, Sensitivity 100.00%, Specificity 91.18%, binary F1 99.25%; accuracy 90.13%, macro F1 68.72% (Mild F1 0 — the DB-keyword floor is Moderate, so this path cannot predict Mild), Moderate 91.39% / Severe 85.07% — measured $0.42 per full refresh (250 calls, 0 errors; low thinking cut thinking tokens ~3× vs medium).

### Visualizing the Advantage
The project includes a suite of presentation-ready visualizations that highlight the clinical and structural advantages of the local ML approach. Run `uv run generate_comparative_charts.py` to generate:

1. **Trade-off Radar Chart (`chart_6_tradeoff_radar.png`):** Proves that while Cloud LLMs match the Custom ML on F1-Score and AUC-ROC, our Local ML completely dominates on Speed, Privacy, and Cost-Efficiency.
2. **Alert Fatigue vs. Missed Threats (`chart_7_alert_fatigue.png`):** Maps abstract ROC curves into real-world hospital metrics: Nuisance Alerts vs Missed Severe Threats per 100 prescriptions, identifying the tunable "Sweet Spot".
3. **Reliability Calibration Diagram (`chart_8_calibration.png`):** Model predicted probability vs. observed severe-interaction fraction. Status note: temperatures sit at 1.5 evidence-backed and the stacker trains on simulated out-of-fold features (see UPDATE_3.md), so day-to-day reliability comes from the codebook decision policy; regenerate with `generate_comparative_charts.py`.

### Memory Requirements

| Component | Approx. Size |
|---|---|
| GLiNER NER model | Downloads on first run (cached) |
| Granite embedding (278M params) | ~1.1 GB weights + precomputed safetensors cache |
| DrugBank data (cache) | ~30–50 MB compressed index |

### Inference Speed (CPU)

- Entity Extraction: 0.5–1.5 s per prescription
- Interaction Detection: < 0.2 s
- Total Pipeline: ~2–3 s per prescription

---

## Safety and Clinical Validation

This system is designed for **clinical decision support**, not autonomous decision-making:

- All predictions include confidence scores.
- Low-confidence entities (< 0.7) are flagged for review.
- Drug interactions are presented as warnings, not absolute contraindications.
- Final validation must be performed by licensed healthcare professionals.

### Known Limitations

1. Performance depends on quality of input text (OCR/ASR noise degrades results).
2. DrugBank database may not include all regional or recently approved drugs.
3. The system does not replace clinical judgment or pharmacological expertise.

---

## Regulatory and Compliance Notes

This system is designed as a Clinical Decision Support Tool (CDST):

- **India:** Medical Device Rules 2017, Digital Health Mission guidelines.
- **Data Privacy:** Must comply with Personal Data Protection Bill.
- **Validation:** Requires clinical validation studies before deployment.
- **Liability:** Healthcare providers remain responsible for final prescribing decisions.

**Disclaimer:** This software is provided for research and development purposes. It is not intended for diagnostic or therapeutic use without proper clinical validation and regulatory approval.

---

## License and Citation

**Citation:**
```
AI Clinical Copilot Team (2026).
AI Clinical Copilot - Backend Infrastructure with ML/DL/NLP.
Symbiosis Institute of Technology, Pune.
```

---

## Contact

**Institution:** Symbiosis Institute of Technology, Pune
**Project Guide:** Dr. Ranjeet Bidwe
**Development Team:**
- Aayush Joshi (PRN: 23070122008)
- Ankush Dutta (PRN: 23070122032)
- Archisha Yadav (PRN: 23070122041)
- Aryan Srivastava (PRN: 23070122055)

---

## Acknowledgments

- **GLiNER** — E3-JSI (medical named entity recognition)
- **Sentence Transformers** — UKP Lab, TU Darmstadt
- **DrugBank** — University of Alberta, Canada
- **Hugging Face Transformers** — Hugging Face Inc.
- **Google Gemini** — Google DeepMind (interaction description rephrasing)