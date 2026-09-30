# Embedding Model Comparison & Selection

**Decision (2026-09-15):** `ibm-granite/granite-embedding-278m-multilingual` replaces
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` as the production embedding model in `config.yaml`.
The cache (`cache/medicine_embeddings.safetensors`) has been rebuilt with Granite embeddings
(19,830 drug names x 768 dims) and `outputs/Ensemble/` reflects a Granite run.

## Results

Evaluated end-to-end on 100 prescriptions (the original `100_prescriptions.json`, 215 ground-truth drug mentions).
The severity ensemble was held fixed so differences come only from entity linking.
The fixture set has since grown to 296 entries in `test_prescriptions.json` (the current default dataset);
numbers below still reflect the original 100-prescription run until re-evaluated.

| Model | F1 | AUC-ROC | Sensitivity | Specificity | Drug Name Match | Latency |
|---|---|---|---|---|---|---|
| **granite-embedding-278m** | **91.60** | **88.36** | 85.71 | 96.67 | **95.81%** (206/215) | **0.24s** |
| WeMM-Embedding-2B | 91.60 | 88.36 | 85.71 | 96.67 | 95.81% (206/215) | 0.92s |
| Qwen3-VL-Embedding-2B | 90.77 | 87.62 | 84.29 | 96.67 | 95.35% (205/215) | 0.46s |
| pubmedbert-base | 90.77 | 87.62 | 84.29 | 96.67 | 95.35% (205/215) | 0.27s |
| SapBERT (previous baseline) | 90.77 | 87.62 | 84.29 | 96.67 | 95.35% (205/215) | 0.25s |
| embeddinggemma-300m | 89.92 | 86.88 | 82.86 | 96.67 | 94.88% (204/215) | 0.39s |
| Qwen3-Embedding-0.6B | 83.33 | 73.67 | 85.71 | 53.33 | 95.35% (205/215) | 0.36s |

Granite ties WeMM-2B on every accuracy metric; it wins on latency (4x), model size (7x smaller),
and it needs no `trust_remote_code`. Versus the previous SapBERT baseline: **+0.83 F1, +0.74 AUC,
+0.5pp drug-name match rate**.

## Why the drug-name match metric

Embeddings do not classify severities directly. They link extracted drug mentions to canonical
DrugBank names (Phase 2: Semantic Matching), which determines which DDI pairs are looked up.
The final F1/AUC is dominated by the (fixed) severity ensemble, so several models tie there.
Drug Name Match Rate = (215 - missed mentions) / 215, where missed mentions are ground-truth drugs
the pipeline failed to identify. It isolates the embedding's contribution to the pipeline.
The metric is computed by `evaluate_ddi_system.py` and stored under `drug_name_matching` in
`outputs/Ensemble/metrics.json`.

## Method

- `run_embedding_comparison.py` swaps `models.embeddings.model_name` in `config.yaml`, rebuilds the
  embedding cache (`build_cache.py`), and re-runs `evaluate_ddi_system.py --mode ensemble` per model.
- The severity ensemble (`cache/ensemble_*.pt`) is not retrained between models, so it cannot
  confound the comparison.
- Environment: RTX 4060 Laptop (8 GB), `sentence-transformers` 6.0.1, `torch` 2.11.0+cu128.
- The 2B multimodal models run in bfloat16; `build_cache.py` falls back to smaller batches
  automatically on CUDA OOM. Batch sizes: Granite 64, gemma 256, SapBERT 2048, 2B models 64.
- `google/embeddinggemma-300m` is gated on Hugging Face and was only run after accepting the
  Gemma license and authenticating with an HF token.

## Notes & caveats

- Granite and WeMM-2B produced identical metrics; the tie-break is operational (latency, size,
  no remote code), which is why Granite was selected.
- `Qwen/Qwen3-Embedding-0.6B` collapses on specificity (53%), producing many false-positive
  "Severe" calls.
- `full_database.json` (2.4 GB DrugBank export) was missing during this study. `build_cache.py`
  now reuses cached medicine names/synonyms/index when the JSON is unavailable and only recomputes
  embeddings, instead of rebuilding an empty cache. Restore the JSON for full cache rebuilds.
- Comparison artifacts are kept in `outputs/`: `embedding_comparison_results.json`,
  `chart_embedding_comparison.png`, and per-model metric backups (`metrics_*.json`).

## Architecture

The backend is now organized as the `clinical_copilot` package; the root-level scripts are thin entry points.

```
clinical_copilot/
  config.py       ConfigLoader: single source of config access and defaults
  data.py         MedicineDataLoader (DrugBank JSON)
  ner.py          BioNERModule: GLiNER/BioBERT medication entity extraction
  embeddings.py   SemanticEmbeddingGenerator + shared model loading (dtype/batch/trust_remote_code)
  extraction.py   EntityExtractor: FlashText + fuzzy matching + semantic entity linking
  ddi.py          InteractionDetector: DrugBank index, fuzzy lookup, severity routing, Gemini client
  severity.py     single severity taxonomy (Mild/Moderate/Severe + No Interaction outcome), ranks, risk scores
  confidence.py   ConfidenceEstimator: Monte Carlo dropout, ensemble, temperature scaling
  pipeline.py     ClinicalCopilotPipeline.process_input(text, progress=...): orchestration and reporting
  plots.py        shared matplotlib/seaborn helpers (confusion matrix, ROC, decision curve, metric bars)
  paths.py        central output directory (outputs/)
  cli.py          scenario runner
```

Design rules applied:

- **One pipeline.** The UI no longer re-implements entity extraction, interaction
  detection, and confidence scoring; `AnalysisWorker` calls
  `pipeline.process_input(text, progress=...)` and renders the result. Phase progress
  messages are emitted by the pipeline through the callback.
- **One severity taxonomy.** Classifier mapping, evaluator, tuner, UI, tests, and the
  label-prep scripts all use `severity.py` (`Mild/Moderate/Severe` model classes,
  `No Interaction` as the prescription-level outcome). Legacy strings (`VERY SEVERE`)
  normalize through the same table.
- **One plotting layer.** Evaluation reports and the embedding comparison chart share
  `plots.py`; scripts contain no matplotlib boilerplate.
- **One output directory.** Every generated artifact (evaluation reports, comparison
  results and logs, charts, tuning output, scenario dumps) is written under `outputs/`,
  resolved through `clinical_copilot/paths.py`.

## Reproducing

```bash
uv run run_embedding_comparison.py                       # all models
uv run run_embedding_comparison.py --models WeMM,Qwen3-VL  # subset
uv run build_cache.py                                    # rebuild cache for config.yaml model
uv run evaluate_ddi_system.py --mode ensemble            # regenerate outputs/Ensemble/
```

## Severity ensemble architecture search

**Decision (2026-09-16):** the severity ensemble's architectures, training
hyperparameters, and XGB stacker settings are now chosen by search evidence,
not hand-picked values. Production weights (`cache/ensemble_*.pt`) were replaced
with the winners; every model improved (table below).

**Why.** All three base models (MedSeverityNet, FastClinicalCNN, TinyClinicalFormer)
used hand-chosen layer sizes, dropout, learning rate, and stacker settings with no
search behind any of them. The search answers "how did you decide" with data.

**Protocol** (`tune_severity_search.py`, CV-only: the 1600/400 split is never re-cut).

1. *Screen.* 48 random configs per model plus the hand-picked baseline under the
   identical protocol (20 epochs, fixed 1300/300 internal split). Objective: dev
   macro-F1 (3-class imbalance), AUC secondary.
2. *Refine.* Top 8 per model plus baseline through stratified 3-fold CV on the full
   1600 train (identical folds for every config). Winner = best pooled out-of-fold
   macro-F1; ties within 0.002 go to fewer non-embedding parameters.
3. *Stack.* LogReg C {0.1, 1, 10} and 18 XGB configs via 5-fold CV on the winners'
   temperature-calibrated OOF features (same calibration as `evaluate_ensemble.py`).
4. *Finalize.* Winners retrained on the full 1600 (val-400 checkpointing, same recipe
   as `train_ensemble.py`) to `outputs/severity_search/final_ensemble_*.pt`; 95%
   bootstrap CIs (2000 resamples) on pooled OOF for winner, baseline, and their
   difference; params-vs-F1 Pareto plot.

**Search spaces.** Shared: word embed {64,128,256}, char embed {32,64}, char filters
{32,64,128}, char kernels {[2,3,4],[2,3,4,5]}, classifier hidden {64,128,256},
dropout {0.2,0.3,0.4,0.5}. Training: lr {3e-4,1e-3,3e-3}, weight decay {0,1e-4,1e-3},
batch {64,128}, label smoothing {0,0.1}, distillation alpha {0.25,0.5,0.75}.
Arch: LSTM hidden {128,256,384} x layers {1,2,3}; CNN filters {64,128,256} x kernels
{[2,3,4],[2,3,4,5],[3,4,5]}; Transformer layers {1,2,4} x heads {2,4,8} x ff
{128,256,512} (heads constrained to divide d_model). ~400k-1.2M combos per model;
random search covers each dimension with 48 distinct values (cf. Bergstra & Bengio 2012).

**Results** (3-fold OOF; CIs are winner-minus-baseline macro-F1, p=1.0 throughout).

| Model | Winner OOF F1 | Baseline OOF F1 | Gain 95% CI | Core params |
|---|---|---|---|---|
| MedSeverityNet | 0.957 | 0.898 | +0.041 – +0.076 | 3.15M → 0.43M |
| FastClinicalCNN | 0.958 | 0.888 | +0.053 – +0.087 | 0.81M → 0.18M |
| TinyClinicalFormer | 0.944 | 0.875 | +0.051 – +0.088 | 1.69M → 2.41M |

Val-400 after cutover (caveated, see below): LSTM 0.9395 → 0.9539, CNN 0.9442 → 0.9518,
Transformer 0.8237 → 0.9450; LogReg stack 0.9600. `test_backend.py` 8/8.

**What the search found.** Smaller wins almost everywhere: LSTM 2x256 → 1x128, CNN
filters 128 → 64 with kernels [2,3,4], char-CNNs and classifier heads shrunk;
weight decay → 0 in all three winners (it was hurting); distillation alpha 0.5 → 0.75
in all three; the transformer went deeper but narrower (4 layers x 8 heads, ff 128,
lr 3e-4). Stacker evidence kept LogReg C=1.0 and shrank XGB to 50 trees/depth 2
(applied in `evaluate_ensemble.py`). Locked architectures live in `ENSEMBLE_ARCHS`
(`custom_severity_model.py`); all construction sites (`ddi.py`, `evaluate_ensemble.py`,
`train_ensemble.py`) build through `build_ensemble_model()`, so shapes cannot drift
from the weights again.

**Artifacts.** `outputs/severity_search/`: `screen_trials.jsonl` (147), `refine_trials.jsonl`
(27), per-candidate OOF arrays, `stack_results.json`, `summary.json`, `pareto.png`,
111 charts in `charts/` (pivot/correlation heatmaps, per-hyperparameter boxplots,
fold-stability bars, rankings, stacker comparison), final weights.

```bash
uv run tune_severity_search.py --stage all --n-configs 48 --screen-epochs 20  # full search
uv run plot_search_charts.py          # regenerate outputs/severity_search/charts/
uv run evaluate_ensemble.py           # refit temps + stacker on promoted weights
```

## Retrieval upgrades: fuzzy net, alias embeddings, cleaning, combos (2026-09-17)

Four of the nine retrieval ideas are implemented: #1, #3, #6 (`92f0bad`) and #2
(`4ea3a74`). New deps: `rapidfuzz==3.14.6`, `symspellpy==6.10.0`.

**#3 — `_clean_mention()`** (`clinical_copilot/extraction.py`): the old greedy `.*`
regexes replaced with ordered strip rules (dose/units, 20+ dosage forms,
frequencies incl. BD/OD/TDS, XL/SR-style modifiers, bare numbers, stray
punctuation; `- / + &` preserved for #6). Applied before semantic retrieval and
inside `_resolve_names`. `Dolo 650 twice daily→dolo`, `Lescol 40 MG→lescol`,
`Starpress XL 25mg TAB→starpress`.

**#6 — `_split_combo()`**: splits on `/ + &` and and/with/plus, plus a
suffix→ingredient map (AM→amlodipine, H→hydrochlorothiazide, M→metformin,
D→domperidone; Plus/LS stem-only). Suffix splits require an EXACT stem, so
`vitamin-d` never becomes domperidone. `amoxicillin + clavulanate` yields both
drugs. `ramistar-am` still resolves via the `build_cache.py` data override,
kept deliberately: it is baked into `cache/medicine_synonyms.json` and the
reuse path never re-applies overrides, so deleting the line changes nothing
until a full DrugBank rebuild — removable once #4 lands. Extraction itself has
no special case anymore.

**#1 — fuzzy stage 3** (the dead `difflib` code, revived): RapidFuzz WRatio
proposes over 55k generics+aliases; SymSpell with a dictionary built from our
drug corpus (not general English) corroborates. Accept a hit only at RF ≥92
with a 3pt margin over a different-drug runner-up, or when both stages agree on
the same drug — otherwise abstain (a wrong drug is worse than a miss).
Whole-text net fires only when nothing matched: 1–2 grams, skipping connectors,
stopwords, and all-English tokens via symspellpy's 82k word list (kills
`takes→Tantalum`, `infection→Technetium` junk found during testing).
`paracetmol/amoxcillin/rantidine` resolve at ~0.95; `telma/clav/augmentin/xyzqqq`
abstain. pytest 9/9, incl. the OCR-error and combination-drug scenarios.

**#2 — alias embeddings** (`build_cache.py`, `embeddings.py`, `pipeline.py`):
all 35,002 alias surface forms encoded (`cache/alias_embeddings.safetensors`,
102 MB + `cache/alias_data.json`; `build_cache.py --aliases-only` rebuilds just
these), max-pooled per concept at query time (`find_most_similar_concepts`),
threaded through `pipeline.py` with graceful fallback on old caches. The pool
must earn its win: a different-concept winner needs a 0.03 lead over the
generics-only winner (stops `amoxil` matching minoxidil's alias over
amoxicillin's generic); exact-alias queries bypass the margin. Measured on
granite-278m-multilingual (n=300 each): raw aliases top-1 0.40→**1.000**,
typo'd aliases 0.37→**0.960**, generics unchanged 1.000, multi-word aliases
3/8→**8/8**, +22 ms/query on CPU. Data gap found: the synonym file holds
chemical variants but zero trade brands (Amoxil/Tylenol/Lipitor/Ventolin all
absent) — that is #4's job.

**Known limits then:** Telma-H, Augmentin, Starpress honestly missed instead of
false-matching — fixed by #4 below. Still open: the pre-existing
`No known allergies→Nitric Oxide` flashtext hit, left alone as a stage-1
disambiguation problem for #7.

## Indian brand lexicon (idea #4, 2026-09-22)

**Decision:** `data/indian_brands.json` (2,000 brand concepts, 586 KB) now maps
Indian brand names to DrugBank-canonical ingredient lists and merges into
`synonym_map` at cache-build AND runtime. The brief's failure cases — Dolo,
Crocin, Augmentin, Telma-H, Ramistar-AM, Starpress — all resolve to correct
ingredients (verified end-to-end, 15/16 failure-battery cases pass; the 1 fail is
the pre-existing `no → nitric oxide` flashtext issue owned by #7).

**Sources.** (1) Common Drug Codes for India flat files (NRCeS/C-DAC Pune,
CC BY 4.0 — attribute C-DAC/NRCeS; nrces.in is captcha/WAF-gated so the
ohcnetwork/cdci mirror of the official package is used), joined
`BrandMaster → ProductMaster (short brand) → GenericMaster → SubstanceMaster`
(proven on a real row: `Dolo (paracetamol) 650 mg oral tablet` → `Dolo` →
`Acetaminophen 650 mg oral tablet` → `Acetaminophen`; combos `+`-join their
substance IDs). (2) `junioralive/Indian-Medicine-Dataset` (254k rows) as
frequency-ranked gap-fill — grey license, internal use only, never
redistributed (raw files stay in gitignored `data/raw/`). RxNorm/DailyMed were
researched and rejected as the primary source (US-only coverage: Dolo/Crocin/Telma
all MISS in RxNorm; DailyMed has no Micro Labs `Dolo 650`).

**Build** (`build_indian_brands.py`): CDCI join → junioralive gap-fill →
ingredient normalization to DrugBank names → brand-key collapsing → emission
with source/verified/priority flags. Outputs merged by `clinical_copilot/brands.py`
(`build_cache.py` before alias encoding, `pipeline.py` at load). Raw sources are
NOT in the repo: rebuild with the two downloads, or edit the JSON directly.

Hard-won correctness rules (each caught by the built-in QA report):
- **Collapsing only strips release/strength modifiers** (`xl/xr/sr/er/cr/ds/odt/la/pr`).
  Formula-changing suffixes never collapse: `meftal-forte` (mefenamic+paracetamol)
  != `meftal`, `telma-h` != `telma`, `pan-d` != `pan`.
- **Salt stripping only removes SALT_WORDS** — stripping any trailing word
  wrongly mapped `povidone iodine` → `povidone`.
- **`aspirin` is not a DrugBank canonical name** (`acetylsalicylic acid` is) —
  without the synonym every aspirin product was silently dropped.
- **Parenthetical word order** handled (`Calcitonin (Salmon)` → `salmon calcitonin`),
  and hyphen/spacing unified (`telma-h` == `telma h`).
- **CDCI votes 3x over junioralive** and longer tuples win near-ties: the scrape's
  `short_composition` has only 2 slots and truncates 3+ ingredient combos.
- Ambiguous brand names reused across companies for different drugs (`Alcet` =
  levocetirizine OR cetirizine; `Gemcal` = calcitonin OR calcitriol) are flagged
  `AMBIGUOUS`, not silently resolved.

**Verified output:** 2,000 concepts (206 priority, 1,896 multi-row verified,
211 combos). QA review queue (11 entries, flagged in the JSON `notes`): 4 market
ambiguities/data questions (`zydol`, `calcirol`, `nasivion`, `polybion`) and 7
truncated combos (`seroflo`, `sinarest`, `montair-lc`, `telekast-l`, `neosporin`,
`tusq`, `grilinctus`) needing pharmacist confirmation (Phase 3 of the plan).

```bash
python build_indian_brands.py            # regenerate data/indian_brands.json
uv run build_cache.py --aliases-only     # bake brands into synonyms + alias vectors
```

**#4 research verdict.** Three parallel research tracks agree. Primary source:
Common Drug Codes for India (NRCeS/C-DAC Pune, free TSV flat files, CC BY 4.0,
Aug 2026 release: 93,911 brands) — its `Brand Master → Generic Master →
Substance Master` join is exactly brand→ingredient(s), combos `+`-joined.
RxNorm covers only ~20–30% of Indian brands (live-verified Dolo/Crocin/Telma
MISS; Augmentin HIT) — use for normalization + MIN combos, not the map.
DailyMed/Orange Book are US-biased dead ends for brand discovery. Best gap-fill
seed: `junioralive/Indian-Medicine-Dataset` GitHub CSV (254k rows, grey
license — internal use only, never redistribute); CIMS for browser
verification; never scrape 1mg (ToS). Recommended build (3–5 days): CDCI join →
collapse to base brands → gap-fill → pharmacist spot-check ~30 FDCs vs CIMS →
`indian_brands.json` with source/verified flags → merge into `synonym_map` →
`--aliases-only` gives new brands vectors free while `_split_combo` and the
fuzzy corpus pick them up automatically.

```bash
uv run build_cache.py --aliases-only   # rebuild alias vectors after synonym_map changes
uv run pytest -q                       # 9 tests incl. OCR + combo scenarios
```



---

## Short-key disambiguation: the 
o fix

The pre-existing `no → nitric oxide` failure (`No known allergies` matching
DrugBank's chemical-formula alias `NO`) is resolved. Root cause: 67 short
synonym keys (≤2 chars) were registered as case-insensitive FlashText keywords
without filtering — `no`, `of`, `co`, `d`, `m`, etc. would replace any
matching English word with the canonical drug name before any stopword check.

**Fix (`clinical_copilot/extraction.py`):**

1. **Dual KeywordProcessor** — keys ≤2 chars go into a *case-sensitive*
   `KeywordProcessor` (`flashtext_short_processor`); keys ≥3 chars stay in
   the main case-insensitive processor. Short keys that are common English
   words (in `_ENTITY_STOPWORDS`) are excluded from the short processor
   entirely — `no` never matches regardless of case.
2. **`_resolve_names` guard** — when the surface form is a stopword ≤2 chars,
   resolution is skipped (blocks the NER/fuzzy fallback path too).
3. **`_ENTITY_STOPWORDS` expanded** — added `no`, `yes`, `not`, `is`,
   `are`, `was`, `be`, `do`, `so`, `up`, `if`, `it`, `as`,
   `or`, `an`, `we`, `he`, `me`, `my`.

63 short keys survive in the case-sensitive processor (4 English stopwords
filtered from 67). Valid drug abbreviations like `fe` → iron, `k` →
potassium, `zn` → zinc, `pa` → acetaminophen still match — they just
require exact case now (`FE` won't match `fe`, but `fe` will).

**Verified:** `No known allergies` → 0 medicines (was: Nitric Oxide).
`say no to drugs` → 0 medicines. `no known allergies` → 0 medicines.
`metformin 500mg` → Metformin. pytest 9/9, test_backend 8/8.

The failure battery is now **16/16** — all brief cases resolve including the
previously failing `no → nitric oxide` case.


---

## Encoder spike (#8): BioLORD-2023-M wins - straight swap recommended

Ran a lightweight multi-model retrieval spike (`eval_retrieval_spike.py`) on
36 queries across 5 categories (brand, typo, generic, multi-word, negative)
using the existing cache corpus (19,830 generics + 37,936 aliases) with the
production max-pool + margin retrieval logic. GPU: RTX 4060 Laptop.

| Model | Overall | brand | typo | generic | multi | neg |
|---|---|---|---|---|---|---|
| **BioLORD-2023-M** | **94.4%** | 85.7% | **100%** | **100%** | **100%** | **100%** |
| granite-278m (current) | 91.7% | 85.7% | 100% | 100% | 100% | 75.0% |
| BioLORD-2023 | 88.9% | 85.7% | 83.3% | 100% | 75.0% | 100% |
| BioLORD-2023-C | 88.9% | 85.7% | 66.7% | 100% | 100% | 100% |
| CODER-eng | 86.1% | 85.7% | 66.7% | 100% | 75.0% | 100% |
| SapBERT (mean-pool) | 83.3% | 85.7% | 50.0% | 100% | 75.0% | 100% |

**Key finding:** the only category where models differ meaningfully is negative
rejection. Granite false-positives on gibberish (`xyzqqq` -> benzquinamide at
0.853). BioLORD-2023-M correctly abstains on all 4 negative cases. In a
clinical system where false DDI alerts cost more than misses, this matters.

**Shared "failures" (all 6 models, test-design issues):**
- `crocin` -> predicted `crocin` not `acetaminophen`: crocin is itself a
  concept name in the generics list (score 1.000). Expected value should accept
  either. Not a retrieval failure.
- `ramistar-am` -> predicted `amlodipine` not `ramipril`: combo product,
  both ingredients are correct. Expected should accept either.

**Ship-it ablation (override-free):** 5/5 PASS with granite - salbutamol->albuterol,
HCTZ->hydrochlorothiazide, paracetamol->acetaminophen, PCM->acetaminophen,
amoxil->amoxicillin all resolve from representation alone (score 1.000).

**Decision: straight swap to BioLORD-2023-M** (`FremyCompany/BioLORD-2023-M`,
XLM-RoBERTa, 768d, ST-native drop-in). It beats granite on the benchmark and
requires no fusion, no pooling wrapper, no fine-tuning. The UMLS/SNOMED
yearly-reporting obligation applies to production use - resolve before shipping
commercially. For internal research/eval use, no restriction.

**Not done yet:** the actual `config.yaml` swap + cache rebuild
(`build_cache.py --aliases-only`) + full test battery re-run. That is the
next step when ready to cut over.

**Corrected model IDs** (research had wrong org names):
- `FremyCompany/BioLORD-2023-M` (not `FremyBel/BioLORD-2023-C`)
- `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (needs mean-pooling wrapper)
- `GanjinZero/coder_eng` (not "CODER-multilingual"; lowest traction, skip)

```bash
uv run python eval_retrieval_spike.py               # full comparison
uv run python eval_retrieval_spike.py --ablation     # override-free 5/5 check
```


---

## TF-IDF sparse retrieval + score fusion (idea #9)

Built a char-ngram TF-IDF index alongside the dense embeddings and fused
retrieval scores (dense + TF-IDF + RapidFuzz) via min-max-normalised linear
combination.

**New files:**
- `clinical_copilot/tfidf_index.py` — build/query/save/load + `fuse_scores()`,
  `should_accept()`, `segment_glued_token()`
- `cache/tfidf_vectorizer.pkl` (277 KB) + `cache/tfidf_matrix.npz` (12 MB)
  + `cache/tfidf_manifest.json` (content-hash for staleness detection)

**Build:**
```bash
uv run build_cache.py --tfidf-only   # rebuild sparse index from cached texts
```

**Architecture:**
- `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=2,
  sublinear_tf=True)` over 57,766 texts (generics + aliases) -> 16,611 features
- Query: sparse dot product, top-50 candidates, max-pool over aliases
- Fusion: `0.55*dense + 0.30*tfidf + 0.15*rf`, min-max normalized per query
- Accept at >= 0.62 with 0.03 lead rule (mirrors `_POOL_MARGIN`)
- RapidFuzz scores the union top-50 (not full corpus) -> fixes its 20-50 ms cost

**Glued-token segmentation:**
- Pre-peel dose/form with existing regexes
- Run only on spaceless tokens >= 8 chars
- Accept splits only if EVERY piece is a known drug (in `_known_names`)
- Prevents over-splitting: `metformin` -> `met+form+in` is rejected because
  `met` and `form` are not known drugs

**Wiring:**
- `build_cache.py`: step `[2c/4]` + `--tfidf-only` flag
- `pipeline.py`: loads TF-IDF at cache-load time, passes to `EntityExtractor`
- `extraction.py`: `_fuse_retrieval()` replaces `find_most_similar_concepts`
  when TF-IDF is available; `_segment_glued()` tries token splitting first

**Known behavioral change:** the fusion threshold (0.62) is lower than the
old dense-only threshold (0.85), so the diabetes scenario now identifies 4
medicines instead of 3 (extra "Metoprolol" false positive). The min-max
normalisation maps the best candidate to ~1.0 regardless of absolute scores,
making the 0.62 floor nearly always cleared. The 0.03 lead rule is the real
filter. Threshold tuning needed — the initial 0.55/0.30/0.15 weights and
0.62 cutoff are starting points from research, not calibrated values.

**Verified:** pytest 9/9, test_backend 8/8, `--tfidf-only` builds in 1.2s.

```bash
uv run build_cache.py --tfidf-only   # build index
uv run pytest -q                      # 9 tests
uv run python test_backend.py         # 8 scenarios
```

## Alias-pool cleanup (fix TNR regression from Indian-brand short aliases)

Phantom entities (\omeprazole\ -> \olanzapine\ via alias \ole\) traced to short brand fragments in the dense alias pool (subagent audit: 464 len<=3 + 514 len==4 surfaces, 45 mapping to olanzapine). Fixes, all in data/build rather than runtime:
- \data/indian_brands.json\: pruned 2011 -> 1819 concepts (drop len<=2, len<=4 non-priority, junk patterns); priority shorts (\pan\,\mox\,\dolo\,\	usq\,\iso\) kept
- \uild_indian_brands.py\: emission guard (\_ok_surface\) so future rebuilds stay clean
- \clinical_copilot/brands.py\: merge-time junk guard (defense in depth)
- \uild_cache.py\: DrugBank junk-alias filter (IUPAC>60, parens/brackets, CAS, non-alpha) + brand sidecar (\cache/brand_keys.json\) deleting stale brand keys on rebuild
- \cache/medicine_synonyms.json\ (gitignored): one-time migration removed 269 stale + 8819 junk keys (37943 -> 28867); alias pool 37936 -> 28860 vectors (102 -> 85 MB)
- Runtime keeps phantom-entity grounding filter (exact + fuzzy word match) + dual FlashText as safety nets

100-prescription ensemble eval after rebuild: AUC 0.8507, Sens 0.8143, TNR 0.9667, F1 0.8906 (TNR fully restored; old: 0.879/0.857/0.967/0.916). pytest 9/9, test_backend 8/8.


## Full-dataset ensemble eval (296 prescriptions, cleaned cache)

AUC 0.7164, Sens 0.8164, TNR 0.6966, F1 0.8387, drug-name match 0.797 (515/646). Exact 4-way severity match 143/296.

Error breakdown (from detailed_results.csv):
- 27 FPs, all n=1: real DrugBank-listed interactions (Azithromycin+Loratadine QTc, Ramipril+Amlodipine, Losartan+Amlodipine...) that annotators labeled No Interaction — dataset label gap, not extraction bugs.
- 38 FNs in two flavors: n=0 extraction/lookup misses (OCR typos Capcitabine/Tridihexethil/Icosopent/Cefpodoxim, glued Simvastatin20mg, rare Penfluridol/Amoxapine, Oxymetholone/AMG-222) and n=1 pred=Mild where the model under-scores real interactions (Abacavir+Tenofovir, Warfarin+Ibuprofen).
- Brand-name misses (Omez, Voveran, Telma, Dolo...) are eval name-matching artifacts: pipeline extracts correct generics (verified case-by-case); severity predictions on those rows are right.

Caveat: NER shows run-to-run variance on some brand rows (GLiNER), so name-match counts wobble between runs; severity metrics are the stable signal. Keyword-mode full eval still pending (needs paid Gemini key).


## Test-set cleanup + cleaned full eval (233 rows)

Per audit: removed 63 contrived rows (fake/invest/vet/withdrawn/IV-as-oral drugs, duplicate-therapy pairs, lethal unit errors, Loratadine/Omeprazole template clones), fixed 6 accidental text typos (Capcitabine, Cefpodoxim x2, Tridihexethil, Icosopent, Mofebutasone; kept deliberate V1tam1n OCR noise), corrected absurd doses on 16 kept rows (Loratadine 500mg->10mg, Omeprazole 500mg->20mg, Levothyroxine-mcg rows untouched, etc.). GT drugs[]/interactions untouched. Old file recoverable via git history.

Cleaned ensemble eval: AUC 0.6627, Sens 0.8187, TNR 0.5968, F1 0.8333, name-match 0.763 (389/510). TNR drop vs uncleaned (0.70) is mechanical: removed rows were 25 TNs + 2 FPs (easy negatives), leaving the 25 hard DrugBank-vs-label disagreements concentrated.

Verified the prune caused zero GT regressions (no dropped alias hits any GT token). Remaining gaps: (1) never-covered brands (warf->warfarin, looz), rare generics (amoxapine, penfluridol); (2) severity model under-scoring real interactions to Mild (17 FNs incl. Abacavir+Tenofovir, Warfarin+Ibuprofen); (3) one-directional DrugBank lookup + 0.85 name gate; (4) NER run variance on some brand rows; (5) 25 FPs are all real DrugBank-listed interactions the annotators call No Interaction (label-policy question). 100_prescriptions.json left untouched (frozen IDs 1-100 slice).


## Fix experiments A-D (all kept, +12 TP / +0 FP)

Threshold tuning rejected without running: risk scores are bimodal (~0.997 found vs ~0.05 none), so moving 0.5 only flips Mild preds (Sens 0.82->0.92 but TNR 0.60->0.47) - bad trade, conceptually wrong.

A. Reverse DrugBank lookup (ddi.py check_interactions retries swapped on miss): 405 Pheniramine-Alprazolam fixed (fwd name-match 0.64, rev 0.87); 247 found but scored Mild. +1 TP, +0 FP.

B. segment_glued_token guard (tfidf_index.py: skip split when token itself is known) + multi-word grounding (_is_mentioned_in_text: consecutive-word regex for spaced surfaces): root-caused via call-spy bisection - 'phenobarbital' was split to phe/no/barbital (all 'known'), fragments matched wrong drugs, phantom filter dropped them, high fragment scores skipped fuzzy recovery (double kill); 'ethinyl estradiol' surface could never fuzzy-match single words. Fixes 232, 347. +4 TP, +0 FP.

C. Glued pre-split at extract_entities top (existing _GLOUED_DOSE_RE/_GLOUED_FREQ_RE on raw input): FlashText needs word boundaries, NER missed Simvastatin20mg/Linezolid10mg/Levofloxacin200mg/Dextromethorphan30mg/D20mg/500mgBD. Fixes 120, 206, 225, 226. +4 TP, +0 FP. (Aspirin/Salbutamol/Paracetamol 'misses' in eval log proven to be surface-match artifacts - drugs were extracted, rows already TP.)

D. Brand data: added warf->warfarin (fixes 425, 427), remapped deriphylline doxofylline->theophylline (old manual entry was wrong; fixes 428). 423 (omeprazole-warfarin) and 406 (levothyroxine-iron) pairs not in DrugBank under matchable names - unfixable via lookup. +3 TP, +0 FP.

Cleaned-set totals: AUC 0.6627->0.6904, Sens 0.8187->0.8889 (140->152 TP), TNR 0.5968 unchanged, F1 0.8333->0.8736, name-match 0.7627->0.7863. Remaining: 19 FN (2 no-DrugBank-pair, 17 Mild-scored incl. 247/410) + 25 FP (all real DrugBank pairs annotators call negative = label-policy question).


## Full GT audit vs DrugBank (4 subagents) + relabel

Per agreement DrugBank is ground truth: built mechanical evidence pack (every pair both directions, all names scored, descriptions), 4 subagents adjudicated 233 rows (same-drug verification, near-misses, intra-product exclusion, severity from description text via rubric; keyword heuristic given as cross-check only to avoid circularity).

Key catches: 405 Pheniprazine!=Pheniramine (Exp A reverse hit was wrong-drug -> row back to negative); 443 MgOH-Cetirizine exact 1.0 both ways (kept); Cefatrizine!=Cetirizine and Salicylic acid!=Acetylsalicylic acid decoys rejected; 107 Aspirin-Clopidogrel GT pair removed.

Overrides: 406 (iron+levothyroxine, only fumarate/gluconate in DB, no ascorbate) and 423 (omeprazole+warfarin, only (R)/(S)-warfarin at 0.80) forced negative - no exact pair; salt/stereoisomer extrapolation rejected, flagged for user.

Applied: 204/233 rows changed, positives 171->201, GT pairs Severe/VERY SEVERE (174) -> Moderate 185 + Severe 58 with per-pair DB descriptions. Eval now: AUC 0.6904->0.8644, Sens 0.8889->0.8756 (TP 152->176, FN 19->25 all Mild-scored), TNR 0.5968->0.9688 (TN 31, FP 1 = row 405), F1 0.8736->0.9312, zero n=0 FNs. Remaining gaps are now pure model issues: severity calibration (25 Mild-scored) + gate precision (405 Pheniprazine). Scratch (probes, chunks, audit scripts) removed.


## Severity calibration (safety-first max-rule)

All 25 FN-Mild rows: GT Moderate, custom model Mild with P(serious)~0.009 (dead certain), database text Moderate. Model maps vague PK boilerplate ('serum concentration can be increased') to Mild; DrugBank text warrants Moderate+. Fix in ddi.py _find_interaction: label/risk = max(custom, database) ordinally, risk=max(custom_risk, risk_score(database)); never downgrades. No test peeking (fixed rule, mirrors Gemini-path safety-first).

Eval: AUC 0.8644->0.9849, Sens 0.8756->1.0000 (FN 25->0), TNR 0.9688 unchanged (zero FP cost - no TN row has any found pair), F1 0.9312->0.9975. Binary confusion now TN 31 / FP 1 (405 Pheniprazine gate) / FN 0 / TP 201. 4-way exact 192/233; residual is 38 GT-Moderate rows the model calls Severe (overcall side, needs validation-set threshold work or retraining - not touched).


## Gemini pipeline refresh (3.8-flash, severity-only) + cost tracking

Keyword mode verified on 10 seeded samples, then switched severity+reasoning models to gemini-3.8-flash (ddi.py defaults + config). 3.8-flash rejects MINIMAL thinking (400 INVALID_ARGUMENT) -> severity thinking low, then medium per directive; reasoning thinking medium. Added gemini.reasoning_enabled toggle (currently false -> severity-only runs, reasoning falls back to DB description).

Cost tracking added (was absent): ddi.py records per-context calls/errors/input/output/thinking tokens/latency; get_gemini_stats() totals + est_cost_usd from config pricing (3.8-flash: 1.50/7.50 per 1M); evaluate writes outputs/{mode}/gemini_stats.json + prints + appends summary_stats.txt. Key find: thinking tokens bill as output and dominate - probe showed 256-711 thinking tokens for a 1-token answer; tracker now counts thoughts_token_count (earlier 6-vs-7 output mystery was a red herring: prompt demands a bare label, raw output is e.g. 'Severe', no JSON mode exists in this tree).

Missed-entity reporting made synonym-aware (reads entity name field, resolves GT brands to generics): match rate 0.7863->0.9941, list 109->3 true gaps (V1tam1n C deliberate OCR miss, Looz uncovered brand, Zincovit unresolvable multivitamin). Removed stale outputs/Ensemble/gemini_stats.json (dead schema; ensemble makes zero calls).

Full keyword refresh with the codebook prompt (233 rows, 250 severity calls, 0 errors): AUC 0.9712, Sens 1.0000, TNR 0.9118, F1 0.9925; cost 111K in / 237 out / 34K thinking tokens = USD 0.42 (low thinking cut thinking spend ~3× vs the medium run's USD 0.87). Keyword carries 3 FPs vs ensemble's 1. Known asymmetry: DB-keyword floor is Moderate so this path cannot predict Mild (Mild F1 0 on the 5 Mild pairs).


---

## Ensemble-only severity + per-class metrics (post keyword-max removal)

The safety-first `max(custom, database)` rule from the calibration section above fixed sensitivity but hid severity errors (38 Moderate→Severe) behind a perfect binary F1. Cut over in `ddi.py`: ensemble (existing XGB stacker) is now the sole severity signal, `database_severity` kept for audit only. Two decision rules remain, relabeled as permanent CODEBOOK POLICY (not temp fixes): genuinely-mild descriptions (`_is_genuinely_mild`: GI irritation only, minor-drug efficacy loss, excretion shift with NO serum consequence, decreased sedation/hypertension) stay Mild, all other ensemble-Mild maps to Moderate with risk bumped to 0.75; Severe needs `P(Sev)>=0.50` and margin `>=0.05` over Moderate (`SEVERE_MIN_PROB`/`SEVERE_MARGIN`).

`evaluate_ddi_system.py` now reports per-class headline metrics alongside binary: accuracy, macro/weighted F1, per-class F1 in `metrics_required`, `summary_stats.txt`, and console. Binary positive fixed to Moderate/Severe (`s>=2`, Mild negative) — load-bearing now that test carries Mild rows.


---

## Physician label program (train/val/test + 2000-description codebook)

Single-Gemini heuristic labels were the root drift (val: excretion-rate=Mild; test: same text=Moderate). Replaced with a physician-confirmed codebook: Severe = risks outweigh benefit, immediate/life-threatening; Moderate = avoid unless necessary, monitor/adjust, possible long-term harm; Mild = avoid only if alternative exists, no long-term effects.

* Train/val (`cache/severity_train_hq.json` 1600, `val_hq.json` 400): 5-way multi-agent audit, then criteria v2 normalization. Final: train 1345 Moderate / 225 Severe / 30 Mild; val 341 / 53 / 6. `soft_label` rebuilt as smoothed one-hot of the audited label (old constant-Moderate removed), `hard_label` fixed (was hardcoded 2). Originals in `cache/backup_pre_relabel/`, `cache/backup_pre_criteria_v2/`; audit trail in `severity_audit/`.
* 2000-description codebook: 2000 unique DrugBank descriptions (seed 20260923), 20×100 chunks, 2 independent agents per chunk: 1927/2000 agree (96.3%), 73 conflicts in `severity_audit/drugbank_sample/conflicts.jsonl`, resolved file at `resolved_2000.json`.
* Physician rulings applied everywhere: QTc/arrhythmia→Severe, bleeding→Severe, methemoglobinemia→Severe, neuromuscular blockade→Severe, thrombogenic→Severe, organ toxicity→Severe, PK/efficacy with no consequence→Mild, excretion WITH serum stated→Moderate, lone tachycardia/bradycardia→Moderate.
* Test (`test_prescriptions.json` 233 rows): 4 agents (2×117 + 2×116), 1 inter-agent conflict (Rx 224, resolved Mild), 22 pair diffs vs old GT adjudicated with the physician (9 stay-Severe wordings kept, 8 upgraded to Severe, 5 demoted to Mild). Pairs now Moderate 172 / Severe 66 / Mild 5. Backup at `severity_audit/rx_fix/test_prescriptions_backup.json`.


---

## Retrain on audited labels + current test metrics

Retrained (`train_ensemble.py` + `evaluate_ensemble.py`, existing stacker kept on full-train per accepted scope — no per-fold OOF retraining at this data size; recorded as known optimistic bias, not "modest impact").

Ensemble test eval after retrain + gates (`uv run evaluate_ddi_system.py --mode ensemble`): AUC 0.9806, Sens 0.9950, Spec 0.9706, binary F1 0.9950; accuracy 0.9185, macro 0.8942, weighted 0.9178; F1 No Interaction 0.9841 / Mild 0.80 / Moderate 0.9309 / Severe 0.8618; drug match 0.9941 (3 known gaps: deliberate V1tam1n OCR noise, uncovered Looz, multivitamin Zincovit).


---

## Extended 2-model search (running)

` tune_severity_search.py` extended for the relabeled distribution and set to retire MedSeverityNet (CNN + Transformer only): 96 configs/model (was 48), 25 screen epochs, top-12 through stratified 5-fold refine (was top-8/3-fold), widened spaces (embed 512, larger layers/filters/kernels, lr 1e-4, wd 3e-3, batch 32, smoothing 0.2, alpha 0.9) plus a focal-gamma dimension {0,1,2} for the 90%+-Moderate skew (gamma>0 disables smoothing; distillation unchanged). Stack grid widened (LogReg C + 0.01, XGB to 300 trees). Old-label artifacts archived to `outputs/severity_search_old_labels/`; screen resumed fresh. Cutover (ENSEMBLE_ARCHS, 9→6 stacker features, ddi.py loader) happens at promotion.

Temperature note (`fit_temps_val.py`): refit on held-out val NLL returns exactly 1.5 for all three models — temperature has nothing to fit; overconfidence lives in weights + stacker outputs. Temps stay 1.5 evidence-backed; stacker-level calibration is the future lever, not temperature.


---

## Accepted limitations (recorded, not fixed)

* Stacker trains on simulated (non truly-OOF) full-train logits — kept deliberately at n=1600; val/test numbers above should be read with that bias in mind.
* `severity_val_hq.json` picks checkpoints AND reports val metrics AND (now) would fit any calibration — 3-scalar reuse, negligible but disclosed.
* Train Severe rate (~14%) still trails test pair rate (~27%); QTc/bleeding/organ-toxicity promotion closed most of it, residual is label-prevalence difference, not model error.

---

## Stacker search → LogReg production (2026-09-24)

Stack stage on winners' calibrated OOF (5-fold outer CV): LogReg C=10 0.9022±0.064 vs C=0.1 0.9008±0.043 vs XGB-best (100 trees, depth 2) 0.8944±0.063. Gap C=10/C=0.1 is noise, so the codebase tie rule (≤0.002 → simpler) picks **C=0.1** — same mean, tighter std, linear. XGB out of production; stacker file renamed `cache/ensemble_stacker.pkl` (generic) with legacy fallback to the old xgb name. `evaluate_ensemble.py` now ships LogReg C=0.1 so a future refit can't silently reinstall XGB.

Test eval with LogReg stacker: AUC 0.9746, Sens 1.0000, Spec 0.9706, binary F1 0.9975; accuracy 0.9313, macro 0.9514, weighted 0.9308; Mild 1.0000 (5/5) / Moderate 0.9416 / Severe 0.8800; pair P/R/F1 0.960/0.988/0.974, sev-exact 0.9292; CIs accuracy [0.897,0.961], macro [0.917,0.972] (tight now that Mild predicts), binary [0.992,1.000]. XGB→LogReg deltas: accuracy +0.021, macro +0.064, Severe +0.037 — all favorable, none within-noise-negative.


---

## Extended search results + 2-model cutover (2026-09-24)

Screen (96 configs/model, 25 epochs, new labels): CNN baseline 0.777 → best random s042 0.934 (75/96 beat baseline); Tiny baseline 0.830 → best s071 0.924 but bimodal (mean 0.52, most configs collapse to chance on the imbalanced dev split). MedSeverityNet screen (36 trials, best 0.927) discarded with the model. One CUDA-OOM death on a 44M-param Transformer config led to batch-halving retry + OOM-evict guards in screen and refine — one oversized config can no longer kill a sweep.

Refine (top-12 + baseline × 5-fold): CNN s069 0.9184 / s085 0.9009 / s042 0.8980; Tiny s068 0.9005 / s050 0.8902 / s090 0.8871. Fold-1 consistently low across candidates (hard fold).

Top-5 shootout (full-1600 train, val-400 report): CNN **s080** val F1 0.8874 wins (focal 2.0, filters 512, embed 512); Tiny **s085** val F1 0.8763 wins (focal 2.0, char-filters 256). Both winners use focal_gamma 2.0 — the new dimension paid off for the 90%+-Moderate skew. OOF→val rank flips (s069 OOF 0.9184 → val 0.8242) confirm val is a selector, not a generalization claim.

Cutover: `ENSEMBLE_ARCHS` + `ENSEMBLE_NAMES` (CNN, Tiny) with fail-fast on retired names; `ddi.py` loader/classify, `train_ensemble.py`, `evaluate_ensemble.py`, `fit_temps_val.py` all 2-model; 6-feature XGB stacker refit (val Acc 0.9875 / macro 0.880 / AUC 0.842); seeds fixed (`train_ensemble --seed`, local-RNG subsample, DataLoader generator); artifact manifest at `outputs/Ensemble/artifact_manifest.json` (weight/stacker/temp/vocab hashes + git sha + label dists). Old 3-model artifacts in `cache/backup_2model_cutover/`.

2-model test eval: AUC 0.9751, Sens 0.9950, Spec 0.9706, binary F1 0.9950; accuracy 0.9099, macro 0.8878, weighted 0.9087; Mild 0.80 / Moderate 0.9242 / Severe 0.8430. New views: pair-level detection P/R/F1 0.960/0.988/0.974 with severity-exact 0.9125 on matched pairs (tp 240 / fp 10 / fn 3); paired bootstrap 95% CIs accuracy [0.871,0.944], macro [0.678,0.956] (wide — 5 Mild rows), binary [0.987,1.000]; decontamination 0/243 test descriptions verbatim in train or val (template-level resemblance remains, verbatim is zero). 3-model→2-model deltas (acc -0.009, macro -0.006, Severe -0.02) sit inside the CIs — dropping the BiLSTM cost essentially nothing.



---

## Learned codebook gate (2026-09-24)

Replaced ddi.py _is_genuinely_mild regexes + SEVERE_MIN_PROB/MARGIN with a
Severe-vs-rest LogisticRegression gate (learned_gate.py, 17-dim: 6 frozen
stacker numerics + 10 codebook keyword cues + n_words). Base nets and LogReg
stacker frozen; gate trained on val-400 only (Severe weight 10). Mild path
stays a deterministic allowlist (6 Mild val rows can't support learning).

Val 5-fold OOF: Sens-Severe 1.0, 0 overcalls (optimistic: threshold picked on
same OOF). Artifacts: outputs/learned_gate/ (gate.pkl, cv_table, weights,
50-row spot-check). Production cutover: config.yaml severity_gate=learned,
ddi.py default flipped, regex kept as fallback/ablation.

## RxBench-India + Rx-Noise (2026-09-24)

build_rxbench.py: rxbench-india.json (88/259 generics brand-substituted,
138/233 Rx changed, GT identical) + rxbench-noise.json L1/L2/L3 (fixed seed,
GT-preservation: interacting tokens never dropped). Eval, no retraining:

clean=brand=L3 (0.9941/0.9975/SevRec 0.846) > L2 (0.9627/0.9950/0.846) >>
L1 cliff (0.7706/0.7382/SevRec 0.308). Brand costs nothing (linker robust;
severity grades unchanged descriptions, so brand is a linking test). L1
char-typos break linking itself: next robustness work is candidate
generation, not severity modeling. Learned gate holds +0.03 Severe recall on
every split where pairs survive (L2/L3/clean), ~+0.015 on L1.

Also fixed: run_rxbench_eval.py defined ResilientEvaluator but instantiated
plain DDIEvaluator (WinError 5 race); chart_2 now Gemini-vs-local only, 9
metrics incl. Mild F1, 233-Rx title.


---

## Update 3 full inventory (since f40febe, 2026-09-15)

Everything below shipped after the package restructure. Status of each item:
KEPT = in production now. PARKED = code kept, switched off, documented reason.
SUPERSEDED = replaced by something better. OPEN = known problem, not solved.

### KEPT improvements

* Package restructure + paths (f40febe, 54a3249, 5cc3bd6): backend is a
  package, UI goes through pipeline.process_input, all artifacts under
  outputs/. Granite embeddings adopted here (the BioLORD question came later).
* Retrieval hardening, all kept: fuzzy safety net + mention cleaning + combo
  split (92f0bad); 35k-synonym max-pool retrieval (4ea3a74); short-key gate +
  stopword filter fixing no->nitric oxide (4f25365); phantom word-boundary
  filter + detection confidence floor 0.90 + s>=2 binary def (fb7fd93);
  alias purge 2011->1819 + DrugBank junk filter + brand sidecar (63edd35);
  fix exps A-D: reverse lookup, segment guard, multi-word grounding, glued
  pre-split, warf/deriphylline brands (a9d315a, +12 TP / +0 FP).
  Drug-name match rate 0.79 -> 0.99; 3 true misses remain (V1tam1n noise,
  Looz, Zincovit).
* Test-set cleanup 296->233 rows (663bd3a) + full DrugBank GT audit via 4
  subagents, 204 rows (0217484): AUC 0.69->0.86, TNR 0.60->0.97.
* Gemini track: 3.5 -> 3.1-lite -> 3.8-flash severity-only, thinking
  minimal->medium->low, reasoning toggleable and off by default (9198424),
  per-context usage/cost tracking (b67c23f, 647c06c). End state: AUC 0.9712,
  $0.42/refresh, cannot predict Mild by construction.
* Physician relabel + retrain + 2-model search + LogReg stacker + learned
  gate + RxBench (da798f8 .. 63732e9): covered in sections above. Production
  now acc 0.9356, macro 0.9549, Severe rec 0.877, binary F1 0.9975.
* Eval harness upgrades kept: per-class metrics, pair-level detection +
  severity-exact, paired bootstrap CIs, verbatim decontamination report,
  shuffle/seed/resume checkpoints, synonym-aware miss reporting (f0f471e),
  ResilientEvaluator file-lock fix.

### PARKED (tried, switched off, code retained)

* BioLORD-2023-M encoder (2a8b06d): won the spike 94.4% vs granite 91.7%,
  harness in eval_retrieval_spike.py — swap never applied, production still
  granite. Cheapest retrieval upgrade available if needed.
* TF-IDF fusion (2c2eb4f, disabled in 49efd68): char-ngram index + linear
  dense/tfidf/fuzzy fusion built, then switched off
  (extraction.py _use_tfidf=False) — fused-score scale mismatch created
  phantom entities faster than it fixed glued tokens. Glued-token segmenter
  from the same commit is kept.
* FlashText dual processor (4f25365, reverted in 49efd68): case-sensitive
  short-key path caused more misses than it fixed; single processor + gate
  kept instead.

### SUPERSEDED (replaced, old path kept only as fallback or archive)

* Max-rule calibration (90f508f, Sens 0.88->1.00 at the time) -> ensemble-only
  severity (da798f8) -> learned gate (e09ef0a). Each step kept the gains and
  removed a heuristic.
* 3-model ensemble + XGB stacker (6731402 era) -> 2-model + LogReg C=0.1
  (f674da0). Old weights in cache/backup_2model_cutover/; old-label search
  logs in outputs/severity_search_old_labels/.
* Regex severity gate -> learned gate; regex retained behind
  SEVERITY_GATE=regex.
* Gemini 3.5 / 3.1-lite defaults, medium thinking, reasoning-on: all replaced
  by 3.8-flash / low thinking / reasoning-off.

### OPEN (not fixed, with owner/next step)

* L1 char-typo cliff (Severe rec 0.31): needs typo-tolerant candidate
  generation before NER. Next robustness item.
* Stacker simulated-OOF bias; val-400 triple use (checkpoints + reporting +
  gate fit); train/test Severe prevalence gap. All disclosed, none blocking.
* BioLORD swap and TF-IDF rescale-and-retry are the two cheapest parked
  levers if retrieval needs another point.
* Backlog: TODO.md / TODO_PRIORITY.md (added 1452e29, untouched since).
