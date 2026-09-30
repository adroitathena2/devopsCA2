# TODO Priority Triage

5 subagents used (≤30 points each): Bugs 1-30, 31-60, 61-78; Assumptions 1-32, 33-64.

Rubric: **High** = silent wrong clinical output / safety risk / crash-race / data loss / RCE / leakage invalidating safety claims. **Medium** = degraded function / misleading metrics / partial failure. **Low** = cosmetic / dead-code / edge.

Sorted High → Medium → Low within each section, stable by original order.

## Bugs: 43 High / 28 Medium / 7 Low

| Rank | Orig # | Location | Severity |
|---|---|---|---|
| 1 | 2 | `local_llm.py:54-55` URLError as success text | High |
| 2 | 3 | `clinical_copilot_ui.py:465,487` `finished` overrides `QThread.finished` | High |
| 3 | 5 | `clinical_copilot_ui.py:863,882,901` no `wait()/deleteLater`, no `closeEvent` | High |
| 4 | 6 | `clinical_copilot_ui.py:844-845` concurrent OCR+analysis, blocking `singleShot` | High |
| 5 | 9 | `tfidf_index.py:135-140` unanimous `_norm`→0.0 rejected | High |
| 6 | 10 | `tfidf_index.py:163` fused ceiling <0.62 accepts nothing | High |
| 7 | 13 | `tfidf_index.py:195-205` first-fit glued-token split | High |
| 8 | 14 | `embeddings.py:270-272` fallback returns first-k | High |
| 9 | 15 | `embeddings.py:139` difflib on cosine channel | High |
| 10 | 16 | `embeddings.py:225-234` alias length mismatch→`[]` | High |
| 11 | 17 | `embeddings.py:245` unbounded cache, thread-unsafe, dtype skew, margin bypass | High |
| 12 | 19 | `severity.py:47-49` unknowns→No Interaction + crash | High |
| 13 | 21 | `build_indian_brands.py:69-81` aspirin cycle flips canonical | High |
| 14 | 22 | `build_indian_brands.py:355,364` validation no-op, unverified `priority=True` | High |
| 15 | 26 | `ddi.py:658` `.clear()` caller mutation + name-only index | High |
| 16 | 27 | `ddi.py:650-655,819` IDs as name keys fuzzy-matched | High |
| 17 | 28 | `ddi.py:726-778` first-match wins, dual 0.85, O(n²) | High |
| 18 | 30 | `ddi.py:775,438-478,384-430` silent Moderate default, hardcoded indices | High |
| 19 | 35 | `pipeline.py:226-228/144-154` cache/raw diverge, identity map, wrong gate | High |
| 20 | 36 | `pipeline.py:127,133,135,165` unvalidated `.numpy()` GPU fail | High |
| 21 | 37 | `pipeline.py:299-301` `np.float64`, double mean, confidence conflation | High |
| 22 | 39 | `pipeline.py:239-316` never fails + phase contradiction | High |
| 23 | 40 | `ner.py:68,89,154,233-249` hardcoded GLiNER/device/labels/regex/schema/OOM | High |
| 24 | 41 | `confidence.py:58,88-115` `1-std`, broken calibrate, `1-mean(var)` | High |
| 25 | 43 | `evaluate_ddi_system.py:168-169` `s>=1` Mild as positive | High |
| 26 | 45 | `evaluate_ddi_system.py:146-162` substring-before-synonym inflates rate | High |
| 27 | 46 | `evaluate_ddi_system.py:102-109` max-severity-only scoring | High |
| 28 | 47 | `evaluate_ensemble.py:88-92` val fit+score double-dip, leaked OOF | High |
| 29 | 48 | `evaluate_ensemble.py:82,197-201` temps==`[1.5]` + overwrites prod | High |
| 30 | 49 | `evaluate_ensemble.py:26,51-61` shuffle+sort misalign, unstable calibration | High |
| 31 | 53 | `tune_gemini_model.py:44-47` 30-row/18-config same-file tune+eval | High |
| 32 | 54 | `tune_severity_search.py:522-526` unpaired bootstrap, val leak | High |
| 33 | 55 | `train_ensemble.py:28-33` no seed, double-smooth, constant soft targets | High |
| 34 | 56 | `custom_severity_model.py:65,81,95-117` global seed, zero loss, sort permute, OOB | High |
| 35 | 57 | `add_typos.py:11-34` import-time test-set rewrite | High |
| 36 | 58 | `relabel_chunks_gemini.py:59-60` non-atomic + skip-if-exists | High |
| 37 | 61 | `build_indian_brands.py:284,592-594` wrong vote, 3x weight, QA no-block, deadlock | High |
| 38 | 62 | `brands.py:20,25-76` over-broad `^ns`, artifacts pass, type instability | High |
| 39 | 63 | `brands.py` lexicon→ingredients absent from index, silent skip/mismatch | High |
| 40 | 64 | `extraction.py:107-115` FlashText only `canon[0]` drops agent | High |
| 41 | 67 | `extraction.py:556-712` O(E*35k), excludes B12/D3, SymSpell recall loss | High |
| 42 | 75 | `data.py:40,48-63` whole-file load OOM, silent `{}` empty pipeline | High |
| 43 | 77 | `tfidf_index.py:38,50,95-117` `min_df=2`, pickle RCE, unchecked hash | High |
| 44 | 1 | `extraction.py:551-552` confidence_map misalignment | Medium |
| 45 | 4 | `clinical_copilot_ui.py:469-477` `use_gemini` no-op | Medium |
| 46 | 7 | `clinical_copilot_ui.py:791-802` generic WARNING downgrade + `os.chdir` | Medium |
| 47 | 8 | `clinical_copilot_ui.py:289,358` Severe default / None badge | Medium |
| 48 | 11 | `tfidf_index.py:165` `ranked[0][0]!=ranked[1][0]` always true | Medium |
| 49 | 12 | `tfidf_index.py:136` 0.0 fillers vs negative cosine | Medium |
| 50 | 18 | `embeddings.py:295-296` per-query full encode | Medium |
| 51 | 20 | `merge_labels.py:30` unknowns→Mild(0) | Medium |
| 52 | 23 | `generate_comparative_charts.py:188-190` `input_tokens` vs `total_*` | Medium |
| 53 | 24 | `generate_comparative_charts.py:327-336` %×100 + hardcoded fallbacks | Medium |
| 54 | 29 | `ddi.py:480-508` upward-biased keywords + `max()` overcall | Medium |
| 55 | 33 | `ddi.py:117-119` missing key silent fallback | Medium |
| 56 | 38 | `pipeline.py:355-365` OCR swallows errors, `strip` assumes str | Medium |
| 57 | 42 | `pipeline.py:288` cross-drug spread as reliability | Medium |
| 58 | 44 | `evaluate_ddi_system.py:171-183` mixed ROC scales | Medium |
| 59 | 50 | `tune_severity_search.py:236-239` bare except→0.5 / →False | Medium |
| 60 | 51 | `evaluate_ddi_system.py:104-106,250` no CIs, Gemini latency | Medium |
| 61 | 59 | `run_embedding_comparison.py:67-81` config rewrite no `try/finally` | Medium |
| 62 | 60 | `build_cache.py:235,329,365-368` `list(set())`, no seed/atomic/version | Medium |
| 63 | 65 | `extraction.py:414-426` duplicates double-resolve | Medium |
| 64 | 66 | `extraction.py:194-212` order-dependent clean, `500mg` passes, first-not-max | Medium |
| 65 | 68 | `eval_retrieval_spike.py:37-79` 37 queries, lenient, drift, survivorship | Medium |
| 66 | 70 | `cli.py:27,37-48` `name` outside try crashes run | Medium |
| 67 | 72 | `plots.py:78-83` `%.1f%%`/`ylim` wrong for 0-1 | Medium |
| 68 | 73 | `test_backend.py:4-211` global leakage, weak oracle | Medium |
| 69 | 74 | `test_backend_severity.py:1-146` print-only, 3 cases, chdir | Medium |
| 70 | 76 | `config.py:84-103` None-collapse, mutable leak, divergent defaults | Medium |
| 71 | 78 | `evaluate_ddi_system.py:53-55` post-load config mutation | Medium |
| 72 | 25 | `ddi.py:601` scrub omits Moderate | Low |
| 73 | 31 | `ddi.py:797-802` tie logged as mismatch | Low |
| 74 | 32 | `ddi.py:555` `severity` shadows module | Low |
| 75 | 34 | `ddi.py:302-303` unreachable cleanup | Low |
| 76 | 52 | `evaluate_ensemble.py:52-59` labels only if `MedSeverityNet` | Low |
| 77 | 69 | `eval_retrieval_spike.py:155` dummy leftover | Low |
| 78 | 71 | `local_llm.py:10,23,50-55` hardcoded, no timeout, dead uncalled | Low |

## Incorrect assumptions: 31 High / 22 Medium / 11 Low

| Rank | Orig # | Assumption | Severity |
|---|---|---|---|
| 1 | 1 | error strings are model output | High |
| 2 | 2 | `confidence_scores` stays parallel after filter | High |
| 3 | 5 | min-max usable + 1 signal reaches 0.62 | High |
| 4 | 7 | unknown→No Interaction fail-safe | High |
| 5 | 10 | RISK constants == P(Mod)+P(Sev) scale | High |
| 6 | 13 | tuning Gemini on test unbiased | High |
| 7 | 14 | val-fit weights can score val | High |
| 8 | 17 | zero vector valid soft label / `get(i,1)` sane | High |
| 9 | 18 | `collate_fn` order matches labels | High |
| 10 | 19 | difflib calibrated vs cosine / fallback most-similar | High |
| 11 | 21 | `1-std` is confidence / spread is reliability | High |
| 12 | 25 | first 2-piece split correct / `max_pieces` generalizes | High |
| 13 | 26 | `^ns` narrow / concentration blocked | High |
| 14 | 27 | `normalize_ingredient` canonical | High |
| 15 | 31 | button disable = mutex | High |
| 16 | 34 | `add_typos` import side-effect-free | High |
| 17 | 35 | skip-if-exists ⇒ valid chunk | High |
| 18 | 36 | `drugbank_json` safe to clear | High |
| 19 | 37 | cache/raw + defaults equivalent | High |
| 20 | 41 | `hasattr/getattr` guards needed / half-init safe | High |
| 21 | 43 | first DrugBank match is right | High |
| 22 | 48 | chunk IDs stable | High |
| 23 | 50 | 4-class soft idx 0 + per-row soft meaningful | High |
| 24 | 52 | non-OOF impact modest | High |
| 25 | 53 | `val>0` is deployed task / 0.99 AUC transfers | High |
| 26 | 54 | `severity_val_hq.json` pure validation | High |
| 27 | 55 | detection metrics measure system | High |
| 28 | 56 | brand ingredients self-verified | High |
| 29 | 60 | max-only fusion safe | High |
| 30 | 61 | single Gemini labels = ground truth | High |
| 31 | 62 | chunks/test decontaminated | High |
| 32 | 3 | `use_gemini` honored | Medium |
| 33 | 4 | `finished` free on QThread | Medium |
| 34 | 6 | `ranked[0][0]!=ranked[1][0]` can be false | Medium |
| 35 | 8 | `s>=1` = Moderate/Severe | Medium |
| 36 | 9 | RISK ladder is ROC probability | Medium |
| 37 | 11 | max severity = pair correctness | Medium |
| 38 | 12 | substring is synonym-aware | Medium |
| 39 | 15 | bootstrap paired | Medium |
| 40 | 16 | runs reproducible / Dataset `seed` harmless | Medium |
| 41 | 24 | fusion active | Medium |
| 42 | 28 | `return c` validation filters | Medium |
| 43 | 30 | `singleShot` makes async | Medium |
| 44 | 32 | worker ref droppable on slot | Medium |
| 45 | 33 | `config.yaml` always restored | Medium |
| 46 | 39 | regex 0.5 is measurement | Medium |
| 47 | 40 | `urlopen` no timeout / only URLError | Medium |
| 48 | 51 | train-temp prevents leakage | Medium |
| 49 | 57 | `eval_retrieval_spike` parity | Medium |
| 50 | 58 | non-stratified shuffle + in-place safe | Medium |
| 51 | 59 | `TRANSFORMERS_AVAILABLE` gates / `safe_load`≠None / throttle suffices | Medium |
| 52 | 63 | relative paths resolve same | Medium |
| 53 | 64 | `pyproject` lower-bounds + cu128 reproducible | Medium |
| 54 | 20 | `calibrate_confidence` is temp scaling / `ci_95` are CIs | Low |
| 55 | 22 | MC Dropout/ensemble implemented | Low |
| 56 | 23 | `content_hash` detects staleness | Low |
| 57 | 29 | `gemini_stats` keys `input_tokens/calls` | Low |
| 58 | 38 | `VERY SEVERE` reachable | Low |
| 59 | 42 | `if self.config:` meaningful | Low |
| 60 | 44 | `.title()` recovers casing | Low |
| 61 | 45 | Phase 2 does work | Low |
| 62 | 46 | `print`==`logger` | Low |
| 63 | 47 | 100 rows | Low |
| 64 | 49 | `generic_scores`/`max_pieces` effective | Low |
