"""
Lightweight multi-model retrieval spike for idea #8.

Compares embedding models on drug-name retrieval (top-1 accuracy by category)
WITHOUT rebuilding the full cache — encodes the existing generic + alias
strings from cache/medicine_data.json + cache/alias_data.json and scores
queries with the same max-pool + margin logic as production.

Usage:
    uv run python eval_retrieval_spike.py                       # all models
    uv run python eval_retrieval_spike.py --models biolord,granite
    uv run python eval_retrieval_spike.py --ablation             # override-free 5
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MODELS = [
    {"name": "ibm-granite/granite-embedding-278m-multilingual", "label": "granite-278m"},
    {"name": "FremyCompany/BioLORD-2023-C", "label": "BioLORD-2023-C"},
    {"name": "FremyCompany/BioLORD-2023", "label": "BioLORD-2023"},
    {"name": "FremyCompany/BioLORD-2023-M", "label": "BioLORD-2023-M"},
    {"name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "label": "SapBERT"},
    {"name": "GanjinZero/coder_eng", "label": "CODER-eng"},
]

POOL_MARGIN = 0.03
ACCEPT_THRESHOLD = 0.85

# (query, expected_canonical_or_None, category)
# expected=None means "should abstain / no match"
TEST_QUERIES = [
    # Raw brand aliases
    ("dolo", "acetaminophen", "brand"),
    ("crocin", "acetaminophen", "brand"),
    ("augmentin", "amoxicillin", "brand"),
    ("telma-h", "telmisartan", "brand"),
    ("telmisartan h", "telmisartan", "brand"),
    ("ramistar-am", "ramipril", "brand"),
    ("starpress xl", "metoprolol", "brand"),
    ("ecosprin", "acetylsalicylic acid", "brand"),
    ("glycomet", "metformin", "brand"),
    ("storvas", "atorvastatin", "brand"),
    ("azithral", "azithromycin", "brand"),
    ("pan-d", "pantoprazole", "brand"),
    ("amoxil", "amoxicillin", "brand"),
    ("pcm", "acetaminophen", "brand"),
    # Typo'd aliases
    ("paracetmol", "acetaminophen", "typo"),
    ("amoxcillin", "amoxicillin", "typo"),
    ("rantidine", "ranitidine", "typo"),
    ("metforminn", "metformin", "typo"),
    ("atorvastatn", "atorvastatin", "typo"),
    ("clopidegrel", "clopidogrel", "typo"),
    # Generics
    ("metformin", "metformin", "generic"),
    ("atorvastatin", "atorvastatin", "generic"),
    ("warfarin", "warfarin", "generic"),
    ("amoxicillin", "amoxicillin", "generic"),
    ("paracetamol", "acetaminophen", "generic"),
    ("aspirin", "acetylsalicylic acid", "generic"),
    ("salbutamol", "albuterol", "generic"),
    ("hydrochlorothiazide", "hydrochlorothiazide", "generic"),
    # Multi-word / combo
    ("amoxicillin clavulanate", "amoxicillin", "multiword"),
    ("acetylsalicylic acid", "acetylsalicylic acid", "multiword"),
    ("amlodipine besylate", "amlodipine", "multiword"),
    ("metformin hcl", "metformin", "multiword"),
    # Negative / should abstain
    ("no known allergies", None, "negative"),
    ("xyzqqq", None, "negative"),
    ("take with food", None, "negative"),
    ("patient chart", None, "negative"),
]

# Ship-it ablation: 5 cases that must resolve WITHOUT manual overrides
ABLATION_QUERIES = [
    ("salbutamol", "albuterol", "ablation"),
    ("hydrochlorothiazide", "hydrochlorothiazide", "ablation"),
    ("hctz", "hydrochlorothiazide", "ablation"),
    ("paracetamol", "acetaminophen", "ablation"),
    ("pcm", "acetaminophen", "ablation"),
    ("amoxil", "amoxicillin", "ablation"),
]


def load_corpus():
    with open("cache/medicine_data.json", encoding="utf-8") as f:
        generics = json.load(f)["medicine_names"]
    with open("cache/alias_data.json", encoding="utf-8") as f:
        ad = json.load(f)
    return generics, ad["alias_texts"], ad["alias_concepts"]


def encode_texts(model, texts, batch_size=64, mean_pool=False):
    """Encode texts, optionally with mean-pooling (for SapBERT)."""
    if mean_pool:
        out = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
            output_value="token_embeddings",
        )
        pooled = np.zeros((len(texts), out[0].shape[-1]), dtype="float32")
        for i, emb in enumerate(out):
            emb_np = emb.cpu().numpy() if hasattr(emb, 'cpu') else np.asarray(emb)
            pooled[i] = emb_np.mean(axis=0)
        return pooled
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")


def find_most_similar_concepts(
    query_emb, generic_texts, generic_embs, alias_texts, alias_embs, alias_concepts, top_k=3
):
    """Max-pool retrieval with margin rule (mirrors embeddings.py logic)."""
    if generic_embs is not None:
        sims = generic_embs @ query_emb / (
            np.linalg.norm(generic_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        generic_scores = {}
        for i, g in enumerate(generic_texts):
            s = float(sims[i])
            if s > generic_scores.get(g, -2.0):
                generic_scores[g] = s
    else:
        generic_scores = {}

    concept_scores = dict(generic_scores)
    if alias_embs is not None and alias_texts:
        sims = alias_embs @ query_emb / (
            np.linalg.norm(alias_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        for j, concepts in enumerate(alias_concepts):
            s = float(sims[j])
            for c in concepts:
                if s > concept_scores.get(c, -2.0):
                    concept_scores[c] = s

    ranked = lambda scores: sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    pooled = ranked(concept_scores)

    if generic_scores and pooled:
        query_lower = " ".join(str(query_emb.shape))  # dummy, not used for exact-alias bypass
        gen_best = max(generic_scores.items(), key=lambda kv: kv[1])
        if pooled[0][0] != gen_best[0] and pooled[0][1] - gen_best[1] < POOL_MARGIN:
            return ranked(generic_scores)
    return pooled


def evaluate_model(entry, generics, alias_texts, alias_concepts, queries):
    """Encode corpus + queries with one model, return per-category top-1 accuracy."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean_pool = "sapbert" in entry["name"].lower()

    print(f"\n{'=' * 60}")
    print(f"Loading {entry['label']} ({entry['name']}) on {device}...")
    t0 = time.time()
    try:
        model = SentenceTransformer(entry["name"], device=device)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return {"label": entry["label"], "status": "load_failed", "error": str(e)}
    load_time = time.time() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"  Loaded in {load_time:.1f}s  dim={dim}")

    # Encode corpus
    t0 = time.time()
    print(f"  Encoding {len(generics)} generics + {len(alias_texts)} aliases...")
    gen_embs = encode_texts(model, generics, batch_size=64, mean_pool=mean_pool)
    alias_embs = encode_texts(model, alias_texts, batch_size=64, mean_pool=mean_pool)
    encode_time = time.time() - t0
    print(f"  Corpus encoded in {encode_time:.1f}s")

    # Evaluate queries
    results = {}
    for query, expected, category in queries:
        q_emb = encode_texts(model, [query], mean_pool=mean_pool)[0]
        matches = find_most_similar_concepts(
            q_emb, generics, gen_embs, alias_texts, alias_embs, alias_concepts, top_k=3
        )

        predicted = matches[0][0] if matches and matches[0][1] >= ACCEPT_THRESHOLD else None
        score = matches[0][1] if matches else 0.0

        if expected is None:
            correct = predicted is None
        else:
            correct = predicted is not None and (
                expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
            )

        results[query] = {
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "correct": correct,
            "category": category,
        }

    # Category accuracy
    cats = {}
    for q, r in results.items():
        cats.setdefault(r["category"], []).append(r["correct"])
    cat_acc = {c: sum(v) / len(v) for c, v in cats.items()}

    return {
        "label": entry["label"],
        "status": "ok",
        "load_time": load_time,
        "encode_time": encode_time,
        "dim": dim,
        "category_accuracy": cat_acc,
        "overall": sum(r["correct"] for r in results.values()) / len(results),
        "results": results,
    }


def run_ablation(generics, alias_texts, alias_concepts):
    """5-case override-free check: can representation alone resolve these?"""
    print(f"\n{'=' * 60}")
    print("ABLATION (override-free): salbutamol, HCTZ, paracetamol, PCM, amoxil")
    print(f"{'=' * 60}")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODELS[0]["name"], device=device)
    gen_embs = encode_texts(model, generics, batch_size=64)
    alias_embs = encode_texts(model, alias_texts, batch_size=64)

    all_ok = True
    for query, expected, _ in ABLATION_QUERIES:
        q_emb = encode_texts(model, [query])[0]
        matches = find_most_similar_concepts(
            q_emb, generics, gen_embs, alias_texts, alias_embs, alias_concepts, top_k=3
        )
        predicted = matches[0][0] if matches and matches[0][1] >= ACCEPT_THRESHOLD else None
        score = matches[0][1] if matches else 0.0
        ok = predicted is not None and (
            expected.lower() in predicted.lower() or predicted.lower() in expected.lower()
        )
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {query!r} -> {predicted!r} (score={score:.3f}, expected={expected!r})")

    print(f"\n  Ablation: {'5/5 PASS' if all_ok else 'SOME FAILURES'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Multi-model retrieval spike")
    parser.add_argument("--models", default="", help="Comma-separated substrings to filter models")
    parser.add_argument("--ablation", action="store_true", help="Run override-free ablation only")
    args = parser.parse_args()

    generics, alias_texts, alias_concepts = load_corpus()
    queries = TEST_QUERIES if not args.ablation else ABLATION_QUERIES

    if args.ablation:
        run_ablation(generics, alias_texts, alias_concepts)
        return

    selected = MODELS
    if args.models:
        filters = [f.strip().lower() for f in args.models.split(",") if f.strip()]
        selected = [m for m in MODELS if any(f in m["label"].lower() or f in m["name"].lower() for f in filters)]

    all_results = []
    out_path = Path("outputs/retrieval_spike_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Load any prior results (for incremental runs)
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                prior = json.load(f)
            done = {r.get("label") for r in prior if r.get("status") == "ok"}
            all_results = [r for r in prior if r.get("label") in done]
            selected = [e for e in selected if e["label"] not in done]
            if not selected:
                print("All selected models already evaluated.")
        except Exception:
            pass
    for entry in selected:
        r = evaluate_model(entry, generics, alias_texts, alias_concepts, queries)
        all_results.append(r)
        # Incremental save so crashes don't lose completed results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    header = f"{'Model':<20} {'Status':<12} {'Overall':>8} {'brand':>7} {'typo':>7} {'generic':>8} {'multi':>7} {'neg':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r["status"] != "ok":
            print(f"{r['label']:<20} {r['status']:<12}")
            continue
        ca = r["category_accuracy"]
        print(
            f"{r['label']:<20} {'ok':<12} {r['overall']:>7.1%} "
            f"{ca.get('brand', 0):>7.1%} {ca.get('typo', 0):>7.1%} "
            f"{ca.get('generic', 0):>8.1%} {ca.get('multiword', 0):>7.1%} "
            f"{ca.get('negative', 0):>7.1%}"
        )

    # Save detailed results
    out_path = Path("outputs/retrieval_spike_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
