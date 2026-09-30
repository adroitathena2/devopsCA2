"""
Build Cache Script for Clinical Copilot
========================================
Pre-computes medicine name embeddings on GPU (with CPU fallback) and caches
all heavy data so the UI loads in seconds instead of minutes.

Usage:
    python build_cache.py

Outputs (in cache/ directory):
    - medicine_embeddings.safetensors   ([n_generics × D] float32 tensor)
    - alias_embeddings.safetensors      ([n_aliases × D] float32 tensor (idea #2))
    - alias_data.json                   (alias texts + per-alias concepts (idea #2))
    - medicine_data.json                (ordered names + cleaned CSV records)
    - drugbank_index.json.gz            (pre-built DrugBank lookup index)

Pass --aliases-only to skip the generic-embedding rebuild and only
(re)build the alias files from the cached names + synonyms.
"""

import json
import gzip
import re
import time
import os
from pathlib import Path

# Junk-alias patterns for DrugBank synonyms (IUPAC/DNA truncations, CAS
# numbers, CJK strings). Long/junk aliases get their own 768-d vectors and
# pollute max-pooled retrieval — nobody types them in a prescription.
_CAS_RE = re.compile(r'\d{5,}-\d{2}-\d')
_NON_ALPHA_RE = re.compile(r'^[^a-z]+$')
_JUNK_RES = [re.compile(r'^ns(-|%| )?'), re.compile(r'[%\[\(/:+]'),
            re.compile(r'stayhappi')]


def _is_junk_alias(a: str) -> bool:
    t = (a or '').strip().lower()
    if not t or len(t) <= 1 or len(t) > 60:
        return True
    if any(p.search(t) for p in _JUNK_RES):
        return True
    if _CAS_RE.search(t) or _NON_ALPHA_RE.match(t):
        return True
    return False

import torch
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

# Reuse the backend's data loader and config
from clinical_copilot.brands import apply_brand_lexicon
from clinical_copilot_backend import ConfigLoader, MedicineDataLoader, resolve_torch_dtype


def build_cache(config_path: str = "config.yaml", aliases_only: bool = False,
                tfidf_only: bool = False):
    overall_start = time.time()

    # ── Load config ──────────────────────────────────────────────────
    config = ConfigLoader(config_path)
    cache_dir = Path(config.get('cache', 'directory', default='cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = cache_dir / config.get(
        'cache', 'embeddings_file', default='medicine_embeddings.safetensors')
    medicine_data_file = cache_dir / config.get(
        'cache', 'medicine_data_file', default='medicine_data.json')
    drugbank_index_file = cache_dir / config.get(
        'cache', 'drugbank_index_file', default='drugbank_index.json.gz')
    medicine_synonyms_file = cache_dir / config.get(
        'cache', 'medicine_synonyms_file', default='medicine_synonyms.json')
    alias_embeddings_file = cache_dir / config.get(
        'cache', 'alias_embeddings_file', default='alias_embeddings.safetensors')
    alias_data_file = cache_dir / config.get(
        'cache', 'alias_data_file', default='alias_data.json')

    model_name = config.get(
        'models', 'embeddings', 'model_name',
        default='sentence-transformers/all-MiniLM-L6-v2')

    # ── --tfidf-only: rebuild just the sparse index from cached texts ──
    if tfidf_only:
        print("[tfidf-only] Rebuilding TF-IDF index from cached texts...")
        t0 = time.time()
        med_data_path = cache_dir / 'medicine_data.json'
        alias_data_path = cache_dir / 'alias_data.json'
        if not med_data_path.exists():
            print(f"ERROR: {med_data_path} not found. Run a full build first.")
            return
        with open(med_data_path, encoding='utf-8') as f:
            medicine_names = json.load(f)['medicine_names']
        alias_texts = []
        if alias_data_path.exists():
            with open(alias_data_path, encoding='utf-8') as f:
                alias_texts = json.load(f).get('alias_texts', [])
        from clinical_copilot.tfidf_index import build_tfidf_index, save_tfidf_index
        tfidf_corpus = list(medicine_names) + alias_texts
        vec, mat = build_tfidf_index(tfidf_corpus)
        save_tfidf_index(vec, mat, str(cache_dir))
        print(f"       Done in {time.time() - t0:.1f}s")
        return

    # ── Step 1: Load JSON (DrugBank) ─────────────────────────────────
    print("[1/4] Loading DrugBank data to extract medicine names...")
    t0 = time.time()
    loader = MedicineDataLoader(config)
    
    json_data = loader.load_json_data()

    drugbank = json_data.get('{http://www.drugbank.ca}drugbank', {})
    drugs = drugbank.get('{http://www.drugbank.ca}drug', [])

    drug_index = {}
    medicine_names_set = set()
    synonym_map = {}
    for drug in drugs:
        name = drug.get('{http://www.drugbank.ca}name', '')
        canonical_name = name.lower() if name else ""

        def extract_list(raw_data, child_key):
            if not raw_data: return []
            items = []
            if isinstance(raw_data, list):
                # Handle cases where raw_data is directly a list of strings
                if len(raw_data) > 0 and isinstance(raw_data[0], str):
                    return raw_data
                for item in raw_data:
                    if isinstance(item, dict):
                        val = item.get(child_key, [])
                        if isinstance(val, list): items.extend(val)
                        else: items.append(val)
            elif isinstance(raw_data, dict):
                val = raw_data.get(child_key, [])
                if isinstance(val, list): items.extend(val)
                else: items.append(val)
            return items

        # Extract synonyms, international brands, and products
        aliases = set()

        synonyms_raw = drug.get('{http://www.drugbank.ca}synonyms')
        s_list = extract_list(synonyms_raw, '{http://www.drugbank.ca}synonym')
        for s in s_list:
            if isinstance(s, dict):
                alias = s.get('#text', s.get('text', ''))
            else:
                alias = str(s)
            if alias: aliases.add(alias.lower())

        brands_raw = drug.get('{http://www.drugbank.ca}international-brands')
        b_list = extract_list(brands_raw, '{http://www.drugbank.ca}international-brand')
        for b in b_list:
            if isinstance(b, dict):
                alias = b.get('{http://www.drugbank.ca}name', b.get('name', ''))
            else:
                alias = str(b)
            if alias: aliases.add(alias.lower())

        products_raw = drug.get('{http://www.drugbank.ca}products')
        p_list = extract_list(products_raw, '{http://www.drugbank.ca}product')
        for p in p_list:
            if isinstance(p, dict):
                alias = p.get('{http://www.drugbank.ca}name', p.get('name', ''))
                if alias: aliases.add(alias.lower())
        if canonical_name:
            for alias in aliases:
                if alias and alias != canonical_name:
                    a = alias.strip().lower()
                    # Skip junk DrugBank aliases: IUPAC/DNA 254-mers, CAS
                    # numbers, CJK strings, concentration/form artifacts.
                    # (Short aliases are kept — dual FlashText + phantom
                    # filter handle them; long ones are pure vector bloat.)
                    if _is_junk_alias(a):
                        continue
                    synonym_map[a] = canonical_name

        minimal_drug = {}
        interactions_raw = drug.get('{http://www.drugbank.ca}drug-interactions')
        if interactions_raw:
            i_list = []
            if isinstance(interactions_raw, dict):
                i_list = interactions_raw.get('{http://www.drugbank.ca}drug-interaction', [])
            elif isinstance(interactions_raw, list):
                i_list = interactions_raw
                
            if isinstance(i_list, dict):
                i_list = [i_list]
                
            minimal_interactions = []
            for inter in i_list:
                minimal_interactions.append({
                    '{http://www.drugbank.ca}name': inter.get('{http://www.drugbank.ca}name', ''),
                    '{http://www.drugbank.ca}description': inter.get('{http://www.drugbank.ca}description', ''),
                    '{http://www.drugbank.ca}drugbank-id': inter.get('{http://www.drugbank.ca}drugbank-id', '')
                })
            
            if minimal_interactions:
                minimal_drug['{http://www.drugbank.ca}drug-interactions'] = {
                    '{http://www.drugbank.ca}drug-interaction': minimal_interactions
                }

        if name:
            drug_index[name.lower()] = minimal_drug
            medicine_names_set.add(name.lower())

        db_ids = drug.get('{http://www.drugbank.ca}drugbank-id', [])
        if isinstance(db_ids, str):
            db_ids = [db_ids]
        for db_id in db_ids:
            drug_index[db_id.lower()] = minimal_drug

    # Manual overrides for brand names missing from DrugBank
    synonym_map["dolo"] = "acetaminophen"
    synonym_map["paracetamol"] = "acetaminophen"
    synonym_map["salbutamol"] = "albuterol"
    synonym_map["ramistar-am"] = ["amlodipine", "ramipril"]

    reuse_existing_cache = False
    if not drugs:
        if medicine_data_file.exists() and medicine_synonyms_file.exists():
            print("[WARN] DrugBank JSON unavailable - reusing cached medicine data for embeddings only.")
            with open(medicine_data_file, 'r', encoding='utf-8') as f:
                medicine_names = json.load(f)['medicine_names']
            with open(medicine_synonyms_file, 'r', encoding='utf-8') as f:
                synonym_map = json.load(f)
            reuse_existing_cache = True
            print(f"       Loaded {len(medicine_names)} cached generic medicine names in {time.time() - t0:.1f}s")
        else:
            raise FileNotFoundError(
                f"DrugBank JSON '{loader.json_path}' not found and no cached medicine data exists. "
                "Restore the DrugBank JSON or build the cache once with it available."
            )
    else:
        medicine_names = list(medicine_names_set)
        print(f"       Loaded {len(medicine_names)} generic medicine names in {time.time() - t0:.1f}s")

    # Indian brand lexicon (data/indian_brands.json, built by
    # build_indian_brands.py) merges into synonym_map AFTER the manual
    # overrides so fresh brand data beats them. Must run before alias
    # embedding so new brand surfaces get vectors. NOTE: only marks the
    # synonym file dirty - never flips reuse_existing_cache, which also
    # guards the DrugBank index write (an empty rebuild would wipe it).
    from clinical_copilot.brands import load_brand_lexicon as _load_lex
    _prev_brand_keys = set()
    _sidecar = cache_dir / 'brand_keys.json'
    if _sidecar.exists():
        try:
            _prev_brand_keys = set(json.load(open(_sidecar, encoding='utf-8')))
        except Exception:
            pass
    n_brands = apply_brand_lexicon(synonym_map)
    synonyms_dirty = bool(n_brands)
    if n_brands:
        print(f"       Merged {n_brands} Indian brand aliases into synonym_map")
    # Delete stale brand keys: surfaces from a previous lexicon that are
    # gone from the current one (merge only adds). Sidecar tracks exactly
    # which keys brands contributed, so DrugBank shorts are untouched.
    try:
        _lex = _load_lex()
        _cur = set()
        for _b, _e in _lex.items():
            _cur.add(str(_b).strip().lower())
            for _v in _e.get('variants', []):
                _cur.add(str(_v).strip().lower())
        _cur.discard('')
        _stale = [k for k in _prev_brand_keys if k not in _cur and k in synonym_map]
        for k in _stale:
            del synonym_map[k]
        if _stale:
            synonyms_dirty = True
            print(f"       Removed {len(_stale)} stale brand keys")
        with open(_sidecar, 'w', encoding='utf-8') as f:
            json.dump(sorted(_cur), f, ensure_ascii=False)
    except Exception as e:
        print(f"       [WARN] brand sidecar update failed: {e}")
    # Drop junk-pattern keys that predate the step-1 filter (stale cache
    # from older builds): long IUPAC, parens/brackets, CAS, non-alpha.
    # Generic canonical names are never dropped.
    _generics = set(medicine_names)
    _junk_keys = [k for k in list(synonym_map)
                  if k not in _generics and _is_junk_alias(k)]
    for k in _junk_keys:
        del synonym_map[k]
    if _junk_keys:
        synonyms_dirty = True
        print(f"       Removed {len(_junk_keys)} junk alias keys")

    # ── Step 2: Compute embeddings on GPU ────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[2/4] Computing embeddings on {device.upper()}...")
    t0 = time.time()

    trust_remote_code = config.get('models', 'embeddings', 'trust_remote_code', default=False)
    dtype_name = config.get('models', 'embeddings', 'dtype', default=None)
    resolved_dtype = resolve_torch_dtype(dtype_name)
    model_kwargs = {'dtype': resolved_dtype} if resolved_dtype is not None else None

    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=bool(trust_remote_code),
        model_kwargs=model_kwargs,
    )

    batch_override = config.get('models', 'embeddings', 'batch_size', default=None)
    large_markers = ('qwen', 'gemma', 'wemm', 'vl-embedding', '278m', '2b', '4b', '7b', '8b')
    if batch_override:
        batch_size = int(batch_override)
    elif any(marker in model_name.lower() for marker in large_markers):
        batch_size = 16 if device == 'cuda' else 4
    else:
        batch_size = 2048 if device == 'cuda' else 128

    if aliases_only:
        print("       --aliases-only: skipping generic-embedding rebuild.")
    else:
        embeddings_np = model.encode(
            medicine_names,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings_np = embeddings_np.astype('float32')

        # Convert to torch tensor and save via safetensors
        embeddings_tensor = torch.from_numpy(embeddings_np)
        save_file({"embeddings": embeddings_tensor}, str(embeddings_file))

        embed_time = time.time() - t0
        print(f"       Shape: {embeddings_tensor.shape}  |  "
              f"Saved to {embeddings_file}  |  {embed_time:.1f}s")

    # ── Step 2b: Alias embeddings (idea #2: BioSyn-style synonym marginalization)
    # Every alias surface form gets its own vector; at query time scores are
    # max-pooled over all aliases of a concept. Aliases identical to a generic
    # are skipped (the generic vector already represents them).
    print("[2b/4] Computing alias embeddings...")
    t0 = time.time()
    generic_set = set(medicine_names)
    alias_texts, alias_concepts = [], []
    seen_aliases = set()
    for alias, val in synonym_map.items():
        a = (alias or '').strip().lower()
        if not a or a in generic_set or a in seen_aliases:
            continue
        concepts = val if isinstance(val, list) else [val]
        concepts = [str(c).strip().lower() for c in concepts if str(c or '').strip()]
        if not concepts:
            continue
        seen_aliases.add(a)
        alias_texts.append(a)
        alias_concepts.append(concepts)
    print(f"       {len(alias_texts)} alias surface forms "
          f"(skipped {len(synonym_map) - len(alias_texts)} generics/empties)")

    alias_np = model.encode(
        alias_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype('float32')
    save_file({"embeddings": torch.from_numpy(alias_np)}, str(alias_embeddings_file))
    with open(alias_data_file, 'w', encoding='utf-8') as f:
        json.dump({"alias_texts": alias_texts, "alias_concepts": alias_concepts},
                  f, ensure_ascii=False)
    print(f"       Shape: {alias_np.shape}  |  Saved to {alias_embeddings_file}  |  "
          f"{time.time() - t0:.1f}s")

    # ── Step 2c: TF-IDF index (idea #9: sparse char-ngram retrieval) ─
    print("[2c/4] Building TF-IDF index...")
    t0 = time.time()
    from clinical_copilot.tfidf_index import build_tfidf_index, save_tfidf_index
    tfidf_corpus = list(medicine_names) + alias_texts
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(tfidf_corpus)
    save_tfidf_index(tfidf_vectorizer, tfidf_matrix, str(cache_dir))
    print(f"       {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features  |  "
          f"{time.time() - t0:.1f}s")

    # ── Step 3: Save medicine metadata ───────────────────────────────
    if reuse_existing_cache and not synonyms_dirty:
        print("[3/4] Reusing cached medicine metadata and synonyms.")
    else:
        print("[3/4] Saving medicine metadata and synonyms...")
        t0 = time.time()

        if not reuse_existing_cache:
            medicine_cache = {
                "medicine_names": medicine_names,
            }
            with open(medicine_data_file, 'w', encoding='utf-8') as f:
                json.dump(medicine_cache, f, ensure_ascii=False)
            print(f"       Saved {len(medicine_names)} names to {medicine_data_file}")

        with open(medicine_synonyms_file, 'w', encoding='utf-8') as f:
            json.dump(synonym_map, f, ensure_ascii=False)

        print(f"       Saved {len(synonym_map)} synonyms to {medicine_synonyms_file}  |  "
              f"{time.time() - t0:.1f}s")

    # ── Step 4: Build and save DrugBank index ────────────────────────
    if reuse_existing_cache:
        print(f"[4/4] Reusing existing DrugBank index at {drugbank_index_file}.")
    else:
        print(f"[4/4] Saving DrugBank index to {drugbank_index_file}...")
        t0 = time.time()

        # Gzip-compressed JSON (typically ~30-50 MB vs 2.4 GB raw)
        # Serialize fully to bytes first, then compress in one shot
        print("Creating json dump...")
        json_bytes = json.dumps(drug_index, ensure_ascii=False).encode('utf-8')
        print("Compressing...")
        compressed = gzip.compress(json_bytes, compresslevel=9)
        print("Saving zipped data...")
        with open(str(drugbank_index_file), 'wb') as f:
            f.write(compressed)
        del json_bytes, compressed  # free memory immediately
        print(f"       {len(drug_index)} index entries → {drugbank_index_file}  |  "
              f"{time.time() - t0:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    total = time.time() - overall_start
    sizes = []
    for p in [embeddings_file, alias_embeddings_file, medicine_data_file,
              drugbank_index_file]:
        sz = os.path.getsize(p) / (1024 * 1024)
        sizes.append(f"  {p.name:40s} {sz:>8.1f} MB")

    print(f"\n{'='*58}")
    print(f"  Cache build completed in {total:.1f}s")
    print(f"{'='*58}")
    for s in sizes:
        print(s)
    print(f"{'='*58}")
    print("  Run the UI - it will now load from cache automatically.")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    import sys
    build_cache(
        aliases_only="--aliases-only" in sys.argv,
        tfidf_only="--tfidf-only" in sys.argv,
    )
