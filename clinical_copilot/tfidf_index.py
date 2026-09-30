"""
Sparse TF-IDF retrieval for idea #9: char-ngram index alongside dense embeddings.

Builds a TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3)) over the same
corpus strings used for dense retrieval. At query time, scores are fused with
dense + RapidFuzz signals via min-max-normalised linear combination.
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Fusion weights (idea #9): dense > tfidf > rf
_FUSION_DENSE = 0.55
_FUSION_TFIDF = 0.30
_FUSION_RF = 0.15
_ACCEPT_THRESHOLD = 0.62
_LEAD_MARGIN = 0.03
_TFIDF_TOP_K = 50


def build_tfidf_index(texts: List[str]) -> Tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a char-ngram TF-IDF index over the given texts."""
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts).tocsr()
    logger.info(
        f"TF-IDF index built: {len(texts)} docs, "
        f"{matrix.shape[1]} features, nnz={matrix.nnz}, "
        f"{time.time() - t0:.1f}s"
    )
    return vectorizer, matrix


def query_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    top_k: int = _TFIDF_TOP_K,
) -> List[Tuple[int, float]]:
    """Return top-k (doc_index, score) pairs for a query string."""
    q = vectorizer.transform([query])
    scores = (matrix @ q.T).toarray().ravel()
    if top_k >= len(scores):
        idx = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idx = part[np.argsort(scores[part])[::-1]]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def save_tfidf_index(
    vectorizer: TfidfVectorizer, matrix: sp.csr_matrix, cache_dir: str = "cache"
) -> None:
    """Persist vectorizer (pickle) + sparse matrix (.npz) + manifest."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    vec_path = d / "tfidf_vectorizer.pkl"
    mat_path = d / "tfidf_matrix.npz"
    man_path = d / "tfidf_manifest.json"

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(str(mat_path), matrix)

    manifest = {
        "n_docs": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "content_hash": _content_hash(vectorizer),
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"TF-IDF saved: {vec_path.stat().st_size / 1024:.0f} KB vec, "
        f"{mat_path.stat().st_size / 1024:.0f} KB mat"
    )


def load_tfidf_index(cache_dir: str = "cache"):
    """Load TF-IDF artifacts. Returns (vectorizer, matrix) or (None, None)."""
    d = Path(cache_dir)
    vec_path = d / "tfidf_vectorizer.pkl"
    mat_path = d / "tfidf_matrix.npz"
    if not vec_path.exists() or not mat_path.exists():
        return None, None
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        matrix = sp.load_npz(str(mat_path))
        logger.info(f"TF-IDF loaded: {matrix.shape[0]} docs, {matrix.shape[1]} features")
        return vectorizer, matrix
    except Exception as e:
        logger.warning(f"Failed to load TF-IDF index: {e}")
        return None, None


def _content_hash(vectorizer: TfidfVectorizer) -> str:
    """Stable hash of the vectorizer vocabulary for staleness detection."""
    vocab = sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])
    raw = json.dumps(vocab).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def fuse_scores(
    dense_scores: Dict[str, float],
    tfidf_scores: Dict[str, float],
    rf_scores: Dict[str, float],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Linear fusion of three score dicts keyed by concept name.
    Min-max normalises each signal across the union, then combines
    0.55*dense + 0.30*tfidf + 0.15*rf.  Returns top-k (concept, fused_score).
    """
    all_concepts = set(dense_scores) | set(tfidf_scores) | set(rf_scores)
    if not all_concepts:
        return []

    def _norm(scores: Dict[str, float]) -> Dict[str, float]:
        vals = [scores.get(c, 0.0) for c in all_concepts]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {c: 0.0 for c in all_concepts}
        return {c: (scores.get(c, 0.0) - lo) / (hi - lo) for c in all_concepts}

    d_n = _norm(dense_scores)
    t_n = _norm(tfidf_scores)
    r_n = _norm(rf_scores)

    fused = {
        c: _FUSION_DENSE * d_n[c] + _FUSION_TFIDF * t_n[c] + _FUSION_RF * r_n[c]
        for c in all_concepts
    }
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ranked


def should_accept(
    ranked: List[Tuple[str, float]],
    generic_scores: Optional[Dict[str, float]] = None,
) -> Optional[Tuple[str, float]]:
    """
    Accept the top-ranked concept if it clears the fused threshold and
    holds a sufficient lead over a different-concept runner-up.
    Mirrors the _POOL_MARGIN rule from embeddings.py.
    """
    if not ranked or ranked[0][1] < _ACCEPT_THRESHOLD:
        return None
    if len(ranked) > 1 and ranked[0][0] != ranked[1][0]:
        if ranked[0][1] - ranked[1][1] < _LEAD_MARGIN:
            return None
    return ranked[0]


# --- Glued-token segmentation (idea #9) ---

def segment_glued_token(
    token: str, known_names: set, min_len: int = 8, max_pieces: int = 3
) -> Optional[List[str]]:
    """
    Split a spaceless token into known drug names.
    Only runs on tokens >= min_len chars.  Accepts a split only if EVERY
    piece is a known drug (in known_names).  Returns the split list or None.
    """
    token = token.strip().lower()
    if len(token) < min_len or " " in token:
        return None
    # Never split a token that is itself a known drug name: segmentation
    # exists for glued unknowns, but 'phenobarbital' -> 'phe'+'no'+'barbital'
    # (all "known" fragments) destroys the real mention and the fragments
    # then match wrong drugs (phenylalanine, nitric oxide, barbital).
    if token in known_names:
        return None

    # Try all split points (2 or 3 pieces)
    n = len(token)
    for i in range(2, min(n - 1, n - 2 + 1)):
        left = token[:i]
        if left in known_names:
            right = token[i:]
            if right in known_names:
                return [left, right]
            # Try splitting the right part again
            if max_pieces >= 3 and len(right) >= 3:
                for j in range(2, len(right) - 1):
                    mid = right[:j]
                    far = right[j:]
                    if mid in known_names and far in known_names:
                        return [left, mid, far]
    return None
