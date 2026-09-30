"""
Semantic embedding generation for clinical text.
"""

import difflib
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from clinical_copilot.config import ConfigLoader

try:
    import torch
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_name: Optional[str]):
    """Map a config dtype string (e.g. 'bfloat16') to a torch dtype. None means model default."""
    if not dtype_name:
        return None
    key = str(dtype_name).strip().lower()
    if key in ('auto', 'default', 'none'):
        return None
    if 'torch' not in globals():
        return None
    mapping = {
        'float32': 'float32', 'fp32': 'float32',
        'float16': 'float16', 'fp16': 'float16', 'half': 'float16',
        'bfloat16': 'bfloat16', 'bf16': 'bfloat16',
    }
    attr = mapping.get(key)
    return getattr(torch, attr) if attr else None


class SemanticEmbeddingGenerator:
    """
    Generates semantic embeddings for clinical text using sentence transformers.
    Enables similarity search and semantic matching.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.model_name = config.get('models', 'embeddings', 'model_name',
                                     default='sentence-transformers/all-MiniLM-L6-v2')
        self.device = config.get('models', 'embeddings', 'device', default='cpu')
        self.similarity_threshold = config.get('models', 'embeddings', 'similarity_threshold', default=0.75)
        self.trust_remote_code = config.get('models', 'embeddings', 'trust_remote_code', default=False)
        self.dtype = config.get('models', 'embeddings', 'dtype', default=None)
        self.batch_size = config.get('models', 'embeddings', 'batch_size', default=None)
        
        self.model = None
        self.embedding_cache = {}
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Sentence transformers not available. Embeddings disabled.")
    
    def _initialize_model(self):
        """Initialize sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            model_kwargs = None
            resolved_dtype = resolve_torch_dtype(self.dtype)
            if resolved_dtype is not None:
                model_kwargs = {'dtype': resolved_dtype}
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=bool(self.trust_remote_code),
                model_kwargs=model_kwargs,
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Generate semantic embedding for text.
        
        Args:
            text: Input text
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector or None if failed
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return None
        
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding = np.asarray(embedding, dtype=np.float32)
            
            if use_cache:
                self.embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        similarity = util.cos_sim(emb1, emb2).item()
        return float(similarity)
    
    def precompute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Pre-compute embeddings for a list of texts in one batch.
        Much faster than encoding one-by-one.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings or None
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return None
        
        try:
            t0 = time.time()
            logger.info(f"Pre-computing embeddings for {len(texts)} items (batch)...")
            batch_size = int(self.batch_size) if self.batch_size else 128
            embeddings = self.model.encode(texts, convert_to_numpy=True,
                                           show_progress_bar=False, batch_size=batch_size)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            logger.info(f"Embeddings pre-computed in {time.time() - t0:.2f}s")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return None

    # Minimum lead the alias pool needs over the generics-only winner to
    # overrule it with a different concept. Prefer the smaller hypothesis
    # (generic string match) on near-ties.
    _POOL_MARGIN = 0.03

    def find_most_similar_concepts(
        self, query: str, generic_texts: List[str],
        generic_embeddings: Optional[np.ndarray],
        alias_texts: List[str], alias_embeddings: Optional[np.ndarray],
        alias_concepts: List[List[str]], top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        BioSyn-style synonym marginalization (idea #2): score the query against
        every generic AND every alias vector, then max-pool over all surface
        forms of each concept. A brand/shorthand mention (Augmentin, PCM)
        matches its own alias vector even when far from the generic string.

        The pool must EARN its win: 35k extra vectors add false-positive
        surface (e.g. 'amoxil' matching minoxidil's alias at 0.92 over
        amoxicillin's generic at 0.906). If the pool winner is a different
        concept than the generics-only winner but leads by less than
        _POOL_MARGIN, the generics-only ranking is returned instead.

        Args:
            query: Query text (already cleaned by the caller)
            generic_texts: Canonical generic names (lowercase)
            generic_embeddings: Pre-computed generic vectors [G, D]
            alias_texts: Alias strings (lowercase, disjoint from generics)
            alias_embeddings: Pre-computed alias vectors [A, D]
            alias_concepts: Per-alias list of canonical concepts (lowercase)
            top_k: Number of top concepts to return

        Returns:
            List of (canonical_concept, pooled_score) tuples, best first
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return []
        try:
            query_emb = self.generate_embedding(query)
            if query_emb is None:
                return []

            generic_scores: Dict[str, float] = {}
            if generic_embeddings is not None:
                sims = util.cos_sim(query_emb, generic_embeddings)[0]
                sims_np = (sims.cpu().numpy() if hasattr(sims, 'cpu')
                           else np.array(sims))
                for i, g in enumerate(generic_texts):
                    s = float(sims_np[i])
                    if s > generic_scores.get(g, -2.0):
                        generic_scores[g] = s

            concept_scores = dict(generic_scores)
            if alias_embeddings is not None and alias_texts:
                if alias_embeddings.shape[0] != len(alias_texts):
                    logger.warning(
                        f"Alias embeddings/texts length mismatch "
                        f"({alias_embeddings.shape[0]} vs {len(alias_texts)}); "
                        f"skipping alias pool.")
                else:
                    sims = util.cos_sim(query_emb, alias_embeddings)[0]
                    sims_np = (sims.cpu().numpy() if hasattr(sims, 'cpu')
                               else np.array(sims))
                    for j, concepts in enumerate(alias_concepts):
                        s = float(sims_np[j])
                        for c in concepts:
                            if s > concept_scores.get(c, -2.0):
                                concept_scores[c] = s

            ranked = lambda scores: sorted(scores.items(), key=lambda kv: kv[1],
                                           reverse=True)[:top_k]
            pooled = ranked(concept_scores)
            if generic_scores and pooled:
                # Exact alias hit: the query IS a known surface form, trust it.
                if query.strip().lower() in set(alias_texts or []):
                    return pooled
                gen_best = max(generic_scores.items(), key=lambda kv: kv[1])
                if (pooled[0][0] != gen_best[0]
                        and pooled[0][1] - gen_best[1] < self._POOL_MARGIN):
                    return ranked(generic_scores)  # pool didn't earn it
            return pooled
        except Exception as e:
            logger.error(f"Concept similarity search failed: {e}")
            return []

    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5,
                           precomputed_embeddings: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Find most similar candidates to query using semantic search.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            precomputed_embeddings: Optional pre-computed embeddings for candidates
            
        Returns:
            List of (candidate, similarity_score) tuples
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return [(c, difflib.SequenceMatcher(None, query.lower(), c.lower()).ratio()) 
                   for c in candidates[:top_k]]
        
        try:
            t0 = time.time()
            query_emb = self.generate_embedding(query)
            if query_emb is None:
                return []
            
            if precomputed_embeddings is not None:
                # Use pre-computed embeddings — fast cosine similarity
                similarities_tensor = util.cos_sim(query_emb, precomputed_embeddings)[0]
                similarities_np = similarities_tensor.cpu().numpy() if hasattr(similarities_tensor, 'cpu') else np.array(similarities_tensor)
                
                # Get top-k indices efficiently
                if len(similarities_np) > top_k:
                    top_indices = np.argpartition(similarities_np, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(similarities_np[top_indices])[::-1]]
                else:
                    top_indices = np.argsort(similarities_np)[::-1]
                
                result = [(candidates[i], float(similarities_np[i])) for i in top_indices]
            else:
                # Fallback: batch-encode candidates (still faster than one-by-one)
                candidate_embs = self.model.encode(candidates, convert_to_numpy=True,
                                                    show_progress_bar=False,
                                                    batch_size=int(self.batch_size) if self.batch_size else 128)
                candidate_embs = np.asarray(candidate_embs, dtype=np.float32)
                similarities_tensor = util.cos_sim(query_emb, candidate_embs)[0]
                similarities_np = similarities_tensor.cpu().numpy() if hasattr(similarities_tensor, 'cpu') else np.array(similarities_tensor)
                
                if len(similarities_np) > top_k:
                    top_indices = np.argpartition(similarities_np, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(similarities_np[top_indices])[::-1]]
                else:
                    top_indices = np.argsort(similarities_np)[::-1]
                
                result = [(candidates[i], float(similarities_np[i])) for i in top_indices]
            
            logger.info(f"Semantic search completed in {time.time() - t0:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
