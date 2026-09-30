"""
Medicine entity extraction combining NER, embeddings, and fuzzy matching.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clinical_copilot.config import ConfigLoader
from clinical_copilot.embeddings import SemanticEmbeddingGenerator
from clinical_copilot.ner import BioNERModule

try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from symspellpy.symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Enhanced entity extractor with ML/DL integration.
    Combines BioBERT NER, semantic embeddings, and traditional fuzzy matching.
    """
    
    def __init__(self, medicine_names: List[str], config: ConfigLoader,
                 ner_module: Optional[BioNERModule] = None,
                 embedding_gen: Optional[SemanticEmbeddingGenerator] = None,
                 precomputed_embeddings: Optional[np.ndarray] = None,
                 synonym_map: Optional[Dict[str, str]] = None,
                 alias_texts: Optional[List[str]] = None,
                 alias_embeddings: Optional[np.ndarray] = None,
                 alias_concepts: Optional[List[List[str]]] = None,
                 tfidf_vectorizer=None,
                 tfidf_matrix=None):
        """
        Initialize entity extractor with ML capabilities.

        Args:
            medicine_names: List of known generic medicine names
            config: Configuration loader instance
            ner_module: Optional BioBERT NER module
            embedding_gen: Optional embedding generator
            precomputed_embeddings: Optional pre-loaded embeddings (from cache)
            synonym_map: Optional mapping of brand/alias names to generic canonical names
            alias_texts: Optional alias strings for synonym-marginalized retrieval (idea #2)
            alias_embeddings: Optional pre-loaded alias vectors, aligned with alias_texts
            alias_concepts: Optional per-alias canonical concept list, aligned with alias_texts
            tfidf_vectorizer: Optional fitted TfidfVectorizer (idea #9)
            tfidf_matrix: Optional sparse TF-IDF matrix aligned with generics+aliases
        """
        self.medicine_names = medicine_names
        self.synonym_map = synonym_map or {}
        self.config = config
        self.ner_module = ner_module
        self.embedding_gen = embedding_gen
        self._alias_texts = alias_texts or []
        self._alias_embeddings = alias_embeddings
        self._alias_concepts = alias_concepts or []
        self._use_alias_pool = bool(
            self._alias_texts and self._alias_embeddings is not None
            and len(self._alias_texts) == len(self._alias_concepts))
        self._tfidf_vectorizer = tfidf_vectorizer
        self._tfidf_matrix = tfidf_matrix
        self._use_tfidf = False  # Disabled: fused-score scale mismatch causes phantom entities
        
        self.use_ml_ner = config.get('pipeline', 'use_ml_ner', default=True)
        self.use_embeddings = config.get('pipeline', 'use_embeddings', default=True)
        self.fuzzy_threshold = config.get('entity_extraction', 'fuzzy_matching', 'threshold', default=0.75)
        
        self.flashtext_processor = None
        self.flashtext_short_processor = None
        if FLASHTEXT_AVAILABLE:
            logger.info("Initializing FlashText KeywordProcessor...")
            self.flashtext_processor = KeywordProcessor(case_sensitive=False)
            # Short keys (<=2 chars) use a case-sensitive processor and skip
            # common English words entirely — "no" (the word) must never match
            # "NO" -> nitric oxide, even when both are lowercase.
            self.flashtext_short_processor = KeywordProcessor(case_sensitive=True)
            _SHORT_KEY_MAX = 2
            n_short = 0
            for med in self.medicine_names:
                if len(med) <= _SHORT_KEY_MAX:
                    if med.lower() not in self._ENTITY_STOPWORDS:
                        self.flashtext_short_processor.add_keyword(med, med)
                        n_short += 1
                else:
                    self.flashtext_processor.add_keyword(med, med)
            for syn, canon in self.synonym_map.items():
                if isinstance(canon, list):
                    canon = canon[0]
                if len(syn) <= _SHORT_KEY_MAX:
                    if syn.lower() not in self._ENTITY_STOPWORDS:
                        self.flashtext_short_processor.add_keyword(syn, canon)
                        n_short += 1
                else:
                    self.flashtext_processor.add_keyword(syn, canon)
            logger.info(
                f"FlashText initialized with "
                f"{len(self.medicine_names) + len(self.synonym_map)} keywords "
                f"({n_short} short/case-sensitive).")

        # Use pre-loaded embeddings from cache, or compute them
        self._medicine_embeddings = None
        if precomputed_embeddings is not None:
            logger.info("Using pre-loaded medicine embeddings from cache.")
            self._medicine_embeddings = precomputed_embeddings
        elif self.use_embeddings and self.embedding_gen:
            logger.info("Pre-computing medicine name embeddings (one-time)...")
            self._medicine_embeddings = self.embedding_gen.precompute_embeddings(self.medicine_names)
        
    # Common non-medicine stopwords to filter from entity results
    _ENTITY_STOPWORDS = {
        'prescribe', 'prescribed', 'patient', 'chart', 'plan', 'management',
        'diabetes', 'daily', 'twice', 'once', 'thrice', 'bedtime', 'morning',
        'tablet', 'capsule', 'injection', 'mg', 'mcg', 'ml', 'g',
        'before', 'after', 'with', 'meals', 'breakfast', 'lunch', 'dinner',
        'cardiovascular', 'protection', 'antiplatelet', 'cholesterol',
        'hypertension', 'fever', 'hypothetical', 'drug', 'brand',
        'for', 'and', 'the', 'of', 'in', 'to', 'on', 'at', 'by',
        'no', 'yes', 'not', 'is', 'are', 'was', 'be', 'do', 'so', 'up',
        'if', 'it', 'as', 'or', 'an', 'we', 'he', 'me', 'my',
    }

    def _is_valid_entity(self, entity: str) -> bool:
        """
        Check if an extracted entity is likely a valid medicine name.
        Filters out common non-medicine words and very short tokens.
        """
        if not entity or len(entity.strip()) < 2:
            return False
        # Check if the entire entity (lowered) is a stopword
        if entity.strip().lower() in self._ENTITY_STOPWORDS:
            return False
        # Check if all words in the entity are stopwords
        words = entity.strip().lower().split()
        if all(w in self._ENTITY_STOPWORDS for w in words):
            return False
        # Reject purely numeric tokens
        if re.match(r'^\d+$', entity.strip()):
            return False
        return True

    # --- Mention cleaning patterns (idea #3): strip dose / form / frequency
    # tokens from NER/FlashText spans BEFORE retrieval. Order matters: brackets
    # first (keeps parenthesised content for later rules), bare numbers last.
    # Hyphens, slashes, plus and ampersand are preserved for combo splitting.
    _BRACKET_RE = re.compile(r'[\(\)\[\]\{\}]')
    # Stray punctuation (keeps - / + & for combo splitting)
    _PUNCT_RE = re.compile(r'[^\w\s\-/+&]')
    _DOSE_RE = re.compile(
        r'\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|µg|ug|ml|l|iu|units?|meq|mmol|%|w/v|v/v)\b',
        re.IGNORECASE)
    # Glued dose: drug name immediately followed by dose (no space)
    _GLOUED_DOSE_RE = re.compile(
        r'([a-zA-Z])(\d+(?:\.\d+)?\s?(?:mg|g|mcg|µg|ug|ml|l|iu|units?|meq|mmol|%))',
        re.IGNORECASE)
    # Dose unit glued to frequency: '500mgBD' -> '500mg BD'
    _GLOUED_FREQ_RE = re.compile(
        r'(\d+(?:\.\d+)?(?:mg|g|mcg|µg|ug|ml|l|iu|units?|meq|mmol|%))([a-zA-Z]{1,3})',
        re.IGNORECASE)
    _FORM_RE = re.compile(
        r'\b(?:tablets?|tabs?\.?|capsules?|caps?\.?|syrups?|suspensions?|susp\.?|'
        r'injections?|inj\.?|drops?|ointments?|creams?|gels?|solutions?|lotions?|'
        r'powders?|sachets?|patches?|sprays?|inhalers?|rotacaps?|respules?|'
        r'suppositor(?:y|ies))\b', re.IGNORECASE)
    _FREQ_RE = re.compile(
        r'\b(?:twice|thrice|once|daily|weekly|monthly|hourly|qid|qds|tid|tds|'
        r'bid|bd|od|qhs|hs|sos|prn|q\d+h?|morning|evening|night|bedtime|'
        r'breakfast|lunch|dinner|meals?|food|before|after|empty\s+stomach)\b',
        re.IGNORECASE)
    _MODIFIER_RE = re.compile(
        r'\b(?:xl|xr|sr|er|mr|cr|ds|odt|forte)\b', re.IGNORECASE)
    _NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

    def _clean_mention(self, mention: str) -> str:
        """
        Strip dose/strength, dosage form, frequency and release-modifier tokens
        from a raw mention (e.g. 'Dolo 650 twice daily' -> 'dolo',
        'Starpress XL 25mg TAB' -> 'starpress').
        """
        text = mention.strip().lower()
        # Split glued dose: 'simvastatin20mg' -> 'simvastatin 20mg'
        text = self._GLOUED_DOSE_RE.sub(r'\1 \2', text)
        # Split glued frequency: '500mgbd' -> '500mg bd'
        text = self._GLOUED_FREQ_RE.sub(r'\1 \2', text)
        text = self._BRACKET_RE.sub(' ', text)
        text = self._PUNCT_RE.sub(' ', text)
        text = self._DOSE_RE.sub(' ', text)
        text = self._FORM_RE.sub(' ', text)
        text = self._FREQ_RE.sub(' ', text)
        text = self._MODIFIER_RE.sub(' ', text)
        text = self._NUMBER_RE.sub(' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # --- Combination-product decomposition (idea #6). Suffix splits require an
    # EXACT stem match: blindly mapping e.g. 'vitamin-d' to domperidone, or
    # 'ramistar' to a fuzzy guess, would be a false DDI alert. Unknown brand
    # stems (telma, ramistar) need the brand lexicon (idea #4) or RxNorm MIN
    # (idea #5); until then the mention is left whole for exact/fuzzy paths.
    _COMBO_DELIM_RE = re.compile(
        r'\s*(?:/|\+|&|\band\b|\bwith\b|\bplus\b)\s*', re.IGNORECASE)
    _COMBO_SUFFIX_RE = re.compile(
        r'^(?P<stem>.+?)[-_](?P<suffix>am|h|m|d|plus|ls|ln)$', re.IGNORECASE)
    _COMBO_SUFFIX_INGREDIENTS = {
        'am': 'amlodipine',
        'h': 'hydrochlorothiazide',
        'm': 'metformin',
        'd': 'domperidone',
    }
    _COMBO_SUFFIX_STRIP = {'plus', 'ls', 'ln'}  # stem only; 2nd agent unknown

    def _is_known_drug(self, name: str, allow_fuzzy: bool = True) -> bool:
        """Check whether a fragment resolves to a known drug (exact or fuzzy)."""
        name = name.strip().lower()
        if not name:
            return False
        self._ensure_fuzzy_corpus()
        if name in self._known_names:
            return True
        if allow_fuzzy and RAPIDFUZZ_AVAILABLE:
            hit = _rf_process.extractOne(
                name, self._fuzzy_corpus, scorer=_rf_fuzz.WRatio,
                score_cutoff=self.fuzzy_threshold * 100)
            return hit is not None
        return False

    def _split_combo(self, mention: str) -> List[str]:
        """
        Decompose a combination-product mention into linkable fragments:
        'amox/clav' -> ['amox', 'clav'], 'Telma-H' -> ['telma',
        'hydrochlorothiazide'], 'Xyz-Plus' -> ['xyz']. Returns [mention]
        when no decomposition applies.
        """
        m = mention.strip().lower()
        if not m:
            return []
        parts = [p.strip() for p in self._COMBO_DELIM_RE.split(m) if p.strip()]
        if len(parts) > 1:
            return parts
        sm = self._COMBO_SUFFIX_RE.match(m)
        if sm:
            stem, suffix = sm.group('stem').strip(), sm.group('suffix').lower()
            if not stem:
                return [m]
            if suffix in self._COMBO_SUFFIX_STRIP and self._is_known_drug(stem, allow_fuzzy=False):
                return [stem]
            if suffix in self._COMBO_SUFFIX_INGREDIENTS and self._is_known_drug(stem, allow_fuzzy=False):
                return [stem, self._COMBO_SUFFIX_INGREDIENTS[suffix]]
        return [m]

    # --- Score fusion (idea #9): dense + TF-IDF + RapidFuzz, linear
    # combination with per-query min-max normalisation.  Accept at >= 0.62
    # with the 0.03 lead rule (mirrors _POOL_MARGIN).
    _FUSION_DENSE = 0.55
    _FUSION_TFIDF = 0.30
    _FUSION_RF = 0.15
    _FUSION_ACCEPT = 0.62
    _FUSION_LEAD = 0.03

    def _get_dense_concept_scores(self, query: str) -> Dict[str, float]:
        """Raw concept scores from dense embeddings (generics + alias max-pool)."""
        if not self.embedding_gen:
            return {}
        query_emb = self.embedding_gen.generate_embedding(query)
        if query_emb is None:
            return {}
        concept_scores: Dict[str, float] = {}
        if self._medicine_embeddings is not None:
            emb = self._medicine_embeddings
            q = query_emb
            sims = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-8)
            for i, g in enumerate(self.medicine_names):
                s = float(sims[i])
                if s > concept_scores.get(g, -2.0):
                    concept_scores[g] = s
        if self._use_alias_pool:
            emb = self._alias_embeddings
            q = query_emb
            sims = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-8)
            for j, concepts in enumerate(self._alias_concepts):
                s = float(sims[j])
                for c in concepts:
                    if s > concept_scores.get(c, -2.0):
                        concept_scores[c] = s
        return concept_scores

    def _get_tfidf_concept_scores(self, query: str) -> Dict[str, float]:
        """Concept scores from TF-IDF index (max-pooled over aliases)."""
        if not self._use_tfidf:
            return {}
        from clinical_copilot.tfidf_index import query_tfidf
        hits = query_tfidf(query, self._tfidf_vectorizer, self._tfidf_matrix, top_k=50)
        n_gen = len(self.medicine_names)
        concept_scores: Dict[str, float] = {}
        for idx, score in hits:
            if idx < n_gen:
                concept = self.medicine_names[idx]
                if score > concept_scores.get(concept, -2.0):
                    concept_scores[concept] = score
            else:
                alias_idx = idx - n_gen
                if alias_idx < len(self._alias_concepts):
                    for c in self._alias_concepts[alias_idx]:
                        if score > concept_scores.get(c, -2.0):
                            concept_scores[c] = score
        return concept_scores

    def _get_rf_scores(self, query: str, candidates: set) -> Dict[str, float]:
        """RapidFuzz WRatio scores for candidate concepts (0-1 scale)."""
        if not RAPIDFUZZ_AVAILABLE:
            return {}
        self._ensure_fuzzy_corpus()
        scores: Dict[str, float] = {}
        cleaned = query.strip().lower()
        for c in candidates:
            hit = _rf_process.extractOne(
                cleaned, [c], scorer=_rf_fuzz.WRatio, score_cutoff=0)
            if hit:
                scores[c] = hit[1] / 100.0
        return scores

    def _segment_glued(self, token: str) -> Optional[List[str]]:
        """
        Split a spaceless token into known drug names (idea #9).
        Only runs on tokens >= 8 chars.  Accepts a split only if EVERY
        piece is a known drug.  Returns the split list or None.
        """
        from clinical_copilot.tfidf_index import segment_glued_token
        self._ensure_fuzzy_corpus()
        return segment_glued_token(token, self._known_names)

    def _fuse_retrieval(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Linear fusion of dense + TF-IDF + RapidFuzz scores (idea #9).
        Min-max normalises each signal across the candidate union, then
        combines 0.55*dense + 0.30*tfidf + 0.15*rf.  Returns top-k
        (concept, fused_score) with the 0.03 lead rule applied.
        """
        from clinical_copilot.tfidf_index import fuse_scores, should_accept
        dense_scores = self._get_dense_concept_scores(query)
        tfidf_scores = self._get_tfidf_concept_scores(query)
        all_cands = set(dense_scores) | set(tfidf_scores)
        rf_scores = self._get_rf_scores(query, all_cands)
        ranked = fuse_scores(dense_scores, tfidf_scores, rf_scores, top_k=top_k)
        accepted = should_accept(ranked)
        if accepted:
            return [accepted] + [(c, s) for c, s in ranked if c != accepted[0]]
        return ranked

    def _resolve_names(self, entity: str) -> list:
            """Resolve a drug name to its canonical generic name(s) if it's a known synonym/brand."""
            lower_entity = entity.strip().lower()

            # Don't resolve common English words that happen to be short
            # synonym keys ("no" -> nitric oxide, "of" -> ofloxacin).
            if lower_entity in self._ENTITY_STOPWORDS and len(lower_entity) <= 2:
                return [self._get_original_name(lower_entity)]

            # Try exact match first
            if lower_entity in self.synonym_map:
                val = self.synonym_map[lower_entity]
                if isinstance(val, list):
                    return [self._get_original_name(v) for v in val]
                else:
                    return [self._get_original_name(val)]

            # Try the cleaned mention (dose/form/frequency stripped)
            brand = self._clean_mention(lower_entity)

            if brand and brand in self.synonym_map:
                val = self.synonym_map[brand]
                if isinstance(val, list):
                    return [self._get_original_name(v) for v in val]
                else:
                    return [self._get_original_name(val)]

            return [self._get_original_name(lower_entity)]

    def extract_entities(self, input_text: str) -> Dict[str, Any]:
        # Split glued name-dose / dose-frequency tokens BEFORE any matcher
        # runs: FlashText needs word boundaries and NER misses tokens like
        # 'Simvastatin20mg' or '500mgBD'. Same patterns as _clean_mention.
        input_text = self._GLOUED_DOSE_RE.sub(r'\1 \2', input_text)
        input_text = self._GLOUED_FREQ_RE.sub(r'\1 \2', input_text)
        results = {
            'entities': [],
            'ml_entities': [],
            'confidence_scores': [],
            'methods_used': []
        }
        
        extracted_meds = []
        ner_confidence = 0.8
        
        if getattr(self, 'flashtext_processor', None):
            found_keywords = self.flashtext_processor.extract_keywords(input_text, span_info=True)
            if getattr(self, 'flashtext_short_processor', None):
                found_keywords += self.flashtext_short_processor.extract_keywords(input_text, span_info=True)
            if found_keywords:
                results['methods_used'].append('flashtext_dictionary')
                for keyword, start, end in found_keywords:
                    mention = input_text[start:end].strip().lower()
                    synonyms = self.synonym_map.get(mention)
                    if isinstance(synonyms, list):
                        extracted_meds.extend(self._get_original_name(s) for s in synonyms)
                    else:
                        extracted_meds.append(keyword)

        # 1. ML NER
        if self.use_ml_ner and self.ner_module:
            try:
                t0 = time.time()
                logger.info("  Running BioBERT NER inference...")
                ner_results = self.ner_module.extract_entities(input_text)
                extracted_meds.extend([ent['text'] for ent in ner_results['entities'].get('MEDICATION', [])])
                ner_confidence = ner_results.get('confidence', 0.8)
                results['methods_used'].append(ner_results.get('method', 'biobert_ner'))
                logger.info(f"  BioBERT NER completed in {time.time() - t0:.2f}s")
            except Exception as e:
                logger.error(f"ML NER failed: {e}")

        all_entities = []
        confidence_map = {}
        semantic_best: Dict[str, float] = {}

        # 2. Semantic Matching (Entity Linking)
        # Instead of embedding the whole input text, embed the specific entities found by BioBERT
        if self.use_embeddings and self.embedding_gen and extracted_meds:
            try:
                t0 = time.time()
                logger.info("  Running semantic embedding search on extracted entities...")

                accept_thresh = self._FUSION_ACCEPT if self._use_tfidf else 0.85
                for med in extracted_meds:
                    if not self._is_valid_entity(med): continue
                    # Idea #3: strip dose/form/frequency before retrieval
                    query = self._clean_mention(med) or med.strip().lower()

                    # Idea #9: try glued-token segmentation first
                    queries_to_try = [query]
                    segments = self._segment_glued(query)
                    if segments:
                        queries_to_try = segments

                    for q in queries_to_try:
                        if self._use_tfidf:
                            matches = self._fuse_retrieval(q, top_k=3)
                        elif self._use_alias_pool:
                            matches = self.embedding_gen.find_most_similar_concepts(
                                q, self.medicine_names, self._medicine_embeddings,
                                self._alias_texts, self._alias_embeddings,
                                self._alias_concepts, top_k=3)
                        else:
                            matches = self.embedding_gen.find_most_similar(
                                q, self.medicine_names, top_k=3,
                                precomputed_embeddings=self._medicine_embeddings
                            )

                        best = -1.0
                        for match_str, match_score in matches:
                            best = max(best, match_score)
                            if match_score >= accept_thresh:
                                original_name = self._get_original_name(match_str)
                                resolved_names = self._resolve_names(original_name)
                                for res_ent in resolved_names:
                                    if res_ent not in all_entities:
                                        all_entities.append(res_ent)
                                        confidence_map[res_ent] = match_score
                                break
                        semantic_best[med] = max(semantic_best.get(med, -1.0), best)

                results['methods_used'].append('semantic_embeddings')
                logger.info(f"  Semantic search completed in {time.time() - t0:.2f}s")
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        elif extracted_meds:
            # Fallback if embeddings are off: clean -> split combos -> resolve
            for med in extracted_meds:
                if not self._is_valid_entity(med): continue
                cleaned = self._clean_mention(med) or med.strip().lower()
                for frag in self._split_combo(cleaned):
                    resolved = self._resolve_names(frag)
                    for res_ent in resolved:
                        if res_ent not in all_entities:
                            all_entities.append(res_ent)
                            confidence_map[res_ent] = ner_confidence

        # 3. Fuzzy fallback safety net (idea #1): mentions FlashText/NER found
        # but semantic scored < 0.85 (or embeddings are off and nothing known
        # resolved). Whole-text scan only when nothing matched at all.
        fuzzy_stages = set()
        _fuzzy_accept = self._FUSION_ACCEPT if self._use_tfidf else 0.85
        for med in extracted_meds:
            if not self._is_valid_entity(med):
                continue
            if semantic_best.get(med, -1.0) >= _fuzzy_accept:
                continue
            cleaned = self._clean_mention(med) or med.strip().lower()
            for frag in self._split_combo(cleaned):
                for canon, conf, stage in self._fuzzy_match_mention(frag):
                    if canon not in all_entities:
                        all_entities.append(canon)
                        confidence_map[canon] = conf
                        fuzzy_stages.add(stage)

        if not all_entities and (RAPIDFUZZ_AVAILABLE or SYMSPELL_AVAILABLE):
            logger.info("  No entities matched; running whole-text fuzzy safety net...")
            for canon, conf, stage in self._fuzzy_extract(input_text):
                for res_ent in self._resolve_names(canon):
                    if res_ent not in all_entities:
                        all_entities.append(res_ent)
                        confidence_map[res_ent] = conf
                        fuzzy_stages.add(stage)

        for stage in sorted(fuzzy_stages):
            results['methods_used'].append(stage)

        valid_entities = [e for e in all_entities if self._is_valid_entity(e)]

        # Filter phantom entities — semantic search can match a query to a
        # DIFFERENT drug (e.g. "omeprazole" → "olanzapine"). Keep an entity
        # only if its name or a synonym appears in the text (exact or fuzzy).
        input_lower = input_text.lower()
        grounded = []
        for e in valid_entities:
            if self._is_mentioned_in_text(e, input_lower):
                grounded.append(e)
            else:
                logger.debug(f"Filtered phantom entity: {e}")
        valid_entities = grounded

        results['entities'] = [{'name': e, 'confidence': confidence_map[e]} for e in valid_entities]
        results['confidence_scores'] = list(confidence_map.values())

        return results

    def _is_mentioned_in_text(self, entity: str, input_lower: str) -> bool:
        """Check if entity name or any of its synonyms appears in the input text.

        Uses exact word-boundary match first, then fuzzy word match to allow
        OCR typos (e.g. 'Capcitabine' -> 'Capecitabine') while still
        rejecting hallucinated drugs (e.g. 'olanzapine' from 'omeprazole').
        """
        import re as _re
        ent_lower = entity.strip().lower()
        if _re.search(r'\b' + _re.escape(ent_lower) + r'\b', input_lower):
            return True

        surfaces = [ent_lower]
        for surface, canon in self.synonym_map.items():
            if isinstance(canon, list):
                canons = [c.lower() for c in canon]
            else:
                canons = [str(canon).lower()]
            if ent_lower in canons:
                surfaces.append(surface.strip().lower())

        # Fuzzy match against words in the text (catches OCR typos)
        words = set(_re.findall(r'[a-z][a-z\-]+', input_lower))
        if RAPIDFUZZ_AVAILABLE and words:
            for sur in surfaces:
                if not sur:
                    continue
                if ' ' in sur:
                    # Multi-word surface (e.g. 'ethinyl estradiol' for
                    # 'ethinylestradiol'): single-word fuzzy can never match
                    # it, so require the words consecutively in order.
                    if _re.search(r'\b' + r'\s+'.join(
                            _re.escape(w) for w in sur.split()) + r'\b',
                            input_lower):
                        return True
                    continue
                hit = _rf_process.extractOne(
                    sur, words, scorer=_rf_fuzz.ratio, score_cutoff=85)
                if hit:
                    return True
        return False
    
    def _ensure_fuzzy_corpus(self) -> List[str]:
        """Lazily build the fuzzy-match corpus + known-name set (idea #1)."""
        if not hasattr(self, '_fuzzy_corpus'):
            corpus = [m.strip().lower() for m in self.medicine_names]
            corpus += [k.strip().lower() for k in self.synonym_map.keys()]
            # De-duplicate while preserving order
            self._fuzzy_corpus = list(dict.fromkeys(c for c in corpus if c))
            self._known_names = set(self._fuzzy_corpus)
        return self._fuzzy_corpus

    def _ensure_symspell(self):
        """
        Lazily build a SymSpell dictionary from the drug corpus so single-token
        typos/OCR errors map to drug names (not general English words).
        Returns None when symspellpy is unavailable.
        """
        if getattr(self, '_symspell', None) is not None:
            return self._symspell
        if not SYMSPELL_AVAILABLE:
            self._symspell = None
            return None
        logger.info("Building SymSpell dictionary from drug corpus...")
        t0 = time.time()
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for term in self._ensure_fuzzy_corpus():
            for tok in term.split():
                if tok:
                    sym.create_dictionary_entry(tok, 1)
        self._symspell = sym
        logger.info(f"  SymSpell dictionary ready in {time.time() - t0:.2f}s")
        return sym

    # Acceptance rule for fuzzy hits (idea #1): a RapidFuzz hit alone must be
    # strong (>= 92) with a clear margin (>= 3 pts over the runner-up resolving
    # to a DIFFERENT drug); weaker hits (config fuzzy_threshold..92) are
    # accepted only when SymSpell independently corrects to the same drug.
    # Abstaining beats a wrong drug: false DDI alerts are worse than misses.
    _FUZZ_STRONG_CUTOFF = 92.0
    _FUZZ_MARGIN = 3.0

    def _fuzzy_match_mention(self, mention: str) -> List[Tuple[str, float, str]]:
        """
        Fuzzy-match one cleaned mention against the drug corpus.
        Returns [(canonical_name, confidence, stage)] with stages scored
        separately: 'fuzzy_symspell' (edit-distance based) corroborates,
        'fuzzy_rapidfuzz' (WRatio / 100) proposes. See _FUZZ_STRONG_CUTOFF.
        """
        cleaned = mention.strip().lower()
        if not cleaned:
            return []
        self._ensure_fuzzy_corpus()
        weak_cutoff = self.fuzzy_threshold * 100

        # Stage 3a: SymSpell single-token / compound correction. A tied top-2
        # (two drug tokens at the same edit distance) is ambiguous -> skip.
        sym_canons: Dict[str, float] = {}
        sym = self._ensure_symspell()
        if sym is not None:
            try:
                if ' ' in cleaned:
                    suggestions = sym.lookup_compound(cleaned, 2)
                else:
                    suggestions = sorted(
                        sym.lookup(cleaned, Verbosity.ALL, 2),
                        key=lambda s: (s.distance, s.term))
                    if (len(suggestions) > 1
                            and suggestions[1].distance == suggestions[0].distance
                            and suggestions[1].term != suggestions[0].term):
                        suggestions = []  # ambiguous correction
                    suggestions = suggestions[:1]
                for sugg in suggestions:
                    term = sugg.term.strip().lower()
                    if term and term != cleaned and term in self._known_names:
                        conf = max(self.fuzzy_threshold, 1.0 - 0.08 * sugg.distance)
                        for canon in self._resolve_names(term):
                            if self._is_valid_entity(canon):
                                prev = sym_canons.get(canon)
                                if prev is None or conf > prev:
                                    sym_canons[canon] = conf
            except Exception as e:
                logger.debug(f"SymSpell lookup failed for {cleaned!r}: {e}")

        # Stage 3b: RapidFuzz WRatio over generics + synonym keys (top-2 for
        # the margin rule). A close runner-up to a different drug -> abstain.
        rf_canon, rf_score = None, 0.0
        if RAPIDFUZZ_AVAILABLE:
            try:
                top2 = _rf_process.extract(
                    cleaned, self._fuzzy_corpus, scorer=_rf_fuzz.WRatio,
                    score_cutoff=weak_cutoff, limit=2)
                if top2:
                    rf_canon = {c for t, _, _ in top2[:1]
                                for c in self._resolve_names(t)}
                    rf_score = top2[0][1]
                    if len(top2) > 1:
                        runner_canons = {c for t, _, _ in top2[1:2]
                                         for c in self._resolve_names(t)}
                        if (runner_canons.isdisjoint(rf_canon)
                                and top2[0][1] - top2[1][1] < self._FUZZ_MARGIN):
                            rf_canon, rf_score = None, 0.0  # ambiguous
            except Exception as e:
                logger.debug(f"RapidFuzz lookup failed for {cleaned!r}: {e}")

        hits: List[Tuple[str, float, str]] = []
        if rf_canon and rf_score >= self._FUZZ_STRONG_CUTOFF:
            for canon in rf_canon:
                if self._is_valid_entity(canon):
                    hits.append((canon, rf_score / 100.0, 'fuzzy_rapidfuzz'))
        elif rf_canon and rf_score >= weak_cutoff:
            agreed = [c for c in rf_canon if c in sym_canons]
            for canon in agreed:
                if self._is_valid_entity(canon):
                    hits.append((canon, max(rf_score / 100.0, sym_canons[canon]),
                                 'fuzzy_symspell+rapidfuzz'))
        return hits

    # Connector words mark multi-drug spans; the per-mention path splits those,
    # so the whole-text net skips any n-gram containing them.
    _NET_CONNECTORS = {'and', 'with', 'plus', 'or'}

    def _ensure_english_words(self):
        """
        Load symspellpy's bundled general-English frequency list. The
        whole-text net skips candidates made entirely of ordinary English
        words ('infection', 'takes', 'patient'): a token that IS English and
        ISN'T a drug keyword can only fuzzy-match by accident.
        """
        if getattr(self, '_english_words', None) is not None:
            return self._english_words
        words = set()
        try:
            import importlib.resources as _res
            p = _res.files('symspellpy') / 'frequency_dictionary_en_82_765.txt'
            with open(str(p), encoding='utf-8') as f:
                for line in f:
                    w = line.split()[0].lower()
                    if w:
                        words.add(w)
            logger.info(f"Loaded {len(words)} general-English words for net filtering.")
        except Exception as e:
            logger.warning(f"Could not load English word list: {e}")
        self._english_words = words
        return words

    def _fuzzy_extract(self, input_text: str) -> List[Tuple[str, float, str]]:
        """
        Whole-text safety net: 1-2 gram scan used only when nothing else
        matched. Returns [(canonical_name, confidence, stage)].
        """
        normalized_text = self._normalize_text(input_text)
        tokens = normalized_text.split()
        english = self._ensure_english_words()

        candidates = []
        for n in range(1, 3):
            for i in range(len(tokens) - n + 1):
                words = tokens[i:i + n]
                if any(w in self._NET_CONNECTORS for w in words):
                    continue
                cand = ' '.join(words)
                if len(cand) < 3:
                    continue
                if all(w in self._ENTITY_STOPWORDS for w in words):
                    continue
                if english and all(w in english for w in words):
                    continue  # ordinary English, not a missed drug mention
                candidates.append(cand)
                if len(candidates) >= 600:
                    break
            if len(candidates) >= 600:
                break

        matched: Dict[str, Tuple[float, str]] = {}
        for candidate in candidates:
            # Strip dose/form tokens so 'paracetmol 500mg' matches as 'paracetmol'
            candidate = self._clean_mention(candidate)
            if not candidate or len(candidate) < 3:
                continue
            for canon, conf, stage in self._fuzzy_match_mention(candidate):
                prev = matched.get(canon)
                if prev is None or conf > prev[0]:
                    matched[canon] = (conf, stage)

        return [(canon, conf, stage)
                for canon, (conf, stage) in matched.items()]
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize input text for processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_original_name(self, lowercase_name: str) -> str:
        """
        Retrieve case-sensitive medicine name (Fallback to title case).
        
        Args:
            lowercase_name: Lowercase medicine name
            
        Returns:
            Original medicine name
        """
        return lowercase_name.title()
