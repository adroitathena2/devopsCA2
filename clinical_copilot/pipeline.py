"""
Main Clinical Copilot pipeline orchestrating extraction and interaction detection.
"""

import gzip
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from clinical_copilot import local_llm
from clinical_copilot.confidence import ConfidenceEstimator
from clinical_copilot.config import ConfigLoader
from clinical_copilot.data import MedicineDataLoader
from clinical_copilot.ddi import InteractionDetector
from clinical_copilot.embeddings import SemanticEmbeddingGenerator
from clinical_copilot.extraction import EntityExtractor
from clinical_copilot.ner import BioNERModule, TRANSFORMERS_AVAILABLE

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClinicalCopilotPipeline:
    """
    Main orchestrator for the clinical copilot backend with ML/DL integration.
    Combines multiple data sources, ML models, and inference pipelines.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the complete ML-enhanced pipeline.
        Loads from cache if available, otherwise falls back to raw CSV/JSON.
        
        Args:
            config_path: Path to configuration YAML file
        """
        logger.info("="*70)
        logger.info("AI CLINICAL COPILOT - ML-Enhanced Backend Initialization")
        logger.info("="*70)
        
        self.config = ConfigLoader(config_path)
        
        self.ner_module = None
        self.embedding_gen = None
        self.confidence_estimator = None
        self.ocr_min_request_interval_seconds = 1.0
        self.ocr_max_retries = 3
        self.ocr_backoff_base_seconds = 1.0
        self.ocr_backoff_max_seconds = 30.0
        self.ocr_last_request_epoch = 0.0

        if self.config:
            self.ocr_min_request_interval_seconds = float(self.config.get(
                'gemini', 'min_request_interval_seconds', default=self.ocr_min_request_interval_seconds
            ))
            self.ocr_max_retries = int(self.config.get(
                'gemini', 'max_retries', default=self.ocr_max_retries
            ))
            self.ocr_backoff_base_seconds = float(self.config.get(
                'gemini', 'backoff_base_seconds', default=self.ocr_backoff_base_seconds
            ))
            self.ocr_backoff_max_seconds = float(self.config.get(
                'gemini', 'backoff_max_seconds', default=self.ocr_backoff_max_seconds
            ))
        
        if TRANSFORMERS_AVAILABLE and self.config.get('pipeline', 'use_ml_ner', default=True):
            self.ner_module = BioNERModule(self.config)
        
        if TRANSFORMERS_AVAILABLE and self.config.get('pipeline', 'use_embeddings', default=True):
            self.embedding_gen = SemanticEmbeddingGenerator(self.config)
        
        self.confidence_estimator = ConfidenceEstimator(self.config)
        
        # Try cache-first loading
        if self._try_load_from_cache():
            logger.info("Pipeline initialized from CACHE (fast path)")
        else:
            logger.info("Cache not found — loading from raw data files...")
            self._load_from_raw()
            logger.info("Pipeline initialized from raw data")
        
        logger.info(f"ML NER: {'Enabled' if self.ner_module else 'Disabled'}")
        logger.info(f"Embeddings: {'Enabled' if self.embedding_gen else 'Disabled'}")
    
    def _try_load_from_cache(self) -> bool:
        """
        Attempt to load all data from the pre-built cache.
        Returns True on success, False if any cache file is missing.
        """
        cache_dir = Path(self.config.get('cache', 'directory', default='cache'))
        embeddings_path = cache_dir / self.config.get(
            'cache', 'embeddings_file', default='medicine_embeddings.safetensors')
        medicine_data_path = cache_dir / self.config.get(
            'cache', 'medicine_data_file', default='medicine_data.json')
        drugbank_index_path = cache_dir / self.config.get(
            'cache', 'drugbank_index_file', default='drugbank_index.json.gz')
        medicine_synonyms_path = cache_dir / self.config.get(
            'cache', 'medicine_synonyms_file', default='medicine_synonyms.json')

        # All three main files must exist
        if not (embeddings_path.exists() and medicine_data_path.exists()
                and drugbank_index_path.exists()):
            return False
        
        if not (SAFETENSORS_AVAILABLE and TRANSFORMERS_AVAILABLE):
            logger.warning("safetensors or transformers not available — cannot load cache")
            return False
        
        try:
            t0 = time.time()
            
            # 1. Load medicine data (names)
            logger.info("Loading medicine data from cache...")
            with open(medicine_data_path, 'r', encoding='utf-8') as f:
                medicine_cache = json.load(f)
            medicine_names = medicine_cache['medicine_names']
            print(f"[SUCCESS] Loaded {len(medicine_names)} medicines from cache")
            
            # 2. Load embeddings from safetensors
            logger.info("Loading embeddings from safetensors cache...")
            tensors = safetensors_load_file(str(embeddings_path))
            embeddings_tensor = tensors['embeddings']
            # Convert to numpy for compatibility with existing code
            precomputed_embeddings = embeddings_tensor.numpy()
            print(f"[SUCCESS] Loaded embeddings {precomputed_embeddings.shape} from cache")
            
            # 3. Load DrugBank index
            logger.info("Loading DrugBank index from cache...")
            with gzip.open(str(drugbank_index_path), 'rt', encoding='utf-8') as f:
                drug_index = json.load(f)
            print(f"[SUCCESS] Loaded {len(drug_index)} drug index entries from cache")

            # Load synonyms if available
            synonym_map = {}
            if medicine_synonyms_path.exists():
                logger.info("Loading medicine synonyms from cache...")
                with open(medicine_synonyms_path, 'r', encoding='utf-8') as f:
                    synonym_map = json.load(f)
                print(f"[SUCCESS] Loaded {len(synonym_map)} synonyms from cache")
            from clinical_copilot.brands import apply_brand_lexicon
            n_brands = apply_brand_lexicon(synonym_map)
            if n_brands:
                print(f"[SUCCESS] Merged {n_brands} Indian brand aliases")

            # 3b. Load alias embeddings for synonym-marginalized retrieval
            # (idea #2). Optional: older caches without these files still work.
            alias_texts, alias_embeddings, alias_concepts = None, None, None
            alias_embeddings_path = cache_dir / self.config.get(
                'cache', 'alias_embeddings_file', default='alias_embeddings.safetensors')
            alias_data_path = cache_dir / self.config.get(
                'cache', 'alias_data_file', default='alias_data.json')
            if alias_embeddings_path.exists() and alias_data_path.exists():
                logger.info("Loading alias embeddings from cache...")
                alias_tensors = safetensors_load_file(str(alias_embeddings_path))
                alias_embeddings = alias_tensors['embeddings'].numpy()
                with open(alias_data_path, 'r', encoding='utf-8') as f:
                    alias_data = json.load(f)
                alias_texts = alias_data['alias_texts']
                alias_concepts = alias_data['alias_concepts']
                if not (len(alias_texts) == len(alias_concepts)
                        == alias_embeddings.shape[0]):
                    logger.warning("Alias cache length mismatch; ignoring alias pool.")
                    alias_texts, alias_embeddings, alias_concepts = None, None, None
                else:
                    print(f"[SUCCESS] Loaded {len(alias_texts)} alias embeddings "
                          f"{alias_embeddings.shape} from cache")
            else:
                logger.info("Alias embedding cache not found; using generics-only retrieval.")

            # 3c. Load TF-IDF index for sparse retrieval (idea #9). Optional.
            tfidf_vectorizer, tfidf_matrix = None, None
            from clinical_copilot.tfidf_index import load_tfidf_index
            tfidf_vectorizer, tfidf_matrix = load_tfidf_index(str(cache_dir))
            if tfidf_vectorizer is not None:
                print(f"[SUCCESS] Loaded TF-IDF index {tfidf_matrix.shape} from cache")
            else:
                logger.info("TF-IDF cache not found; dense-only retrieval.")

            # 4. Wire up components
            self.extractor = EntityExtractor(
                medicine_names=medicine_names,
                config=self.config,
                ner_module=self.ner_module,
                embedding_gen=self.embedding_gen,
                precomputed_embeddings=precomputed_embeddings,
                synonym_map=synonym_map,
                alias_texts=alias_texts,
                alias_embeddings=alias_embeddings,
                alias_concepts=alias_concepts,
                tfidf_vectorizer=tfidf_vectorizer,
                tfidf_matrix=tfidf_matrix,
            )
            self.detector = InteractionDetector(
                prebuilt_drug_index=drug_index,
                config=self.config
            )
            
            elapsed = time.time() - t0
            print(f"[SUCCESS] All cache loaded in {elapsed:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Cache loading failed: {e} — falling back to raw data")
            return False
    
    def _load_from_raw(self):
        """Original loading path: read JSON and extract medicine names/compute embeddings."""
        self.loader = MedicineDataLoader(self.config)
        json_data = self.loader.load_json_data()

        drugbank = json_data.get('{http://www.drugbank.ca}drugbank', {})
        drugs = drugbank.get('{http://www.drugbank.ca}drug', [])
        medicine_names = [drug.get('{http://www.drugbank.ca}name', '').lower() for drug in drugs if drug.get('{http://www.drugbank.ca}name', '')]

        from clinical_copilot.brands import apply_brand_lexicon
        synonym_map = {}
        apply_brand_lexicon(synonym_map)

        self.extractor = EntityExtractor(
            medicine_names=medicine_names,
            config=self.config,
            ner_module=self.ner_module,
            embedding_gen=self.embedding_gen,
            synonym_map=synonym_map
        )
        self.detector = InteractionDetector(json_data, config=self.config)
    
    def process_input(self, input_text: str, progress=None) -> Dict[str, Any]:
        """
        Process prescription text through complete ML-enhanced pipeline.
        
        Args:
            input_text: Raw prescription text (OCR/ASR output)
            progress: Optional callable receiving phase status messages
            
        Returns:
            Structured output with entities, mappings, interactions, confidence scores
        """
        def notify(message: str) -> None:
            if progress:
                progress(message)

        pipeline_start = time.time()
        logger.info("-"*70)
        logger.info("PROCESSING INPUT WITH ML/DL PIPELINE")
        logger.info("-"*70)
        logger.info(f"Input: {input_text[:100]}...")
        
        notify("Phase 1/4 — Extracting medicine entities…")
        logger.info("[PHASE 1] Extracting entities with ML/DL...")
        extraction_results = self.extractor.extract_entities(input_text)
        entities = extraction_results['entities']
        entity_confidences = extraction_results['confidence_scores']
        
        logger.info(f"  → Identified {len(entities)} medicines using {extraction_results['methods_used']}")
        logger.info(f"  → Average confidence: {extraction_results.get('average_confidence', 0):.2f}")

        notify("Phase 2/4 — Formatting medicine data…")
        medicine_names = [e['name'] if isinstance(e, dict) else str(e) for e in entities]
        mappings = [{"brand_name": name, "generic_names": [name]} for name in medicine_names]

        notify("Phase 3/4 — Checking drug interactions…")
        logger.info("[PHASE 2] Checking for drug-drug interactions...")
        interactions = self.detector.check_interactions(medicine_names)

        if interactions:
            logger.info(f"  WARNING: {len(interactions)} potential interaction(s) detected!")
            for interaction in interactions:
                logger.info(f"     * {interaction['drug_a']} <-> {interaction['drug_b']} "
                          f"(confidence: {interaction.get('detection_confidence', 0):.2f})")
        else:
            logger.info("  No interactions detected")
        
        notify("Phase 4/4 — Computing confidence scores…")
        logger.info("[PHASE 4] Computing confidence scores and uncertainty estimates...")
        
        overall_confidence = self.confidence_estimator.estimate_confidence(entity_confidences)
        
        output = {
            'input_text': input_text,
            'identified_medicines': entities,
            'entity_confidence_scores': entity_confidences,
            'extraction_methods': extraction_results['methods_used'],
            'brand_generic_mappings': mappings,
            'drug_interactions': interactions,
            'confidence_metrics': {
                'entity_extraction': overall_confidence,
                'interaction_detection': np.mean([i.get('detection_confidence', 0) for i in interactions]) if interactions else 1.0,
                'overall_confidence': (overall_confidence['confidence'] + 
                                     (1.0 if not interactions else np.mean([i.get('detection_confidence', 0) for i in interactions]))) / 2
            },
            'metadata': {
                'total_medicines': len(entities),
                'total_interactions': len(interactions),
                'has_safety_concerns': len(interactions) > 0,
                'ml_enabled': self.ner_module is not None,
                'embeddings_enabled': self.embedding_gen is not None,
                'low_confidence_entities': sum(1 for c in entity_confidences if c < 0.7)
            }
        }
        
        total_time = time.time() - pipeline_start
        logger.info(f"[COMPLETE] Processing finished in {total_time:.2f}s. Overall confidence: {output['confidence_metrics']['overall_confidence']:.2f}")
        
        return output

    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract prescription text from an image using the local OCR LLM (Ollama).

        Args:
            image_path: Path to prescription image file

        Returns:
            Dict with OCR status and extracted text
        """
        file_path = Path(image_path)
        if not file_path.exists():
            return {
                'success': False,
                'text': '',
                'error': f'Image file not found: {image_path}',
            }

        try:
            prompt = (
                "You are an OCR assistant for handwritten and typed medical prescriptions. "
                "Extract only the visible prescription text from this image. "
                "Do not summarize, do not interpret, and do not add medical advice. "
                "Preserve line breaks and medicine formatting as closely as possible."
            )

            retries = max(self.ocr_max_retries, 0)
            response = None
            for attempt in range(retries + 1):
                elapsed = time.time() - self.ocr_last_request_epoch
                if elapsed < max(self.ocr_min_request_interval_seconds, 0.0):
                    time.sleep(max(self.ocr_min_request_interval_seconds, 0.0) - elapsed)

                try:
                    response = local_llm.query_local_llm(prompt=prompt, image_paths=[file_path])
                    self.ocr_last_request_epoch = time.time()
                    break
                except Exception:
                    self.ocr_last_request_epoch = time.time()

                    if attempt < retries:
                        base = max(self.ocr_backoff_base_seconds, 0.1)
                        cap = max(self.ocr_backoff_max_seconds, base)
                        delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, base * 0.25)
                        time.sleep(delay)
                        continue

                    raise

            if response is None:
                return {
                    'success': False,
                    'text': '',
                    'error': 'OCR did not return a response.',
                }

            text = response.strip() if response else ''
            if not text:
                return {
                    'success': False,
                    'text': '',
                    'error': 'OCR completed but no text was extracted from image.',
                }

            return {
                'success': True,
                'text': text,
                'error': '',
                'model': 'local_llm',
            }
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'error': f'OCR failed: {e}',
            }
    
    def generate_clinical_report(self, processed_output: Dict[str, Any]) -> str:
        """
        Generate human-readable clinical report with confidence scores.
        
        Args:
            processed_output: Output from process_input()
            
        Returns:
            Formatted clinical report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("CLINICAL COPILOT - ML-ENHANCED PRESCRIPTION ANALYSIS")
        report.append("="*70 + "\n")
        
        report.append("IDENTIFIED MEDICATIONS:")
        report.append("-" * 70)
        
        for idx, medicine in enumerate(processed_output['identified_medicines']):
            name = medicine.get('name', '') if isinstance(medicine, dict) else medicine
            confidence = processed_output['entity_confidence_scores'][idx] if idx < len(processed_output['entity_confidence_scores']) else 0.0
            confidence_indicator = "OK" if confidence >= 0.7 else "WARN"
            
            report.append(f"\n{confidence_indicator} Medication: {str(name).title()} (Confidence: {confidence:.2f})")
        
        report.append("\n\n ML/DL ANALYSIS:")
        report.append("-" * 70)
        report.append(f"Extraction Methods: {', '.join(processed_output.get('extraction_methods', []))}")
        report.append(f"ML NER Enabled: {'Yes' if processed_output['metadata']['ml_enabled'] else 'No'}")
        report.append(f"Semantic Embeddings: {'Yes' if processed_output['metadata']['embeddings_enabled'] else 'No'}")
        report.append(f"Low Confidence Entities: {processed_output['metadata']['low_confidence_entities']}")
        
        report.append("\n\n SAFETY ANALYSIS:")
        report.append("-" * 70)
        
        interactions = processed_output['drug_interactions']
        if interactions:
            report.append(f"\nWARNING: {len(interactions)} Drug-Drug Interaction(s) Detected!\n")
            
            for idx, interaction in enumerate(interactions, 1):
                report.append(f"{idx}. {interaction['drug_a']} <-> {interaction['drug_b']}")
                report.append(f"   Description: {interaction['description']}")
                report.append(f"   DrugBank ID: {interaction['drugbank_id']}")
                report.append(f"   Detection Confidence: {interaction.get('detection_confidence', 0):.2f}\n")
        else:
            report.append("\n No drug-drug interactions detected.")
        
        report.append("\n\n CONFIDENCE METRICS:")
        report.append("-" * 70)
        conf_metrics = processed_output['confidence_metrics']
        report.append(f"Entity Extraction Confidence: {conf_metrics['entity_extraction']['confidence']:.2f}")
        report.append(f"Interaction Detection Confidence: {conf_metrics['interaction_detection']:.2f}")
        report.append(f"Overall System Confidence: {conf_metrics['overall_confidence']:.2f}")
        report.append(f"Uncertainty (Std Dev): {conf_metrics['entity_extraction']['std']:.2f}")
        
        report.append("\n\n SUMMARY:")
        report.append("-" * 70)
        report.append(f"Total Medications: {processed_output['metadata']['total_medicines']}")
        report.append(f"Total Interactions: {processed_output['metadata']['total_interactions']}")
        report.append(f"Safety Concerns: {'YES' if processed_output['metadata']['has_safety_concerns'] else 'NO'}")
        report.append(f"Requires Review: {'YES' if processed_output['metadata']['low_confidence_entities'] > 0 or processed_output['metadata']['has_safety_concerns'] else 'NO'}")
        
        report.append("\n" + "="*70 + "\n")
        
        return "\n".join(report)


def load_test_scenarios(test_config_path: str = "configs/test_config.yaml") -> List[Dict[str, Any]]:
    """
    Load test scenarios from configuration file.

    Args:
        test_config_path: Path to test configuration YAML

    Returns:
        List of test scenario dictionaries
    """
    import os

    candidates = [test_config_path]
    if test_config_path.startswith("configs/"):
        candidates.append(os.path.basename(test_config_path))
    for candidate in candidates:
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                test_config = yaml.safe_load(f)
            return test_config.get('test_scenarios', [])
        except FileNotFoundError:
            continue
    logger.warning(f"Test config {test_config_path} not found. Using default test.")
    return [{
        'id': 'default_test',
        'name': 'Default Test',
        'input_text': 'Patient prescribed Dolo 650 twice daily'
    }]
