"""
Named entity recognition for clinical text (BioBERT / GLiNER).
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict

import numpy as np

from clinical_copilot.config import ConfigLoader

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] Transformers not available. Install with: uv pip install transformers sentence-transformers torch")

logger = logging.getLogger(__name__)


class BioNERModule:
    """
    Named Entity Recognition using BioBERT/ClinicalBERT for medical text.
    Extracts: MEDICATION, DOSAGE, FREQUENCY, DURATION, ROUTE, CONDITION
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize BioBERT NER module.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.model_name = config.get('models', 'ner', 'model_name', 
                                     default='alvaroalon2/biobert_chemical_ner')
        self.fallback_model_name = config.get(
            'models', 'ner', 'fallback_model_name',
            default='d4data/biomedical-ner-all'
        )
        self.device = config.get('models', 'ner', 'device', default='cpu')
        self.confidence_threshold = config.get('models', 'ner', 'confidence_threshold', default=0.7)
        
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self.active_model_name = None
        self.method_name = 'regex_fallback'
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Transformers not available. NER will use fallback method.")
    
    def _initialize_model(self):
        """Initialize a token-classification NER model with safe fallback."""
        if GLINER_AVAILABLE:
            try:
                candidate = "E3-JSI/gliner-multi-med-ner-synthetic-v1"
                logger.info(f"Loading GLiNER model: {candidate}")
                self.model = GLiNER.from_pretrained(candidate)
                self.active_model_name = candidate
                self.method_name = 'gliner_ner'
                logger.info("GLiNER model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load GLiNER model: {e}. Falling back to pipeline.")
                
        candidates = [self.model_name]
        if self.fallback_model_name and self.fallback_model_name != self.model_name:
            candidates.append(self.fallback_model_name)

        for idx, candidate in enumerate(candidates):
            try:
                logger.info(f"Loading NER model: {candidate}")
                self.ner_pipeline = pipeline(
                    "ner",
                    model=candidate,
                    tokenizer=candidate,
                    device=-1 if self.device == 'cpu' else 0,
                    aggregation_strategy="simple"
                )
                self.active_model_name = candidate
                self.method_name = 'biobert_ner' if idx == 0 else 'biomedical_ner_fallback'
                logger.info(f"NER model loaded successfully: {candidate}")
                return
            except Exception as e:
                logger.warning(f"Failed to load NER model '{candidate}': {e}")

        logger.error("All configured NER models failed to load. Falling back to regex extraction.")
        self.ner_pipeline = None
        self.active_model_name = None
        self.method_name = 'regex_fallback'

    @staticmethod
    def _clean_entity_text(text: str) -> str:
        """Clean tokenizer artifacts from predicted entities."""
        if not text:
            return ""
        cleaned = text.replace('##', '').replace('_', ' ').strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip(" ,.;:-")

    @staticmethod
    def _normalize_entity_group(group: str) -> str:
        """Map model-specific entity tags to stable internal labels."""
        if not group:
            return 'UNKNOWN'

        tag = str(group).upper().strip()
        if tag.startswith('B-') or tag.startswith('I-'):
            tag = tag[2:]

        # Medication-centric normalization for biomedical NER checkpoints.
        if tag in {'DRUG', 'CHEMICAL', 'CHEM', 'MEDICATION', 'MED', 'TREATMENT'}:
            return 'MEDICATION'
        if 'DRUG' in tag or 'CHEM' in tag or 'MED' in tag:
            return 'MEDICATION'

        if 'DOSE' in tag:
            return 'DOSAGE'
        if 'FREQ' in tag:
            return 'FREQUENCY'
        if 'DUR' in tag:
            return 'DURATION'
        if 'ROUTE' in tag:
            return 'ROUTE'
        if 'DISEASE' in tag or 'CONDITION' in tag or 'SYMPTOM' in tag:
            return 'CONDITION'

        return tag
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract medical entities from text using BioBERT.
        
        Args:
            text: Input clinical text
            
        Returns:
            Dictionary with entities and confidence scores
        """
        if self.method_name == 'gliner_ner' and self.model is not None:
            try:
                labels = ["medication", "drug", "medicine", "vitamin", "supplement"]
                entities = self.model.predict_entities(text, labels)
                grouped_entities = defaultdict(list)
                kept_scores = []
                
                for entity in entities:
                    score = float(entity.get('score', 0.0))
                    if score >= self.confidence_threshold:
                        cleaned_text = self._clean_entity_text(entity.get('text', ''))
                        if not cleaned_text:
                            continue
                        entity_info = {
                            'text': cleaned_text,
                            'label': 'MEDICATION',
                            'confidence': score,
                            'start': entity.get('start', -1),
                            'end': entity.get('end', -1)
                        }
                        grouped_entities['MEDICATION'].append(entity_info)
                        kept_scores.append(score)
                
                overall_confidence = np.mean(kept_scores) if kept_scores else 0.0
                
                return {
                    'entities': dict(grouped_entities),
                    'all_entities': entities,
                    'confidence': float(overall_confidence),
                    'entity_count': sum(len(v) for v in grouped_entities.values()),
                    'method': self.method_name,
                    'active_model': self.active_model_name,
                }
            except Exception as e:
                logger.error(f"GLiNER extraction failed: {e}")
                return self._fallback_extraction(text)

        if not TRANSFORMERS_AVAILABLE or self.ner_pipeline is None:
            return self._fallback_extraction(text)
        
        try:
            entities = self.ner_pipeline(text)
            
            grouped_entities = defaultdict(list)
            kept_scores = []
            
            for entity in entities:
                score = float(entity.get('score', 0.0))
                if score >= self.confidence_threshold:
                    normalized_label = self._normalize_entity_group(entity.get('entity_group'))
                    cleaned_text = self._clean_entity_text(entity.get('word', ''))
                    if not cleaned_text:
                        continue

                    entity_info = {
                        'text': cleaned_text,
                        'label': normalized_label,
                        'confidence': score,
                        'start': entity.get('start', -1),
                        'end': entity.get('end', -1)
                    }
                    grouped_entities[normalized_label].append(entity_info)
                    kept_scores.append(score)
            
            overall_confidence = np.mean(kept_scores) if kept_scores else 0.0
            
            return {
                'entities': dict(grouped_entities),
                'all_entities': entities,
                'confidence': float(overall_confidence),
                'entity_count': sum(len(v) for v in grouped_entities.values()),
                'method': self.method_name,
                'active_model': self.active_model_name,
            }
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback to regex-based extraction if ML fails."""
        medication_pattern = r'\b[A-Z][a-z]+(?:\s+\d+(?:mg|mcg|g|ml)?)?\b'
        dosage_pattern = r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|capsules?)\b'
        frequency_pattern = r'\b(?:once|twice|thrice|\d+\s*times?)\s+(?:daily|per day|a day)\b'
        
        medications = re.findall(medication_pattern, text)
        dosages = re.findall(dosage_pattern, text, re.IGNORECASE)
        frequencies = re.findall(frequency_pattern, text, re.IGNORECASE)
        
        return {
            'entities': {
                'MEDICATION': [{'text': m, 'confidence': 0.5, 'label': 'MEDICATION'} for m in medications],
                'DOSAGE': [{'text': d, 'confidence': 0.5, 'label': 'DOSAGE'} for d in dosages],
                'FREQUENCY': [{'text': f, 'confidence': 0.5, 'label': 'FREQUENCY'} for f in frequencies]
            },
            'confidence': 0.5,
            'entity_count': len(medications) + len(dosages) + len(frequencies),
            'method': 'regex_fallback'
        }
