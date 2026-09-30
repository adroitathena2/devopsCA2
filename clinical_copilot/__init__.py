"""
AI Clinical Copilot - Backend Infrastructure with ML/DL/NLP
Processes multi-source data for medication entity extraction and DDI detection.
Includes: BioBERT NER, Semantic Embeddings, Confidence Scoring, Uncertainty Estimation
"""

from dotenv import load_dotenv

load_dotenv()

import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from clinical_copilot.config import ConfigLoader
from clinical_copilot.data import MedicineDataLoader
from clinical_copilot.ner import BioNERModule, GLINER_AVAILABLE, TRANSFORMERS_AVAILABLE
from clinical_copilot.embeddings import SemanticEmbeddingGenerator, resolve_torch_dtype
from clinical_copilot.extraction import EntityExtractor, FLASHTEXT_AVAILABLE
from clinical_copilot.ddi import InteractionDetector, CUSTOM_MODEL_AVAILABLE
from clinical_copilot.confidence import ConfidenceEstimator
from clinical_copilot.pipeline import (
    ClinicalCopilotPipeline,
    SAFETENSORS_AVAILABLE,
    load_test_scenarios,
)

__all__ = [
    'ConfigLoader',
    'BioNERModule',
    'SemanticEmbeddingGenerator',
    'ConfidenceEstimator',
    'MedicineDataLoader',
    'EntityExtractor',
    'InteractionDetector',
    'ClinicalCopilotPipeline',
    'resolve_torch_dtype',
    'load_test_scenarios',
    'TRANSFORMERS_AVAILABLE',
    'GLINER_AVAILABLE',
    'FLASHTEXT_AVAILABLE',
    'SAFETENSORS_AVAILABLE',
    'CUSTOM_MODEL_AVAILABLE',
]
