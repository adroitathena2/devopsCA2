"""
Compatibility shim for the original monolithic clinical_copilot_backend module.

The implementation now lives in the clinical_copilot package. This module
re-exports the public API so existing imports (build_cache.py,
evaluate_ddi_system.py, tune_gemini_model.py, tests, UI) keep working.
"""

from clinical_copilot import (
    ConfigLoader,
    BioNERModule,
    SemanticEmbeddingGenerator,
    ConfidenceEstimator,
    MedicineDataLoader,
    EntityExtractor,
    InteractionDetector,
    ClinicalCopilotPipeline,
    resolve_torch_dtype,
    load_test_scenarios,
    TRANSFORMERS_AVAILABLE,
    GLINER_AVAILABLE,
    FLASHTEXT_AVAILABLE,
    SAFETENSORS_AVAILABLE,
    CUSTOM_MODEL_AVAILABLE,
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

if __name__ == "__main__":
    from clinical_copilot.cli import main

    main()
