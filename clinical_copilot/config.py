"""
Configuration loading for the Clinical Copilot backend.
"""

import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = self._resolve(config_path)
        self.config = self._load_config()

    @staticmethod
    def _resolve(path: str) -> str:
        """Prefer new configs/ location, fall back to legacy root location."""
        import os

        if os.path.exists(path):
            return path
        legacy = os.path.basename(path)
        if os.path.exists(legacy):
            return legacy
        return path
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            'data': {
                'json_path': 'full_database.json'
            },
            'models': {
                'ner': {
                    'model_name': 'alvaroalon2/biobert_chemical_ner',
                    'fallback_model_name': 'd4data/biomedical-ner-all',
                    'confidence_threshold': 0.7,
                    'device': 'cpu'
                },
                'embeddings': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'device': 'cpu',
                    'similarity_threshold': 0.75
                }
            },
            'entity_extraction': {
                'fuzzy_matching': {
                    'threshold': 0.75
                }
            },
            'pipeline': {
                'use_ml_ner': True,
                'use_embeddings': True
            },
            'gemini': {
                'enabled': False,
                'severity_model': 'gemini-2.5-flash',
                'reasoning_model': 'gemini-2.5-flash',
                'severity_temperature': 0.0,
                'severity_top_p': 0.9,
                'reasoning_temperature': 0.2,
                'reasoning_top_p': 0.9,
                'min_request_interval_seconds': 1.0,
                'max_retries': 3,
                'backoff_base_seconds': 1.0,
                'backoff_max_seconds': 30.0,
                'tuning_config_path': ''
            }
        }
    
    def get(self, *keys, default=None):
        """
        Get nested configuration value.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
