"""
Medicine database loading for the Clinical Copilot backend.
"""

import json
import logging
import os
import time
from typing import Any, Dict

from clinical_copilot.config import ConfigLoader

logger = logging.getLogger(__name__)


class MedicineDataLoader:
    """Handles loading and preprocessing of medicine databases."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.json_path = config.get('data', 'json_path', default='full_database.json')
        self.json_data = {}

    
    def load_json_data(self) -> Dict[str, Any]:
        """
        Load DrugBank JSON database.
        
        Returns:
            Parsed JSON dictionary
        """
        file_size_mb = 0
        try:
            file_size_mb = os.path.getsize(self.json_path) / (1024 * 1024)
        except OSError:
            pass
        
        print(f"[INFO] Loading JSON data ({file_size_mb:.0f} MB) - this may take a moment...")
        t0 = time.time()
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            
            elapsed = time.time() - t0
            drugbank = self.json_data.get('{http://www.drugbank.ca}drugbank', {})
            drugs = drugbank.get('{http://www.drugbank.ca}drug', [])
            
            print(f"[SUCCESS] Loaded {len(drugs)} drugs from DrugBank JSON in {elapsed:.1f}s")
            return self.json_data
            
        except FileNotFoundError:
            print(f"[ERROR] JSON file not found: {self.json_path}")
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {str(e)}")
            return {}
