"""
Command-line scenario runner for the Clinical Copilot backend.
"""

import json
import logging
import traceback

import numpy as np

from clinical_copilot import paths
from clinical_copilot.pipeline import ClinicalCopilotPipeline, load_test_scenarios

logger = logging.getLogger(__name__)


def main() -> None:
    paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    copilot = ClinicalCopilotPipeline(config_path="configs/config.yaml")

    test_scenarios = load_test_scenarios("configs/test_config.yaml")
    
    logger.info(f"\nRunning {len(test_scenarios)} test scenario(s)...")
    
    for scenario in test_scenarios:
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: {scenario['name']}")
        logger.info(f"{'='*70}")
        
        try:
            result = copilot.process_input(scenario['input_text'])
            report = copilot.generate_clinical_report(result)
            print(report)
            
            output_filename = paths.OUTPUTS_DIR / f"results_{scenario['id']}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                def convert_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_types(i) for i in obj]
                    return obj
                
                json.dump(convert_types(result), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_filename}")
            
        except Exception as e:
            logger.error(f"Test {scenario['name']} failed: {e}")
            traceback.print_exc()
    
    logger.info("\n[INFO] All tests completed!")
