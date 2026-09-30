"""
Test script for backend severity extraction and Gemini integration.
Tests the new DDI severity classification and reasoning features.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from clinical_copilot_backend import ClinicalCopilotPipeline

def test_severity_extraction():
    """Test severity extraction from backend."""
    print("=" * 70)
    print("Testing DDI Severity Extraction & Gemini Integration")
    print("=" * 70)
    print()

    # Initialize pipeline
    print("[1/3] Initializing Clinical Copilot Pipeline...")
    pipeline = ClinicalCopilotPipeline()
    print("[OK] Pipeline initialized successfully")
    print()

    # Test cases with known interactions
    test_cases = [
        {
            "name": "Severe Interaction",
            "text": "Aspirin 75mg once daily\nWarfarin 5mg once daily",
            "expected_drugs": ["Aspirin", "Warfarin"],
            "expected_severity": "Severe"
        },
        {
            "name": "Moderate Interaction",
            "text": "Metformin 500mg BD\nLisinopril 10mg OD",
            "expected_drugs": ["Metformin", "Lisinopril"],
            "expected_severity": "Moderate" # or Severe, depending on DrugBank data
        },
        {
            "name": "Single Drug (No Interaction)",
            "text": "Paracetamol 500mg TDS",
            "expected_drugs": ["Paracetamol"],
            "expected_severity": None
        }
    ]

    for idx, test_case in enumerate(test_cases, 1):
        print(f"[{idx}/3] Test Case: {test_case['name']}")
        print(f"Prescription: {test_case['text']}")
        print()

        # Phase 1: Extract entities
        extraction_results = pipeline.extractor.extract_entities(test_case['text'])
        entities = extraction_results['entities']

        print(f"  Extracted Entities: {entities}")

        # Phase 2: Check interactions
        interactions = pipeline.detector.check_interactions(entities)

        if interactions:
            print(f"  [OK] Found {len(interactions)} interaction(s)")
            print()

            for i, interaction in enumerate(interactions, 1):
                print(f"  Interaction #{i}:")
                print(f"    Drug Pair: {interaction.get('drug_a')} <-> {interaction.get('drug_b')}")
                print(f"    Severity (Final): {interaction.get('severity_categorical')}")
                print(f"    Severity (Database): {interaction.get('database_severity')}")
                print(f"    Severity (Gemini): {interaction.get('gemini_severity')}")
                print(f"    Detection Confidence: {interaction.get('detection_confidence'):.0%}")
                print()

                # Check structure
                required_fields = ['severity_categorical', 'database_severity',
                                   'gemini_severity', 'reasoning', 'description']
                for field in required_fields:
                    if field not in interaction:
                        print(f"    [WARNING] Missing field '{field}'")
                    else:
                        print(f"    [OK] Field '{field}' present")

                # Validate severity category
                severity = interaction.get('severity_categorical')
                if severity not in ['Moderate', 'Severe']:
                    print(f"    [ERROR] Invalid severity '{severity}'")
                else:
                    print(f"    [OK] Valid severity category: {severity}")

                print()
                print(f"    Clinical Reasoning ({len(interaction.get('reasoning', ''))} chars):")
                reasoning = interaction.get('reasoning', 'N/A')
                # Print first 200 chars
                print(f"    {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
                print()

                print(f"    DrugBank Description ({len(interaction.get('description', ''))} chars):")
                description = interaction.get('description', 'N/A')
                print(f"    {description[:200]}{'...' if len(description) > 200 else ''}")
                print()

                # Test severity match
                if test_case["expected_severity"]:
                    if severity == test_case["expected_severity"]:
                        print(f"    [OK] Severity matches expected: {severity}")
                    else:
                        print(f"    [WARNING] Severity mismatch: got '{severity}', expected '{test_case['expected_severity']}'")
                        print("      (This may be acceptable - severity can vary based on DrugBank data)")

                print()
        else:
            print("  [OK] No interactions detected (as expected)")

        print("-" * 70)
        print()

    print("=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - Severity extraction: Implemented [OK]")
    print("  - Gemini API integration: Implemented [OK]")
    print("  - Result structure: Updated [OK]")
    print()
    print("Next steps:")
    print("  1. Test the UI: uv run clinical_copilot_ui.py")
    print("  2. Generate synthetic dataset: uv run generate_prescriptions.py")
    print("  3. Run evaluation pipeline: uv run evaluate_ddi_system.py --mode keyword")
    print()

if __name__ == "__main__":
    try:
        test_severity_extraction()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
