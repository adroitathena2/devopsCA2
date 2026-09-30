import sys
from clinical_copilot_backend import ClinicalCopilotPipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = ClinicalCopilotPipeline()
    return _pipeline

NON_MEDICINE_WORDS = {
    'prescribe', 'prescribed', 'patient', 'chart', 'plan', 'management',
    'diabetes', 'daily', 'for', 'and', 'the',
}

def medicine_names(medicines: list) -> list:
    return [
        str(m.get('name', '')) if isinstance(m, dict) else str(m)
        for m in medicines
    ]

def assert_no_stopwords(medicines: list, scenario: str):
    bad = [m for m in medicine_names(medicines) if m.strip().lower() in NON_MEDICINE_WORDS]
    assert not bad, f"[{scenario}] Non-medicine words found in results: {bad}"

def assert_entity_found(medicines: list, expected: str, scenario: str):
    expected_lower = expected.lower()
    found = any(expected_lower in m.lower() for m in medicine_names(medicines))
    assert found, f"[{scenario}] Expected to find '{expected}' in: {medicine_names(medicines)}"

def test_scenario_1_simple_prescription():
    print("\n" + "="*70)
    print("TEST SCENARIO 1: Simple Prescription")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Prescribe Dolo 650 twice daily for fever"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 1")
    assert_entity_found(medicines, "Acetaminophen", "Scenario 1")
    return result

def test_scenario_2_multiple_drugs():
    print("\n" + "="*70)
    print("TEST SCENARIO 2: Multiple Drugs with Potential DDI")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Patient Chart:\n1. Aspirin 75mg - once daily (cardiovascular protection)\n2. Clopidogrel 75mg - once daily (antiplatelet)\n3. Atorvastatin 10mg - bedtime (cholesterol)"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    interactions = result['drug_interactions']
    assert_no_stopwords(medicines, "Scenario 2")
    assert_entity_found(medicines, "Acetylsalicylic Acid", "Scenario 2")
    assert_entity_found(medicines, "Clopidogrel", "Scenario 2")
    assert_entity_found(medicines, "Atorvastatin", "Scenario 2")
    assert len(interactions) > 0, "[Scenario 2] Expected drug interactions, got 0"
    return result

def test_scenario_3_ocr_errors():
    print("\n" + "="*70)
    print("TEST SCENARIO 3: OCR Error Handling")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Prescribe Dlo 650 and Metfrmn 500mg"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 3")
    return result

def test_scenario_4_combination_drugs():
    print("\n" + "="*70)
    print("TEST SCENARIO 4: Combination Drugs")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Ramistar-AM 5 Tablet once daily for hypertension"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 4")
    assert_entity_found(medicines, "Amlodipine", "Scenario 4")
    assert_entity_found(medicines, "Ramipril", "Scenario 4")
    return result

def test_scenario_5_unknown_drug():
    print("\n" + "="*70)
    print("TEST SCENARIO 5: Unknown Drug Handling")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Prescribe XYZ Brand 100mg (hypothetical drug)"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 5")
    return result

def test_scenario_6_diabetes_prescription():
    print("\n" + "="*70)
    print("TEST SCENARIO 6: Diabetes Management Prescription")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "Diabetes Management Plan:\n- Metformin 500mg (twice daily with meals)\n- Glimepiride 2mg (before breakfast)\n- Aspirin 75mg (once daily - cardiovascular protection)"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 6")
    assert_entity_found(medicines, "Metformin", "Scenario 6")
    assert_entity_found(medicines, "Glimepiride", "Scenario 6")
    assert_entity_found(medicines, "Acetylsalicylic Acid", "Scenario 6")
    return result

def test_scenario_7_interaction_detection():
    print("\n" + "="*70)
    print("TEST SCENARIO 7: Interaction Detection Focus")
    print("="*70)
    pipeline = get_pipeline()
    input_text = "1. Warfarin 5mg daily\n2. Aspirin 75mg daily"
    result = pipeline.process_input(input_text)
    print(pipeline.generate_clinical_report(result))
    medicines = result['identified_medicines']
    assert_no_stopwords(medicines, "Scenario 7")
    return result

def test_scenario_9_stopword_filtering():
    print("\n" + "="*70)
    print("TEST SCENARIO 9: Stopword Filtering")
    print("="*70)
    pipeline = get_pipeline()
    test_inputs = [
        "Prescribe Dolo 650 twice daily for fever",
        "Patient Chart: Aspirin 75mg once daily",
        "Diabetes Management Plan: Metformin 500mg",
    ]
    for text in test_inputs:
        result = pipeline.process_input(text)
        medicines = result['identified_medicines']
        assert_no_stopwords(medicines, f"Stopword test: '{text[:40]}...'")

def compare_results(results_list, scenario_names):
    print("\n" + "="*70)
    print("COMPARISON REPORT - ALL TEST SCENARIOS")
    print("="*70 + "\n")
    comparison = []
    for idx, (result, name) in enumerate(zip(results_list, scenario_names), 1):
        comparison.append({
            'scenario': name,
            'medicines_identified': result['metadata']['total_medicines'],
            'interactions_found': result['metadata']['total_interactions'],
            'has_safety_concerns': result['metadata']['has_safety_concerns']
        })
    print(f"{'Scenario':<40} {'Medicines':<12} {'Interactions':<15} {'Safety Alert'}")
    print("-" * 70)
    for item in comparison:
        alert = "YES" if item['has_safety_concerns'] else "OK"
        print(f"{item['scenario']:<40} {item['medicines_identified']:<12} {item['interactions_found']:<15} {alert}")

def run_all_tests():
    print("\n" + "#"*70)
    print("# AI CLINICAL COPILOT - COMPREHENSIVE TEST SUITE")
    print("#"*70)
    results, scenario_names = [], []
    passed, failed = 0, 0
    test_functions = [
        (test_scenario_1_simple_prescription, "Simple Prescription (Dolo 650)"),
        (test_scenario_2_multiple_drugs, "Multiple Drugs with DDI"),
        (test_scenario_3_ocr_errors, "OCR Error Handling"),
        (test_scenario_4_combination_drugs, "Combination Drug"),
        (test_scenario_5_unknown_drug, "Unknown Drug"),
        (test_scenario_6_diabetes_prescription, "Diabetes Management"),
        (test_scenario_7_interaction_detection, "Interaction Detection"),
        (test_scenario_9_stopword_filtering, "Stopword Filtering"),
    ]
    for test_func, name in test_functions:
        try:
            result = test_func()
            if result is not None:
                results.append(result)
                scenario_names.append(name)
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {name}: {str(e)}\n")
            failed += 1
            results.append({'metadata': {'total_medicines': 0, 'total_interactions': 0, 'has_safety_concerns': False}})
            scenario_names.append(f"{name} (FAILED)")
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {str(e)}\n")
            failed += 1
            results.append({'metadata': {'total_medicines': 0, 'total_interactions': 0, 'has_safety_concerns': False}})
            scenario_names.append(f"{name} (ERROR)")
    if results and scenario_names:
        compare_results(results, scenario_names)
    print("\n" + "#"*70)
    print(f"# TEST SUITE COMPLETE: {passed} passed, {failed} failed")
    print("#"*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_number = sys.argv[1]
        test_map = {
            '1': test_scenario_1_simple_prescription, '2': test_scenario_2_multiple_drugs,
            '3': test_scenario_3_ocr_errors, '4': test_scenario_4_combination_drugs,
            '5': test_scenario_5_unknown_drug, '6': test_scenario_6_diabetes_prescription,
            '7': test_scenario_7_interaction_detection, '9': test_scenario_9_stopword_filtering,
        }
        if test_number in test_map:
            test_map[test_number]()
    else:
        run_all_tests()
