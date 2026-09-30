"""
Hyperparameter tuning for Gemini-based DDI severity classification.

This script tunes only Gemini severity settings while keeping the same backend logic.
It writes results to outputs/tuning_results/.

Run with uv:
    uv run tune_gemini_model.py
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse
import itertools
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from clinical_copilot import paths, severity
from clinical_copilot_backend import ClinicalCopilotPipeline


def max_severity_label(interactions: List[Dict[str, Any]]) -> str:
    return severity.max_label(
        i.get("ground_truth_severity") or i.get("severity_categorical")
        for i in interactions
    )


class GeminiTuner:
    def __init__(self, dataset_path: str, validation_size: int = 30):
        self.dataset_path = dataset_path
        self.validation_size = validation_size
        self.output_dir = paths.OUTPUTS_DIR / "tuning_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = ClinicalCopilotPipeline(config_path="configs/config.yaml")

    def load_validation_subset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Prefer interaction-containing rows for severity tuning.
        interaction_rows = [x for x in data if x.get("has_interaction") and x.get("interactions")]
        subset = interaction_rows[: self.validation_size]
        return subset

    def evaluate_config(self, config: Dict[str, Any], subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.pipeline.detector.update_gemini_config(config)

        y_true: List[int] = []
        y_pred: List[int] = []

        for row in subset:
            gt_label = max_severity_label(row.get("interactions", []))
            gt_class = severity.class_index(gt_label)

            # Use known drug names from synthetic dataset for consistent interaction lookup.
            predicted_interactions = self.pipeline.detector.check_interactions(row.get("drugs", []))
            pred_label = max_severity_label(predicted_interactions)
            pred_class = severity.class_index(pred_label)

            y_true.append(gt_class)
            y_pred.append(pred_class)

        y_true_arr = np.array(y_true, dtype=int)
        y_pred_arr = np.array(y_pred, dtype=int)

        labels = list(range(len(severity.OUTCOME_LABELS)))

        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, labels=labels, zero_division=0
        )
        f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

        result = {
            **config,
            "samples": len(subset),
            "f1_weighted": float(f1_weighted),
        }
        for i, name in enumerate(severity.OUTCOME_LABELS):
            key = name.lower().replace(" ", "_")
            result[f"f1_{key}"] = float(f1_per_class[i])
            result[f"recall_{key}"] = float(recall[i])
            result[f"precision_{key}"] = float(precision[i])
        return result

    def run_grid_search(self) -> Dict[str, Any]:
        subset = self.load_validation_subset()
        print(f"Validation subset size: {len(subset)}")

        severity_models = ["gemini-2.5-flash", "gemini-1.5-flash"]
        severity_temperatures = [0.0, 0.2, 0.5]
        severity_top_p = [0.8, 0.9, 1.0]

        results: List[Dict[str, Any]] = []
        all_configs = list(itertools.product(severity_models, severity_temperatures, severity_top_p))

        for idx, (model, temp, top_p) in enumerate(all_configs, start=1):
            cfg = {
                "enabled": True,
                "severity_model": model,
                "reasoning_model": "gemini-2.5-flash",
                "severity_temperature": temp,
                "severity_top_p": top_p,
                "reasoning_temperature": 0.2,
                "reasoning_top_p": 0.9,
            }
            print(f"[{idx}/{len(all_configs)}] Testing: {cfg}")
            scored = self.evaluate_config(cfg, subset)
            results.append(scored)

        df = pd.DataFrame(results).sort_values(by="f1_weighted", ascending=False)
        grid_path = self.output_dir / "grid_search_results.csv"
        df.to_csv(grid_path, index=False)

        best = df.iloc[0].to_dict()
        best_config = {
            "enabled": bool(best.get("enabled", True)),
            "severity_model": best["severity_model"],
            "reasoning_model": best["reasoning_model"],
            "severity_temperature": float(best["severity_temperature"]),
            "severity_top_p": float(best["severity_top_p"]),
            "reasoning_temperature": float(best["reasoning_temperature"]),
            "reasoning_top_p": float(best["reasoning_top_p"]),
        }

        best_path = self.output_dir / "best_config.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)

        summary = {
            "search_space": {
                "severity_model": severity_models,
                "severity_temperature": severity_temperatures,
                "severity_top_p": severity_top_p,
            },
            "trials": len(all_configs),
            "validation_samples": len(subset),
            "best_f1_weighted": float(best["f1_weighted"]),
            "best_config": best_config,
            "output_files": {
                "grid_search_results": str(grid_path),
                "best_config": str(best_path),
            },
        }

        summary_path = self.output_dir / "tuning_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Gemini hyperparameters for DDI severity")
    parser.add_argument("--dataset", default="configs/test_prescriptions.json")
    parser.add_argument("--validation-size", type=int, default=30)
    args = parser.parse_args()

    tuner = GeminiTuner(dataset_path=args.dataset, validation_size=args.validation_size)
    summary = tuner.run_grid_search()

    print("=" * 70)
    print("Tuning complete")
    print(f"Best weighted F1: {summary['best_f1_weighted']:.4f}")
    print(f"Best config saved to: {summary['output_files']['best_config']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
