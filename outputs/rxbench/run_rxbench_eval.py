"""RxBench eval driver (NO retraining, evaluator scoring logic untouched).

Uses DDIEvaluator from evaluate_ddi_system.py via its own load_dataset /
run_inference / evaluate / save_outputs code path. CLI equivalence note:
evaluate_ddi_system.py defines `--dataset` (default test_prescriptions.json)
and load_dataset() opens whatever path is given, so arbitrary bench files
are accepted; this driver calls the same functions programmatically with
per-split output dirs so the production outputs/Ensemble checkpoint and
metrics are never clobbered.

Writes everything under outputs/rxbench/.
Usage: uv run outputs/rxbench/run_rxbench_eval.py [--mode ensemble]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from evaluate_ddi_system import DDIEvaluator  # noqa: E402


class ResilientEvaluator(DDIEvaluator):
    """Same scoring/inference path; checkpoint saves retry on transient
    Windows file locks (WinError 5 from AV/indexer races)."""

    def _save_checkpoint(self, rows, meta=None):
        last = None
        for attempt in range(6):
            try:
                return super()._save_checkpoint(rows, meta)
            except PermissionError as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        # final fallback: direct (non-atomic) write, never touching evaluator logic
        with open(self._checkpoint_path(), "w", encoding="utf-8") as f:
            json.dump({"meta": meta or {}, "rows": rows}, f)
        if last is not None:
            print(f"[WARN] checkpoint direct-write after retries: {last}")

OUT = ROOT / "outputs" / "rxbench"
SPLITS = [
    ("clean", OUT / "split_clean.json"),
    ("brand", OUT / "split_brand.json"),
    ("L1", OUT / "split_L1.json"),
    ("L2", OUT / "split_L2.json"),
    ("L3", OUT / "split_L3.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ensemble", choices=["keyword", "ensemble"])
    args = ap.parse_args()

    evaluator = ResilientEvaluator(mode=args.mode)
    table = []
    for name, path in SPLITS:
        split_dir = OUT / f"eval_{name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        evaluator.output_dir = split_dir  # per-split checkpoints/outputs
        prescriptions = evaluator.load_dataset(str(path))
        print(f"\n===== split {name}: {len(prescriptions)} Rx from {path.name} =====",
              flush=True)
        rows = evaluator.run_inference(
            prescriptions,
            checkpoint_meta={"dataset": str(path), "mode": args.mode,
                             "split": name, "total": len(prescriptions)},
        )
        summary = evaluator.evaluate(rows)
        evaluator.save_outputs(rows, summary)
        req = summary["metrics_required"]
        drug = summary["drug_name_matching"]
        sev = summary["per_class_metrics"].get("Severe", {})
        table.append({
            "split": name,
            "n": summary["samples"],
            "drug_match_rate": round(drug["match_rate"], 4),
            "drug_matched": drug["matched"],
            "drug_total": drug["gt_total"],
            "binary_f1": round(req["f1_score"], 4),
            "auc_roc": round(req["auc_roc"], 4),
            "sensitivity_recall": round(req["sensitivity_recall"], 4),
            "specificity_tnr": round(req["specificity_tnr"], 4),
            "accuracy_4class": round(req["accuracy"], 4),
            "macro_f1_4class": round(req["macro_f1"], 4),
            "severe_recall": round(sev.get("recall", 0.0), 4),
            "severe_precision": round(sev.get("precision", 0.0), 4),
            "severe_f1": round(sev.get("f1", 0.0), 4),
            "severe_support": sev.get("support", 0),
            "avg_latency_s": round(summary["average_latency_seconds"], 4),
        })
        with open(split_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(table[-1], f, indent=2)

    base = table[0]
    for row in table:
        for k in ("drug_match_rate", "binary_f1", "severe_recall",
                  "auc_roc", "accuracy_4class", "macro_f1_4class"):
            row[f"d_{k}_vs_clean"] = round(row[k] - base[k], 4)
    with open(OUT / "metrics_table.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "per_split": table}, f, indent=2)
    with open(OUT / "metrics_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # degradation-curve data (long form: one row per split x metric)
    curve_metrics = ["drug_match_rate", "binary_f1", "severe_recall",
                     "auc_roc", "accuracy_4class", "macro_f1_4class"]
    curve = [{"split": r["split"], "metric": m, "value": r[m],
              "delta_vs_clean": r[f"d_{m}_vs_clean"]}
             for r in table for m in curve_metrics]
    with open(OUT / "degradation.json", "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
    with open(OUT / "degradation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "metric", "value",
                                          "delta_vs_clean"])
        w.writeheader()
        w.writerows(curve)
    print("\nPer-split metric table:")
    for r in table:
        print(f"  {r['split']:6s} drug_match={r['drug_match_rate']:.4f} "
              f"binF1={r['binary_f1']:.4f} sevRec={r['severe_recall']:.4f} "
              f"auc={r['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
