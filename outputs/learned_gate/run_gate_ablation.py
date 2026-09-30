"""Gate ablation driver: SEVERITY_GATE in {learned, none} on test_prescriptions.json.

Mirrors outputs/rxbench/run_rxbench_eval.py (programmatic DDIEvaluator,
per-run output dir so production outputs/Ensemble is never clobbered).
Env SEVERITY_GATE must be set before the evaluator (and its
InteractionDetector) is constructed — enforced below via --gate.

Usage: uv run outputs/learned_gate/run_gate_ablation.py --gate learned
Writes outputs/learned_gate/eval_<gate>/ (metrics.json, summary, checkpoint).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["learned", "none"])
    ap.add_argument("--dataset", default="test_prescriptions.json")
    a = ap.parse_args()

    os.environ["SEVERITY_GATE"] = a.gate
    from evaluate_ddi_system import DDIEvaluator  # noqa: E402

    out = ROOT / "outputs" / "learned_gate" / f"eval_{a.gate}"
    out.mkdir(parents=True, exist_ok=True)
    ev = DDIEvaluator(mode="ensemble")
    ev.output_dir = out
    gate_seen = getattr(getattr(ev, "pipeline", None), "detector", None)
    print(f"detector SEVERITY_GATE={getattr(gate_seen, 'SEVERITY_GATE', '?')}",
          flush=True)
    rx = ev.load_dataset(a.dataset)
    rows = ev.run_inference(
        rx, checkpoint_meta={"dataset": a.dataset, "mode": "ensemble",
                             "split": f"gate_{a.gate}", "total": len(rx)})
    summary = ev.evaluate(rows)
    ev.save_outputs(rows, summary)
    req = summary["metrics_required"]
    sev = summary["per_class_metrics"].get("Severe", {})
    mod = summary["per_class_metrics"].get("Moderate", {})
    mild = summary["per_class_metrics"].get("Mild", {})
    rec = {"gate": a.gate, "n": summary["samples"],
           "accuracy": round(req["accuracy"], 4),
           "macro_f1": round(req["macro_f1"], 4),
           "binary_f1": round(req["f1_score"], 4),
           "auc_roc": round(req["auc_roc"], 4),
           "sens_recall": round(req["sensitivity_recall"], 4),
           "mild_f1": round(mild.get("f1", 0.0), 4),
           "mod_f1": round(mod.get("f1", 0.0), 4),
           "severe_f1": round(sev.get("f1", 0.0), 4),
           "severe_recall": round(sev.get("recall", 0.0), 4),
           "severe_precision": round(sev.get("precision", 0.0), 4)}
    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"GATE {a.gate}: " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
