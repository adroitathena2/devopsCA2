"""Learned-gate x noise benchmark (completes comparison 3).

SEVERITY_GATE=learned on rxbench L1/L2/L3 (clean+learned already exists as
outputs/learned_gate/eval_learned; brand is severity-invariant so
brand+learned == clean+learned by construction). Reuses ResilientEvaluator
from run_rxbench_eval.py (Windows file-lock safe). No retraining.

Usage: uv run outputs/learned_gate/run_gate_bench.py
Writes outputs/learned_gate/bench_learned_{L1,L2,L3}/ + bench_learned_table.json
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs" / "rxbench"))
from run_rxbench_eval import ResilientEvaluator  # noqa: E402

os.environ["SEVERITY_GATE"] = "learned"

SPLITS = ["L1", "L2", "L3"]


def main():
    ev = ResilientEvaluator(mode="ensemble")
    det = getattr(getattr(ev, "pipeline", None), "detector", None)
    print(f"detector SEVERITY_GATE={getattr(det, 'SEVERITY_GATE', '?')}",
          flush=True)
    table = []
    for name in SPLITS:
        path = ROOT / "outputs" / "rxbench" / f"split_{name}.json"
        d = ROOT / "outputs" / "learned_gate" / f"bench_learned_{name}"
        d.mkdir(parents=True, exist_ok=True)
        ev.output_dir = d
        rx = ev.load_dataset(str(path))
        print(f"\n===== learned-gate split {name}: {len(rx)} Rx =====",
              flush=True)
        rows = ev.run_inference(
            rx, checkpoint_meta={"dataset": str(path), "mode": "ensemble",
                                 "split": f"learned_{name}",
                                 "total": len(rx)})
        summary = ev.evaluate(rows)
        ev.save_outputs(rows, summary)
        req = summary["metrics_required"]
        drug = summary["drug_name_matching"]
        sev = summary["per_class_metrics"].get("Severe", {})
        rec = {"gate": "learned", "split": name, "n": summary["samples"],
               "drug_match_rate": round(drug["match_rate"], 4),
               "binary_f1": round(req["f1_score"], 4),
               "auc_roc": round(req["auc_roc"], 4),
               "accuracy_4class": round(req["accuracy"], 4),
               "macro_f1_4class": round(req["macro_f1"], 4),
               "severe_recall": round(sev.get("recall", 0.0), 4),
               "severe_f1": round(sev.get("f1", 0.0), 4)}
        table.append(rec)
        with open(d / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"LEARNED {name}: " + json.dumps(rec), flush=True)
    with open(ROOT / "outputs" / "learned_gate" / "bench_learned_table.json",
              "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
