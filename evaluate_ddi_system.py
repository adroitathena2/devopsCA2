"""
Evaluation pipeline for DDI detection.

Runs on test_prescriptions.json and writes outputs to:
- outputs/Keyword/
- outputs/Ensemble/

Use with uv:
    uv run evaluate_ddi_system.py --mode keyword
    uv run evaluate_ddi_system.py --mode ensemble --tuned-config outputs/tuning_results/best_config.json

Long Gemini runs (bigger prompt => quota deaths mid-run):
    uv run evaluate_ddi_system.py --mode keyword --dataset test_prescriptions.json --shuffle --seed 7
    # ...dies at row N... wait for quota, then:
    uv run evaluate_ddi_system.py --mode keyword --dataset test_prescriptions.json --shuffle --seed 7 --resume
Progress checkpoints to outputs/<Mode>/eval_checkpoint.json after every
prescription (rows + accumulated Gemini cost); --resume skips done rows,
retries rows that had Gemini errors, and restores cost totals. All final
metrics (accuracy, F1, AUC, cost) are written as usual on completion.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)

from clinical_copilot import paths, plots, severity
from clinical_copilot_backend import ClinicalCopilotPipeline


def max_severity_from_interactions(interactions: List[Dict[str, Any]]) -> str:
    """Get a single prescription-level severity as the most severe interaction."""
    if not interactions:
        return "No Interaction"

    return severity.max_label(
        i.get("severity_categorical") or i.get("ground_truth_severity")
        for i in interactions
    )


class DDIEvaluator:
    """Evaluates pre/post tuning performance on the same prescription dataset."""

    def __init__(self, mode: str, tuned_config_path: str = ""):
        self.mode = mode
        self.output_dir = paths.OUTPUTS_DIR / ("Keyword" if mode == "keyword" else "Ensemble")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = ClinicalCopilotPipeline(config_path="config.yaml")
        if mode == "keyword":
            self.pipeline.detector.custom_model = None
            self.pipeline.detector.config.config['models']['custom_severity']['enabled'] = False

        if tuned_config_path:
            resolved_path = self._resolve_tuned_config_path(tuned_config_path)
            with open(resolved_path, "r", encoding="utf-8") as f:
                tuned = json.load(f)
            self.pipeline.detector.update_gemini_config(tuned)
            print(f"Loaded tuned Gemini config: {resolved_path}")

    @staticmethod
    def _resolve_tuned_config_path(path_str: str) -> Path:
        """Resolve tuned config path with helpful fallbacks and errors."""
        direct = Path(path_str)
        tuning_dir = paths.OUTPUTS_DIR / "tuning_results"
        candidates = [
            direct,
            tuning_dir / path_str,
            tuning_dir / "best_config.json",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        searched = "\n".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "Could not find tuned config file. Searched:\n"
            f"{searched}\n"
            "Use: uv run evaluate_ddi_system.py --mode ensemble --dataset test_prescriptions.json "
            "--tuned-config outputs/tuning_results/best_config.json"
        )

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "eval_checkpoint.json"

    def _save_checkpoint(
        self,
        rows: List[Dict[str, Any]],
        meta: Dict[str, Any] | None = None,
    ) -> None:
        """Persist rows + Gemini usage after every prescription so a quota
        death or crash can resume with --resume instead of restarting."""
        det = getattr(getattr(self, "pipeline", None), "detector", None)
        payload = {
            "meta": meta or {},
            "rows": rows,
            "gemini_usage": dict(getattr(det, "gemini_usage", {}) or {}),
            "gemini_pricing": [
                getattr(det, "gemini_price_input_per_1m", 0.0),
                getattr(det, "gemini_price_output_per_1m", 0.0),
            ],
            "gemini_models": [
                getattr(det, "gemini_model_severity", ""),
                getattr(det, "gemini_model_reasoning", ""),
            ],
        }
        tmp = self._checkpoint_path().with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(self._checkpoint_path())

    def _load_checkpoint(self, meta: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with open(self._checkpoint_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_meta = payload.get("meta", {}) or {}
        if meta:
            for k, v in meta.items():
                if saved_meta.get(k) != v:
                    print(
                        f"WARNING: checkpoint {k}={saved_meta.get(k)!r} != "
                        f"current {v!r}; still resuming by prescription id.",
                        flush=True,
                    )
        det = getattr(getattr(self, "pipeline", None), "detector", None)
        if det is not None:
            det.gemini_usage = dict(payload.get("gemini_usage", {}) or {})
            pricing = payload.get("gemini_pricing") or [0.0, 0.0]
            det.gemini_price_input_per_1m, det.gemini_price_output_per_1m = pricing
        return payload.get("rows", [])

    def _gemini_total_errors(self) -> int:
        det = getattr(getattr(self, "pipeline", None), "detector", None)
        get_stats = getattr(det, "get_gemini_stats", None)
        if not callable(get_stats):
            return 0
        try:
            return int(get_stats().get("total_errors", 0))
        except Exception:
            return 0

    def run_inference(
        self,
        prescriptions: List[Dict[str, Any]],
        resume_rows: List[Dict[str, Any]] | None = None,
        retry_errors: bool = True,
        checkpoint_meta: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        import time

        rows: List[Dict[str, Any]] = []
        done_idx: Dict[Any, int] = {}
        if resume_rows:
            for r in resume_rows:
                done_idx[(r.get("prescription_id"), r.get("prescription_index"))] = len(rows)
                rows.append(r)
            print(f"Resumed {len(rows)} checkpointed prescriptions.", flush=True)

        total = len(prescriptions)
        for idx, item in enumerate(prescriptions, start=1):
            prescription_id = item.get("prescription_id", idx)
            key = (item.get("prescription_id"), idx)
            if key in done_idx:
                existing = rows[done_idx[key]]
                if retry_errors and existing.get("gemini_errors", 0) > 0:
                    print(
                        f"\n=== Prescription {idx}/{total} (id={prescription_id}) "
                        f"RETRY (gemini errors: {existing['gemini_errors']}) ===",
                        flush=True,
                    )
                else:
                    continue
            else:
                print(f"\n=== Prescription {idx}/{total} (id={prescription_id}) ===", flush=True)

            text = item.get("prescription_text", "")
            gt_interactions = item.get("interactions", [])
            gt_severity = max_severity_from_interactions(gt_interactions)

            err_before = self._gemini_total_errors()
            t0 = time.time()
            result = self.pipeline.process_input(text)
            latency = time.time() - t0
            row_errors = self._gemini_total_errors() - err_before

            pred_interactions = result.get("drug_interactions", [])
            pred_severity = max_severity_from_interactions(pred_interactions)

            pred_entities = result.get("identified_medicines", [])
            pred_drugs = []
            for d in pred_entities:
                if isinstance(d, dict):
                    pred_drugs.append(str(d.get("name", "")).lower())
                else:
                    pred_drugs.append(str(d).lower())
            gt_drugs = item.get("drugs", [])

            missed_drugs = []
            for gt_d in gt_drugs:
                if not self._gt_drug_found(str(gt_d), pred_drugs):
                    missed_drugs.append(gt_d)

            rows.append(
                {
                    "prescription_id": item.get("prescription_id"),
                    "prescription_index": idx,
                    "prescription_text": text,
                    "gt_severity": gt_severity,
                    "pred_severity": pred_severity,
                    "gt_has_interaction": bool(item.get("has_interaction", False)),
                    "pred_has_interaction": len(pred_interactions) > 0,
                    "pred_interactions_count": len(pred_interactions),
                    "pred_interactions": pred_interactions,
                    "gt_interactions": gt_interactions,
                    "latency_seconds": latency,
                    "missed_drugs": missed_drugs,
                    "gt_drug_count": len(gt_drugs),
                    "missed_drug_count": len(missed_drugs),
                    "gemini_errors": row_errors,
                }
            )
            if key in done_idx:
                rows[done_idx[key]] = rows.pop()
            self._save_checkpoint(rows, checkpoint_meta)

        # Stable output order regardless of retries.
        rows.sort(key=lambda r: (r.get("prescription_index", 0)))
        return rows

    def _gt_drug_found(self, gt_token: str, pred_lowers: List[str]) -> bool:
        """Synonym-aware GT drug matching: a GT token (often a brand like
        'Azithral' or 'Omez') counts as found if any generic it resolves to
        was extracted (e.g. 'azithromycin', 'omeprazole'). Falls back to the
        old substring check if the resolver is unavailable."""
        gt_lower = gt_token.strip().lower()
        if any(gt_lower in pd or pd in gt_lower for pd in pred_lowers):
            return True
        try:
            resolve = getattr(getattr(self, "pipeline", None), "extractor", None)
            resolve = getattr(resolve, "_resolve_names", None)
            if not callable(resolve):
                return False
            resolved = {str(g).strip().lower() for g in resolve(gt_lower)}
            return bool(resolved & set(pred_lowers))
        except Exception:
            return False

    def _build_targets(self, rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.array([severity.class_index(r["gt_severity"]) for r in rows], dtype=int)
        y_pred = np.array([severity.class_index(r["pred_severity"]) for r in rows], dtype=int)

        # Binary task: interaction (Moderate/Severe) vs no interaction (None/Mild).
        y_true_bin = np.array([1 if s >= 2 else 0 for s in y_true], dtype=int)
        y_pred_risk = []
        for r in rows:
            max_risk = 0.0
            if r.get("pred_interactions"):
                for inter in r["pred_interactions"]:
                    risk = inter.get("serious_risk_score")
                    if risk is None:
                        risk = severity.risk_score(inter.get("severity_categorical", ""))
                    max_risk = max(max_risk, float(risk))
            else:
                max_risk = severity.risk_score(r["pred_severity"])
            y_pred_risk.append(max_risk)

        y_pred_risk = np.array(y_pred_risk, dtype=float)

        return y_true, y_pred, y_true_bin, y_pred_risk

    def _binary_metrics(self, y_true_bin: np.ndarray, y_pred_risk: np.ndarray) -> Dict[str, Any]:
        # Use a fixed threshold of 0.5 for binary classification (Moderate/Severe vs None)
        # to prevent data leakage (optimizing threshold on test set)
        best_t = 0.5

        y_pred_bin = (y_pred_risk >= best_t).astype(int)

        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        if len(np.unique(y_true_bin)) < 2:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([1.0, 0.0])
            auc_roc = 0.5
        else:
            fpr, tpr, thresholds = roc_curve(y_true_bin, y_pred_risk)
            auc_roc = auc(fpr, tpr)

        net_benefit = self._decision_curve_net_benefit(y_true_bin, y_pred_risk)

        return {
            "auc_roc": float(auc_roc),
            "sensitivity_recall": float(sensitivity),
            "specificity_tnr": float(specificity),
            "f1_score": float(f1),
            "confusion_binary": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "net_benefit": net_benefit,
        }

    @staticmethod
    def _norm_pair_name(name: Any) -> str:
        return str(name or "").strip().lower()

    @classmethod
    def _pair_key(cls, a: Any, b: Any) -> tuple:
        return tuple(sorted([cls._norm_pair_name(a), cls._norm_pair_name(b)]))

    def _pair_metrics(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-pair (not prescription-max) detection + severity agreement.

        A predicted pair matches a GT pair when the unordered normalized
        drug-name sets are equal. Name-variant mismatches (brand vs generic)
        count against us here; that strictness is intentional.
        """
        tp_det = fp_det = fn_det = 0
        sev_match = 0
        for r in rows:
            gt_map = {}
            for g in r.get("gt_interactions", []) or []:
                gt_map.setdefault(
                    self._pair_key(g.get("drug_a"), g.get("drug_b")),
                    severity.normalize(g.get("ground_truth_severity")
                                       or g.get("severity_categorical")),
                )
            pred_map = {}
            for p in r.get("pred_interactions", []) or []:
                pred_map.setdefault(
                    self._pair_key(p.get("drug_a"), p.get("drug_b")),
                    severity.normalize(p.get("severity_categorical")),
                )
            for k, psev in pred_map.items():
                if k in gt_map:
                    tp_det += 1
                    if psev == gt_map[k]:
                        sev_match += 1
                else:
                    fp_det += 1
            for k in gt_map:
                if k not in pred_map:
                    fn_det += 1
        prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
        rec = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "tp": tp_det, "fp": fp_det, "fn": fn_det,
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "severity_exact_on_matched": float(sev_match / tp_det) if tp_det else 0.0,
        }

    @staticmethod
    def _bootstrap_cis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_bin: np.ndarray,
        y_pred_risk: np.ndarray,
        seed: int = 7,
        n_boot: int = 2000,
    ) -> Dict[str, Any]:
        """Paired bootstrap 95% CIs (resample rows, recompute) for accuracy,
        macro F1, and binary F1. Resampling predictions, not retraining:
        understates true variance; read as a lower bound on uncertainty."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        accs, macros, binf1s = [], [], []
        thr = 0.5
        for idx in rng.integers(0, n, (n_boot, n)):
            yt, yp = y_true[idx], y_pred[idx]
            accs.append(float(np.mean(yt == yp)))
            macros.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
            yb = (y_pred_risk[idx] >= thr).astype(int)
            binf1s.append(float(f1_score(y_true_bin[idx], yb, zero_division=0)))
        out = {}
        for name, vals in (("accuracy", accs), ("macro_f1", macros), ("binary_f1", binf1s)):
            arr = np.array(vals)
            out[name] = {
                "mean": float(arr.mean()),
                "ci95": [round(float(np.percentile(arr, 2.5)), 4),
                         round(float(np.percentile(arr, 97.5)), 4)],
            }
        return out

    @staticmethod
    def _decontamination_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """How many test pair descriptions appear verbatim in train/val texts.
        The severity classifier grades description text, so verbatim overlap
        is the leakage-relevant measure (drug-name overlap is expected: the
        detector is SUPPOSED to look drugs up)."""
        try:
            train_texts = {r.get("text", "") for r in
                           json.load(open("cache/severity_train_hq.json", encoding="utf-8"))}
            val_texts = {r.get("text", "") for r in
                         json.load(open("cache/severity_val_hq.json", encoding="utf-8"))}
        except Exception:
            return {"test_pairs": 0, "verbatim_in_train": 0, "verbatim_in_val": 0}
        pairs, in_train, in_val = 0, 0, 0
        for r in rows:
            for g in r.get("gt_interactions", []) or []:
                d = g.get("drugbank_description", "")
                if not d:
                    continue
                pairs += 1
                if d in train_texts:
                    in_train += 1
                if d in val_texts:
                    in_val += 1
        return {"test_pairs": pairs, "verbatim_in_train": in_train,
                "verbatim_in_val": in_val}


    @staticmethod
    def _decision_curve_net_benefit(y_true_bin: np.ndarray, y_pred_risk: np.ndarray) -> Dict[str, float]:
        """
        Net Benefit at threshold p_t:
        NB = TP/n - FP/n * p_t/(1-p_t)
        """
        n = len(y_true_bin)
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        values: Dict[str, float] = {}

        for pt in thresholds:
            y_pred_bin = (y_pred_risk >= pt).astype(int)
            tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
            values[f"{pt:.1f}"] = float(nb)

        return values

    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true, y_pred, y_true_bin, y_pred_risk = self._build_targets(rows)

        avg_latency = float(np.mean([r.get("latency_seconds", 0.0) for r in rows]))

        labels = list(range(len(severity.OUTCOME_LABELS)))
        target_names = list(severity.OUTCOME_LABELS)

        cls_report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        cls_report_text = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        per_class = {
            target_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        }

        binary = self._binary_metrics(y_true_bin, y_pred_risk)

        accuracy = float(np.mean(y_true == y_pred)) if len(rows) else 0.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        pair = self._pair_metrics(rows)
        boot = self._bootstrap_cis(y_true, y_pred, y_true_bin, y_pred_risk)
        decontam = self._decontamination_report(rows)

        gt_drug_total = sum(r.get("gt_drug_count", 0) for r in rows)
        missed_drug_total = sum(r.get("missed_drug_count", 0) for r in rows)
        matched_drug_total = gt_drug_total - missed_drug_total
        drug_match_rate = matched_drug_total / gt_drug_total if gt_drug_total else 0.0

        return {
            "mode": self.mode,
            "samples": len(rows),
            "average_latency_seconds": avg_latency,
            "metrics_required": {
                "auc_roc": binary["auc_roc"],
                "sensitivity_recall": binary["sensitivity_recall"],
                "specificity_tnr": binary["specificity_tnr"],
                "f1_score": binary["f1_score"],
                "net_benefit": binary["net_benefit"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "per_class_f1": {k: v["f1"] for k, v in per_class.items()},
            },
            "classification_report": cls_report,
            "classification_report_text": cls_report_text,
            "confusion_matrix": {
                "labels": target_names,
                "matrix": cm.tolist(),
            },
            "per_class_metrics": per_class,
            "binary_details": binary,
            "pair_metrics": pair,
            "bootstrap_cis": boot,
            "decontamination": decontam,
            "drug_name_matching": {
                "gt_total": gt_drug_total,
                "matched": matched_drug_total,
                "missed": missed_drug_total,
                "match_rate": float(drug_match_rate),
            },
        }

    def save_outputs(self, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        metrics_json = self.output_dir / "metrics.json"
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report_txt = self.output_dir / "classification_report.txt"
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(summary["classification_report_text"])

        details_csv = self.output_dir / "detailed_results.csv"
        pd.DataFrame(rows).drop(columns=["pred_interactions", "gt_interactions"]).to_csv(details_csv, index=False)

        plots.save_confusion_matrix(summary, self.output_dir / "confusion_matrix.png")
        plots.save_roc_curve(summary, self.output_dir / "roc_curve.png")
        plots.save_decision_curve(summary, self.output_dir / "decision_curve.png")

        summary_txt = self.output_dir / "summary_stats.txt"
        req = summary["metrics_required"]
        drug = summary["drug_name_matching"]
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"Mode: {summary['mode']}\n")
            f.write(f"Samples: {summary['samples']}\n\n")
            f.write("Required Metrics\n")
            f.write("-" * 60 + "\n")
            f.write(f"AUC-ROC: {req['auc_roc']:.4f}\n")
            f.write(f"Sensitivity (Recall): {req['sensitivity_recall']:.4f}\n")
            f.write(f"Specificity (TNR): {req['specificity_tnr']:.4f}\n")
            f.write(f"F1-Score (binary): {req['f1_score']:.4f}\n")
            f.write(f"Accuracy (4-class): {req['accuracy']:.4f}\n")
            f.write(f"Macro F1 (4-class): {req['macro_f1']:.4f}\n")
            f.write(f"Weighted F1 (4-class): {req['weighted_f1']:.4f}\n")
            for cls_name, cls_f1 in req["per_class_f1"].items():
                f.write(f"F1 {cls_name}: {cls_f1:.4f}\n")
            f.write(f"Drug Name Match Rate: {drug['match_rate']:.4f} "
                    f"({drug['matched']}/{drug['gt_total']})\n")
            pair = summary.get("pair_metrics", {})
            if pair:
                f.write("Pair-Level (strict name match)\n")
                f.write(f"  detection P/R/F1: {pair['precision']:.4f} / "
                        f"{pair['recall']:.4f} / {pair['f1']:.4f} "
                        f"(tp={pair['tp']} fp={pair['fp']} fn={pair['fn']})\n")
                f.write(f"  severity exact on matched: "
                        f"{pair['severity_exact_on_matched']:.4f}\n")
            boot = summary.get("bootstrap_cis", {})
            if boot:
                f.write("Bootstrap 95% CIs (paired rows, 2000 resamples)\n")
                for k in ("accuracy", "macro_f1", "binary_f1"):
                    if k in boot:
                        lo, hi = boot[k]["ci95"]
                        f.write(f"  {k}: {boot[k]['mean']:.4f} [{lo:.4f}, {hi:.4f}]\n")
            decontam = summary.get("decontamination", {})
            if decontam:
                f.write("Decontamination (verbatim description overlap)\n")
                f.write(f"  test pairs: {decontam['test_pairs']}, "
                        f"in train: {decontam['verbatim_in_train']}, "
                        f"in val: {decontam['verbatim_in_val']}\n")
            f.write("Net Benefit\n")
            for k, v in req["net_benefit"].items():
                f.write(f"  threshold={k}: {v:.4f}\n")

        print(f"Saved evaluation outputs to {self.output_dir}")

        detector = getattr(getattr(self, "pipeline", None), "detector", None)
        get_stats = getattr(detector, "get_gemini_stats", None)
        if callable(get_stats):
            gemini_stats = get_stats()
            if gemini_stats.get("total_calls", 0) > 0:
                stats_path = self.output_dir / "gemini_stats.json"
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(gemini_stats, f, indent=2)
                print(f"Gemini API usage: {gemini_stats['total_calls']} calls, "
                      f"{gemini_stats['total_input_tokens']} in / "
                      f"{gemini_stats['total_output_tokens']} out / "
                      f"{gemini_stats.get('total_thinking_tokens', 0)} thinking tokens, "
                      f"est. cost ${gemini_stats['est_cost_usd']:.4f}")
                with open(self.output_dir / "summary_stats.txt", "a", encoding="utf-8") as f:
                    f.write(f"\nGemini API Usage\n")
                    f.write(f"  calls: {gemini_stats['total_calls']} "
                            f"(errors: {gemini_stats['total_errors']})\n")
                    f.write(f"  tokens: {gemini_stats['total_input_tokens']} in / "
                            f"{gemini_stats['total_output_tokens']} out / "
                            f"{gemini_stats.get('total_thinking_tokens', 0)} thinking\n")
                    f.write(f"  est. cost: ${gemini_stats['est_cost_usd']:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DDI system (keyword vs ensemble severity)")
    parser.add_argument("--mode", choices=["keyword", "ensemble"], required=True)
    parser.add_argument("--dataset", default="test_prescriptions.json")
    parser.add_argument("--tuned-config", default="", help="Path to outputs/tuning_results/best_config.json")
    parser.add_argument("--shuffle", action="store_true",
                        help="Deterministically shuffle prescriptions before running "
                             "(so a quota death leaves a representative partial).")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from outputs/<Mode>/eval_checkpoint.json "
                             "(skips done rows, retries rows with Gemini errors, "
                             "restores accumulated cost). Use the same flags.")
    parser.add_argument("--no-retry-errors", action="store_true",
                        help="On resume, don't re-run rows that had Gemini errors.")
    args = parser.parse_args()

    evaluator = DDIEvaluator(mode=args.mode, tuned_config_path=args.tuned_config)
    prescriptions = evaluator.load_dataset(args.dataset)
    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        order = list(range(len(prescriptions)))
        rng.shuffle(order)
        prescriptions = [prescriptions[i] for i in order]
        print(f"Shuffled {len(prescriptions)} prescriptions (seed={args.seed}).", flush=True)
    meta = {"dataset": args.dataset, "mode": args.mode,
            "shuffled": args.shuffle, "seed": args.seed,
            "total": len(prescriptions)}

    resume_rows = None
    if args.resume:
        try:
            resume_rows = evaluator._load_checkpoint(meta)
        except FileNotFoundError:
            print("No checkpoint found; starting fresh.", flush=True)
            resume_rows = None

    rows = evaluator.run_inference(
        prescriptions,
        resume_rows=resume_rows,
        retry_errors=not args.no_retry_errors,
        checkpoint_meta=meta,
    )
    summary = evaluator.evaluate(rows)
    evaluator.save_outputs(rows, summary)

    req = summary["metrics_required"]
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"AUC-ROC: {req['auc_roc']:.4f}")
    print(f"Sensitivity (Recall): {req['sensitivity_recall']:.4f}")
    print(f"Specificity (TNR): {req['specificity_tnr']:.4f}")
    print(f"F1-Score (binary): {req['f1_score']:.4f}")
    print(f"Accuracy (4-class): {req['accuracy']:.4f}")
    print(f"Macro F1 (4-class): {req['macro_f1']:.4f}")
    print(f"Weighted F1 (4-class): {req['weighted_f1']:.4f}")
    for cls_name, cls_f1 in req["per_class_f1"].items():
        print(f"F1 {cls_name}: {cls_f1:.4f}")
    pair = summary.get("pair_metrics", {})
    if pair:
        print(f"Pair P/R/F1: {pair['precision']:.4f}/{pair['recall']:.4f}/{pair['f1']:.4f} "
              f"(tp={pair['tp']} fp={pair['fp']} fn={pair['fn']}, "
              f"sev-exact={pair['severity_exact_on_matched']:.4f})")
    boot = summary.get("bootstrap_cis", {})
    if boot:
        for k in ("accuracy", "macro_f1", "binary_f1"):
            if k in boot:
                lo, hi = boot[k]["ci95"]
                print(f"{k} 95% CI: [{lo:.4f}, {hi:.4f}]")
    decontam = summary.get("decontamination", {})
    if decontam:
        print(f"Decontam: {decontam['verbatim_in_train']}/{decontam['test_pairs']} test "
              f"descriptions verbatim in train, {decontam['verbatim_in_val']} in val")
    print(f"Drug Name Match Rate: {summary['drug_name_matching']['match_rate']:.4f}")
    print("=" * 70)

    missed_count = 0
    print("\n--- Missed Named Entities (true misses, synonym-aware) ---")
    for r in rows:
        if r.get("missed_drugs"):
            missed_count += len(r["missed_drugs"])
            print(f"\n[Prescription {r.get('prescription_id', '?')} | "
                  f"{r.get('prescription_index', '?')}/{len(rows)}]")
            print(f"Text: {r.get('prescription_text', '').strip()}")
            print(f"Missed Entities: {', '.join(r['missed_drugs'])}")

    if missed_count == 0:
        print("No entities were missed by the NER!")
    else:
        print(f"\nTotal missed entities across dataset: {missed_count}")

if __name__ == "__main__":
    main()
