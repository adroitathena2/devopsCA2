import argparse
import json
import os
import shutil
import subprocess

import torch
import yaml

from clinical_copilot import paths, plots

BASELINE_EMBEDDINGS = {
    "model_name": "ibm-granite/granite-embedding-278m-multilingual",
    "device": "cpu",
    "trust_remote_code": False,
    "dtype": "auto",
    "batch_size": 64,
}

MODELS = [
    {
        "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "batch_size": 2048,
    },
    {
        "name": "ibm-granite/granite-embedding-278m-multilingual",
        "batch_size": 64,
    },
    {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "batch_size": 32,
    },
    {
        "name": "google/embeddinggemma-300m",
        "batch_size": 256,
    },
    {
        "name": "tencent/WeMM-Embedding-2B",
        "batch_size": 64,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    {
        "name": "Qwen/Qwen3-VL-Embedding-2B",
        "batch_size": 64,
        "dtype": "bfloat16",
    },
    {
        "name": "NeuML/pubmedbert-base-embeddings",
        "batch_size": 128,
    },
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, path="config.yaml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


def apply_model_config(entry, path="config.yaml"):
    config = load_config(path)
    embeddings = config["models"]["embeddings"]
    embeddings["model_name"] = entry["name"]
    embeddings["device"] = DEVICE
    embeddings["trust_remote_code"] = entry.get("trust_remote_code", False)
    embeddings["dtype"] = entry.get("dtype", "auto")
    embeddings["batch_size"] = entry.get("batch_size")
    save_config(config, path)


def restore_baseline(path="config.yaml"):
    config = load_config(path)
    config["models"]["embeddings"].update(BASELINE_EMBEDDINGS)
    save_config(config, path)


def check_access(model_id):
    try:
        from huggingface_hub import get_token, model_info
    except ImportError:
        return True, ""
    try:
        info = model_info(model_id)
    except Exception as exc:
        return True, f"access check unavailable ({exc.__class__.__name__})"
    if getattr(info, "gated", False) and not get_token():
        return False, f"gated model ('{info.gated}') and no HF token available"
    return True, ""


def is_oom_failure(log_path):
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
    except OSError:
        return False
    return (
        "out of memory" in text
        or "outofmemoryerror" in text
        or "cudaerrormemoryallocation" in text
    )


def batch_size_attempts(configured):
    sizes = []
    if configured:
        sizes.append(configured)
        size = configured
        while size > 16:
            size //= 2
            if size >= 16:
                sizes.append(size)
    if not sizes:
        sizes.append(None)
    return sizes


def run_step(args, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    return completed.returncode


def parse_metrics():
    with open(paths.OUTPUTS_DIR / "Ensemble" / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    required = metrics.get("metrics_required", {})
    return {
        "Specificity (TNR)": required.get("specificity_tnr", 0) * 100,
        "Sensitivity (Recall)": required.get("sensitivity_recall", 0) * 100,
        "F1-Score": required.get("f1_score", 0) * 100,
        "AUC-ROC": required.get("auc_roc", 0) * 100,
        "Drug Match %": metrics.get("drug_name_matching", {}).get("match_rate", 0) * 100,
        "Latency (s)": metrics.get("average_latency_seconds", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models end-to-end")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated substrings selecting a subset of MODELS (default: all)",
    )
    args = parser.parse_args()

    selected = MODELS
    if args.models:
        filters = [f.strip().lower() for f in args.models.split(",") if f.strip()]
        selected = [m for m in MODELS if any(f in m["name"].lower() for f in filters)]
        if not selected:
            print(f"No models matched --models={args.models}")
            return

    paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if os.path.exists(paths.OUTPUTS_DIR / "embedding_comparison_results.json"):
        with open(paths.OUTPUTS_DIR / "embedding_comparison_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

    for entry in selected:
        model_id = entry["name"]
        safe_name = model_id.replace("/", "_")
        print(f"\n{'=' * 70}\nTesting Model: {model_id}\n{'=' * 70}")

        allowed, reason = check_access(model_id)
        if not allowed:
            print(f"Skipping {model_id}: {reason}")
            results[model_id] = {"status": "skipped", "reason": reason}
            continue

        build_log = paths.OUTPUTS_DIR / f"build_log_{safe_name}.txt"
        build_ok = False
        build_error = ""
        for size in batch_size_attempts(entry.get("batch_size")):
            attempt = dict(entry)
            attempt["batch_size"] = size
            apply_model_config(attempt)

            print(f"[{model_id}] Running build_cache.py (batch_size={size})...")
            build_rc = run_step(["uv", "run", "build_cache.py"], build_log)
            if build_rc == 0:
                build_ok = True
                break

            build_error = f"build_cache.py exited with code {build_rc}"
            if not is_oom_failure(build_log):
                break
            print(f"[{model_id}] CUDA OOM detected, retrying with a smaller batch size...")

        if not build_ok:
            results[model_id] = {"status": "failed", "reason": build_error}
            print(f"build_cache.py failed for {model_id} - see {build_log}")
            continue

        print(f"[{model_id}] Running evaluate_ddi_system.py...")
        test_log = paths.OUTPUTS_DIR / f"test_log_{safe_name}.txt"
        eval_rc = run_step(
            ["uv", "run", "evaluate_ddi_system.py", "--mode", "ensemble"],
            test_log,
        )
        if eval_rc != 0:
            results[model_id] = {
                "status": "failed",
                "reason": f"evaluate_ddi_system.py exited with code {eval_rc}",
            }
            print(f"evaluate_ddi_system.py failed for {model_id} - see {test_log}")
            continue

        metrics = parse_metrics()
        results[model_id] = metrics
        shutil.copy(paths.OUTPUTS_DIR / "Ensemble" / "metrics.json",
                    paths.OUTPUTS_DIR / f"metrics_{safe_name}.json")
        print(
            f"[{model_id}] F1={metrics['F1-Score']:.2f}%  "
            f"AUC={metrics['AUC-ROC']:.2f}%  Latency={metrics['Latency (s)']:.2f}s"
        )

    restore_baseline()

    with open(paths.OUTPUTS_DIR / "embedding_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nGenerating comparative chart...")
    save_chart(results)


def short_model_name(model_id):
    return (
        model_id.split("/")[-1]
        .replace("granite-embedding-", "granite-")
        .replace("embeddinggemma", "gemma")
        .replace("-from-PubMedBERT-fulltext", "")
    )


def save_chart(results):
    valid = {k: v for k, v in results.items() if "F1-Score" in v}
    if not valid:
        print("No valid results to plot.")
        return

    rows = []
    for model_id, metrics in valid.items():
        for metric_name, value in metrics.items():
            if metric_name == "Latency (s)":
                continue
            rows.append({"Model": short_model_name(model_id), "Metric": metric_name, "Value": value})

    plots.save_metric_comparison(rows, paths.OUTPUTS_DIR / "chart_embedding_comparison.png")
    print(f"Chart saved as {paths.OUTPUTS_DIR / 'chart_embedding_comparison.png'}")


if __name__ == "__main__":
    main()
