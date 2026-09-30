"""Chart battery for the severity-architecture search.

Reads outputs/severity_search/{screen_trials.jsonl,refine_trials.jsonl,
stack_results.json} (skips whatever does not exist yet, so it can be run
mid-search) and writes PNGs to outputs/severity_search/charts/.

Per model: pivot heatmaps (architecture dims, lr x dropout), a Spearman
correlation heatmap, a screen F1-vs-params scatter, and one boxplot per
sampled hyperparameter. Cross-model: refine fold-stability bars, ranked OOF
bars, and a stacker comparison chart.

Run with uv:
    uv run plot_search_charts.py
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ARCH_PAIRS = {
    "MedSeverityNet": ("hidden_dim", "num_layers"),
    "FastClinicalCNN": ("filters", "kernels"),
    "TinyClinicalFormer": ("num_layers", "nhead"),
}
NUMERIC_HINTS = {"word_embed_dim", "char_embed_dim", "char_filters", "classifier_hidden",
                 "dropout", "weight_decay", "alpha", "label_smoothing", "batch_size",
                 "hidden_dim", "num_layers", "filters", "dim_ff", "nhead"}


def load_trials(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cfg = pd.json_normalize(df["config"]).add_prefix("cfg.")
    df = pd.concat([df.drop(columns=["config"]), cfg], axis=1)
    for col in df.columns:
        if col.startswith("cfg.") and df[col].apply(lambda v: isinstance(v, list)).any():
            df[col] = df[col].apply(lambda v: "-".join(map(str, v)) if isinstance(v, list) else v)
    df["score"] = df["oof_f1"] if "oof_f1" in df.columns else df.get("f1")
    return df


def pivot_heatmap(df, model, outdir, index, columns, tag, prefix="screen"):
    sub = df[df["model"] == model]
    if sub.empty:
        return False
    means = sub.pivot_table(values="score", index=f"cfg.{index}", columns=f"cfg.{columns}",
                            aggfunc="mean")
    counts = sub.pivot_table(values="score", index=f"cfg.{index}", columns=f"cfg.{columns}",
                             aggfunc="count")
    if means.size == 0:
        return False
    annot = means.copy().astype(object)
    for i in means.index:
        for c in means.columns:
            annot.loc[i, c] = f"{means.loc[i, c]:.3f}\n(n={int(counts.loc[i, c])})" \
                if pd.notna(means.loc[i, c]) else "—"
    plt.figure(figsize=(max(7, 1.6 * means.shape[1] + 3), max(5, 1.1 * means.shape[0] + 2.5)))
    sns.heatmap(means, annot=annot, fmt="", cmap="YlGnBu", mask=means.isna(),
                cbar_kws={"label": "mean F1"})
    plt.title(f"{model}: mean F1 by {index} x {columns}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_heatmap_{tag}_{model}.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def corr_heatmap(df, model, outdir, prefix="screen"):
    sub = df[df["model"] == model]
    if sub.empty:
        return False
    work = pd.DataFrame({"F1": sub["score"], "Acc": sub.get("acc"), "AUC": sub.get("auc"),
                         "log_params": np.log10(sub["params_core"])})
    for col in sub.columns:
        if col.startswith("cfg.") and col[4:] in NUMERIC_HINTS:
            vals = sub[col]
            if vals.nunique() > 1:
                name = col[4:]
                work[name] = np.log10(vals) if name == "lr" else pd.to_numeric(vals, errors="coerce")
    work = work.dropna(axis=1, how="all")
    if work.shape[1] < 3:
        return False
    corr = work.corr(method="spearman")
    plt.figure(figsize=(max(8, 0.9 * corr.shape[1] + 3), max(6, 0.9 * corr.shape[0] + 2)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Spearman r"})
    plt.title(f"{model}: hyperparameter/metric rank correlations", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_heatmap_corr_{model}.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def screen_scatter(df, model, outdir, prefix="screen"):
    sub = df[df["model"] == model]
    if sub.empty:
        return False
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sub, x="params_core", y="score", hue="cfg.lr", size="cfg.dropout",
                    sizes=(40, 160), palette="viridis", edgecolor="black", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Non-embedding parameters (log scale)")
    plt.ylabel("Dev Macro-F1")
    plt.title(f"{model}: screen F1 vs model size", fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_scatter_{model}.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def param_boxes(df, model, outdir, prefix="screen"):
    sub = df[df["model"] == model]
    if sub.empty:
        return 0
    made = 0
    params = sorted(c[4:] for c in sub.columns
                    if c.startswith("cfg.") and sub[c].nunique() > 1)
    for param in params:
        col = f"cfg.{param}"
        order = sorted(sub[col].unique(),
                       key=lambda v: (float(v) if isinstance(v, (int, float)) else str(v)))
        plt.figure(figsize=(max(7, 1.2 * len(order) + 3), 5.5))
        sns.boxplot(data=sub, x=col, y="score", order=order, hue=col,
                      palette="Blues_d", legend=False)
        sns.stripplot(data=sub, x=col, y="score", order=order, color="black",
                      alpha=0.5, size=4)
        plt.xlabel(param)
        plt.ylabel("Dev Macro-F1")
        plt.title(f"{model}: F1 by {param} (n={len(sub)})", fontweight="bold")
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_box_{model}_{param}.png", dpi=200, bbox_inches="tight")
        plt.close()
        made += 1
    return made


def refine_charts(df, outdir):
    made = 0
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        if "fold_f1s" not in sub.columns:
            continue
        sub["mean_fold"] = sub["fold_f1s"].apply(np.mean)
        sub["std_fold"] = sub["fold_f1s"].apply(lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0)
        sub = sub.sort_values("mean_fold", ascending=True)
        labels = [t.replace(f"{model}_r_", "") for t in sub["trial"]]

        plt.figure(figsize=(10, max(5, 0.55 * len(sub) + 2)))
        plt.barh(labels, sub["mean_fold"], xerr=sub["std_fold"], capsize=4, color="#1f77b4")
        plt.xlabel("Mean fold Macro-F1 ± std")
        plt.title(f"{model}: refine candidates across 3 folds", fontweight="bold")
        plt.tight_layout()
        plt.savefig(outdir / f"refine_folds_{model}.png", dpi=200, bbox_inches="tight")
        plt.close()
        made += 1

        sub2 = sub.sort_values("oof_f1", ascending=True)
        labels2 = [t.replace(f"{model}_r_", "") for t in sub2["trial"]]
        colors = ["#2ca02c" if "baseline" in t else "#1f77b4" for t in sub2["trial"]]
        plt.figure(figsize=(10, max(5, 0.55 * len(sub2) + 2)))
        plt.barh(labels2, sub2["oof_f1"], color=colors)
        plt.xlabel("Pooled OOF Macro-F1")
        plt.title(f"{model}: ranked OOF scores (green = hand-picked baseline)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(outdir / f"topk_{model}.png", dpi=200, bbox_inches="tight")
        plt.close()
        made += 1
    return made


def stacker_chart(search_dir, outdir):
    path = search_dir / "stack_results.json"
    if not path.exists():
        return False
    results = json.loads(path.read_text())
    labels = [f"{r['stacker']} {','.join(f'{k}={v}' for k, v in r['params'].items())}"
              for r in results]
    means = [r["mean_f1"] for r in results]
    stds = [r["std_f1"] for r in results]
    order = np.argsort(means)
    plt.figure(figsize=(12, max(6, 0.45 * len(results) + 3)))
    plt.barh([labels[i] for i in order], [means[i] for i in order],
             xerr=[stds[i] for i in order], capsize=3, color="#9467bd")
    plt.xlabel("5-fold CV Macro-F1 ± std")
    plt.title("Stacker comparison on winners' OOF features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "stacker_compare.png", dpi=200, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Chart battery for the severity search")
    parser.add_argument("--search-dir", default=str(Path("outputs") / "severity_search"))
    args = parser.parse_args()
    search_dir = Path(args.search_dir)
    outdir = search_dir / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    screen = load_trials(search_dir / "screen_trials.jsonl")
    refine = load_trials(search_dir / "refine_trials.jsonl")
    made = 0
    for df, tag in ((screen, "screen"), (refine, "refine")):
        if df.empty:
            print(f"No {tag} trials yet, skipping {tag} charts.")
            continue
        for model in sorted(df["model"].unique()):
            for fn in (lambda: pivot_heatmap(df, model, outdir, *ARCH_PAIRS[model], "arch", tag),
                       lambda: pivot_heatmap(df, model, outdir, "lr", "dropout", "train", tag),
                       lambda: corr_heatmap(df, model, outdir, tag),
                       lambda: screen_scatter(df, model, outdir, tag)):
                try:
                    made += bool(fn())
                except Exception as e:
                    print(f"Chart failed ({tag}/{model}): {e}")
            try:
                made += param_boxes(df, model, outdir, tag)
            except Exception as e:
                print(f"Boxplots failed ({tag}/{model}): {e}")
    if not refine.empty:
        try:
            made += refine_charts(refine, outdir)
        except Exception as e:
            print(f"Refine charts failed: {e}")
    try:
        made += bool(stacker_chart(search_dir, outdir))
    except Exception as e:
        print(f"Stacker chart failed: {e}")
    print(f"Wrote {made} charts to {outdir}")


if __name__ == "__main__":
    main()
