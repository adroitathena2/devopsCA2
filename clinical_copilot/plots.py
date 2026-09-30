"""Shared plotting helpers for evaluation reports and model comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_confusion_matrix(summary, path):
    labels = summary["confusion_matrix"]["labels"]
    matrix = np.array(summary["confusion_matrix"]["matrix"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("DDI Severity Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_roc_curve(summary, path):
    roc_data = summary["binary_details"]["roc"]
    auc_roc = summary["metrics_required"]["auc_roc"]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"ROC (AUC={auc_roc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Serious DDI ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_decision_curve(summary, path):
    points = summary["metrics_required"]["net_benefit"]
    x = [float(k) for k in points.keys()]
    y = list(points.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linewidth=2, label="Model Net Benefit")
    plt.axhline(0.0, linestyle="--", color="gray", label="Treat None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_metric_comparison(rows, path, title="End-to-End Pipeline Evaluation by Embedding Model"):
    df = pd.DataFrame(rows)
    order = (
        df[df["Metric"] == "F1-Score"]
        .sort_values("Value", ascending=False)["Model"]
        .tolist()
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        x="Model",
        y="Value",
        hue="Metric",
        data=df,
        order=order,
        palette=["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"],
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontweight="bold", fontsize=10)

    plt.title(title, pad=20, fontweight="bold", fontsize=16)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xlabel("Embedding Model", fontsize=14)
    plt.ylim(0, 115)
    plt.xticks(rotation=15, ha="center", fontsize=11)
    plt.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pareto_scatter(rows, path, title="Architecture Search: OOF Macro-F1 vs Parameter Count"):
    plt.figure(figsize=(11, 7))
    markers = {"trial": "o", "baseline": "D", "winner": "*"}
    sizes = {"trial": 45, "baseline": 120, "winner": 260}
    models = sorted({r["Model"] for r in rows})
    palette = dict(zip(models, sns.color_palette("tab10", len(models))))

    for kind in ("trial", "baseline", "winner"):
        subset = [r for r in rows if r.get("Kind", "trial") == kind]
        if not subset:
            continue
        for model in models:
            pts = [r for r in subset if r["Model"] == model]
            if not pts:
                continue
            plt.scatter(
                [r["NonEmbedParams"] for r in pts],
                [r["F1"] for r in pts],
                c=[palette[model]],
                marker=markers[kind],
                s=sizes[kind],
                alpha=0.75 if kind == "trial" else 1.0,
                edgecolors="black",
                linewidths=0.8,
                label=f"{model} ({kind})" if kind != "trial" else None,
            )
        if kind != "trial":
            for r in subset:
                plt.annotate(
                    f"{r['Model']} {kind}\nF1={r['F1']:.3f}",
                    (r["NonEmbedParams"], r["F1"]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=9,
                )

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=10)
    plt.xscale("log")
    plt.xlabel("Non-embedding parameters (log scale)", fontsize=12)
    plt.ylabel("Out-of-fold Macro-F1", fontsize=12)
    plt.title(title, pad=15, fontweight="bold", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
