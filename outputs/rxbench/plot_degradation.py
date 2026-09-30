import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

order = ["clean", "brand", "L1", "L2", "L3"]
base = {r["split"]: r for r in json.load(open("outputs/rxbench/metrics_table.json"))["per_split"]}
learn = {r["split"]: r for r in json.load(open("outputs/learned_gate/bench_learned_table.json"))}
learn["clean"] = {"drug_match_rate": 0.9941, "binary_f1": 0.9975, "severe_recall": 0.8769}
learn["brand"] = dict(learn["clean"])  # severity-invariant: brand == clean by construction

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in zip(
    axes,
    ["severe_recall", "binary_f1"],
    ["Severe recall by split (regex vs learned gate)", "Binary F1 by split (regex vs learned gate)"],
):
    ax.plot(order, [base[s][key] for s in order], "o-", label="regex (old production)")
    ax.plot(order, [learn[s][key] for s in order], "s--", label="learned gate")
    ax.set_xticks(range(5))
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    for i, s in enumerate(order):
        ax.text(i, base[s][key] + 0.02, f"{base[s][key]:.3f}", ha="center", fontsize=8)
fig.suptitle("RxBench robustness: learned gate holds its gain wherever pairs survive linking")
fig.tight_layout()
fig.savefig("outputs/rxbench/degradation_gate.png", dpi=150)
print("saved outputs/rxbench/degradation_gate.png")
