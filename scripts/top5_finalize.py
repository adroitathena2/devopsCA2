"""Full train/test of refine top-5 per model on the frozen 1600/400 split.

For each of the 10 configs: train on the full 1600 train rows with
val-400 (severity_val_hq) checkpointing — same recipe as
tune_severity_search.stage_finalize — save weights to
outputs/severity_search/top5_final_<model>_<tag>.pt, and report val-400
macro-F1 (caveated: val picks the checkpoint AND reports, same as ever).

Pick the best per model by val F1 (ties -> fewer core params); that pair
goes to production cutover.

Run with uv:
    uv run top5_finalize.py
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from clinical_copilot import paths
from tune_severity_search import (
    SEARCH_DIR,
    build_model,
    collate_nosort_fn,
    combined_losses,
    count_params,
    dev_metrics,
    item_label,
    load_data,
    read_jsonl,
    seed_everything,
)
from clinical_copilot.custom_severity_model import collate_fn

TOP5 = {
    "FastClinicalCNN": [
        "FastClinicalCNN_s069",
        "FastClinicalCNN_s085",
        "FastClinicalCNN_s042",
        "FastClinicalCNN_s080",
        "FastClinicalCNN_s071",
    ],
    "TinyClinicalFormer": [
        "TinyClinicalFormer_s068",
        "TinyClinicalFormer_s050",
        "TinyClinicalFormer_s090",
        "TinyClinicalFormer_s085",
        "TinyClinicalFormer_s043",
    ],
}

FULL_EPOCHS = 20
SEED = 7


def trial_cfg(model, tag):
    base = tag.split("_r_")[-1] if "_r_" in tag else tag
    short = base.replace(f"{model}_", "", 1)
    for r in read_jsonl(SEARCH_DIR / "refine_trials.jsonl"):
        if r["model"] == model and r["trial"].endswith(short):
            return r
    raise SystemExit(f"No refine row for {model}/{tag}")


def main():
    from clinical_copilot.custom_severity_model import SeverityDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    dataset, labels = load_data()
    val_dataset = SeverityDataset("cache/severity_val_hq.json", dataset.vocab)
    val_labels = np.array([item_label(d) for d in val_dataset.data])
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            collate_fn=collate_nosort_fn)

    report = {}
    for model, tags in TOP5.items():
        report[model] = []
        for tag in tags:
            row = trial_cfg(model, tag)
            cfg = row["config"]
            bs = cfg["batch_size"]
            train_loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                                      collate_fn=collate_fn)
            sub_labels = labels.tolist()
            counts = [max(sub_labels.count(c), 1) for c in range(3)]
            weights = torch.FloatTensor([len(sub_labels) / c for c in counts]).to(device)
            weights = weights / weights.sum() * 3.0
            net = build_model(model, len(dataset.vocab.word2idx),
                              len(dataset.vocab.char2idx), cfg).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
            smooth, alpha = cfg["label_smoothing"], cfg["alpha"]
            gamma = cfg.get("focal_gamma", 0.0)
            if gamma:
                smooth = 0.0
            best_loss, best_state = float("inf"), None
            t0 = time.time()
            for _ in tqdm(range(FULL_EPOCHS), desc=f"[top5 {model}/{tag}]",
                          unit="ep", leave=False):
                net.train()
                for word_ids, char_ids, lengths, hard_labels, soft_labels in train_loader:
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
                    opt.zero_grad()
                    logits = net(word_ids, char_ids, lengths)
                    from tune_severity_search import focal_or_ce_loss
                    hard_loss = focal_or_ce_loss(logits, hard_labels, weights, smooth, gamma)
                    soft_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    (alpha * hard_loss + (1 - alpha) * soft_loss).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    opt.step()
                inner_loss = combined_losses(net, val_loader, device, weights, alpha, smooth, gamma)
                if inner_loss < best_loss:
                    best_loss = inner_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            out = SEARCH_DIR / f"top5_final_{model}_{tag}.pt"
            torch.save(best_state, out)
            net.load_state_dict(best_state)
            net.eval()
            val_logits = []
            with torch.no_grad():
                for word_ids, char_ids, lengths, *_ in val_loader:
                    val_logits.append(net(word_ids.to(device), char_ids.to(device), lengths).cpu().numpy())
            vm = dev_metrics(np.concatenate(val_logits), val_labels)
            rec = {"trial": tag, "params_core": row["params_core"],
                   "oof_f1": row["oof_f1"],
                   "val_f1": round(vm["f1"], 4), "val_acc": round(vm["acc"], 4),
                   "val_auc": round(vm["auc"], 4),
                   "seconds": round(time.time() - t0, 1),
                   "weights": str(out)}
            report[model].append(rec)
            print(f"[top5] {model}/{tag} val400: F1={vm['f1']:.4f} "
                  f"ACC={vm['acc']:.4f} AUC={vm['auc']:.4f} (OOF was {row['oof_f1']})",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(SEARCH_DIR / "top5_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    for model in TOP5:
        ranked = sorted(report[model], key=lambda r: (-r["val_f1"], r.get("params_core", 0)))
        print(f"[top5] {model} winner: {ranked[0]['trial']} val F1={ranked[0]['val_f1']:.4f}",
              flush=True)
    print("[top5] wrote top5_report.json", flush=True)


if __name__ == "__main__":
    main()
