"""Architecture + hyperparameter search for the severity ensemble.

Protocol (CV-only: the cache/severity_train_hq.json + severity_val_hq.json split
is never re-cut, so existing production weights stay valid):
  screen   - N random configs per model (default 96) on one fixed internal
             split of the 1600 train rows (1300 fit / 300 dev, stratified),
             25 epochs. Ranks by dev macro-F1 and records trainable
             parameter counts.
  refine   - top-K screen configs (default 12) PLUS the current hand-picked
             baseline through stratified 5-fold CV on the full 1600 train
             (identical folds for every config), full epochs with inner
             early-stopping. Winner per model = best pooled OOF macro-F1,
             ties within 0.002 go to fewer non-embedding parameters.
  stack    - tune LogReg C and the XGB grid with 5-fold CV on the winners'
             temperature-calibrated OOF features (mirrors evaluate_ensemble.py).
  finalize - retrain each winner on the full 1600 train (val-400 checkpointing,
             same recipe as train_ensemble.py) into outputs/severity_search/
             (production cache/ weights are NOT overwritten), bootstrap 95%
             CIs on the pooled OOF predictions, Pareto plot, summary JSON.

New in this round (post physician-relabel): widened spaces (embed 512,
larger layers/filters, extra kernels), lr 1e-4 / wd 3e-3 / batch 32 /
smoothing 0.2 / alpha 0.9, and a focal-loss gamma dimension {0, 1, 2}
for the 90%+-Moderate imbalance (gamma>0 disables label smoothing;
distillation unchanged).

Every stage appends resume-safe JSONL logs, so an interrupted run can be
re-launched and will skip finished trials. Drop a file named STOP into
outputs/severity_search/ to abort cleanly between trials.

Run with uv:
    uv run tune_severity_search.py --stage screen
    uv run tune_severity_search.py --stage all --n-configs 96
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier

from clinical_copilot import paths, plots
from clinical_copilot.custom_severity_model import (
    SeverityDataset,
    SeverityVocab,
    build_model_from_config,
    collate_fn,
)

SEARCH_DIR = paths.OUTPUTS_DIR / "severity_search"
SCREEN_LOG = SEARCH_DIR / "screen_trials.jsonl"
REFINE_LOG = SEARCH_DIR / "refine_trials.jsonl"
STOP_FILE = SEARCH_DIR / "STOP"


def collate_nosort_fn(batch):
    """Same as custom_severity_model.collate_fn but WITHOUT length-sorting.

    The shared collate sorts each batch by descending length for packing
    efficiency. That is fine for training (the loss is order-free) but it
    silently permutes predictions relative to index-ordered labels, which
    collapses measured F1 to chance level. Every dev/val loader in this
    search uses this version so logits row i always matches dev_idx[i].
    (The models call pack_padded_sequence with enforce_sorted=False, so
    unsorted batches are fully supported.)
    """
    word_ids = [torch.tensor(x['word_ids']) for x in batch]
    hard_labels = torch.tensor([x['hard_label'] for x in batch], dtype=torch.long)
    soft_labels = torch.tensor([x['soft_label'] for x in batch], dtype=torch.float)
    lengths = torch.tensor([len(w) for w in word_ids])
    padded_word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True,
                                                      padding_value=0)
    max_word_len = max(max(len(w) for w in x['char_ids']) if x['char_ids'] else 1
                       for x in batch)
    padded_char_ids = torch.zeros((len(batch), padded_word_ids.size(1), max_word_len),
                                  dtype=torch.long)
    for i, item in enumerate(batch):
        for j, char_seq in enumerate(item['char_ids']):
            if j < padded_word_ids.size(1):
                padded_char_ids[i, j, :len(char_seq)] = torch.tensor(char_seq)
    return padded_word_ids, padded_char_ids, lengths, hard_labels, soft_labels

MODEL_NAMES = ["FastClinicalCNN", "TinyClinicalFormer"]

SHARED_SPACE = {
    "word_embed_dim": [64, 128, 256, 512],
    "char_embed_dim": [32, 64],
    "char_filters": [32, 64, 128, 256],
    "char_kernels": [[2, 3, 4], [2, 3, 4, 5]],
    "classifier_hidden": [64, 128, 256, 512],
    "dropout": [0.2, 0.3, 0.4, 0.5],
}
TRAIN_SPACE = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128],
    "label_smoothing": [0.0, 0.1, 0.2],
    "alpha": [0.25, 0.5, 0.75, 0.9],
    "focal_gamma": [0.0, 1.0, 2.0],
}
ARCH_SPACES = {
    "MedSeverityNet": {"hidden_dim": [128, 256, 384, 512], "num_layers": [1, 2, 3, 4]},
    "FastClinicalCNN": {"filters": [64, 128, 256, 512], "kernels": [[2, 3, 4], [2, 3, 4, 5], [3, 4, 5]]},
    "TinyClinicalFormer": {"num_layers": [1, 2, 4, 6], "nhead": [2, 4, 8], "dim_ff": [128, 256, 512, 1024]},
}
# Current hand-picked settings, evaluated under the identical protocol.
BASELINES = {
    "MedSeverityNet": {"word_embed_dim": 128, "char_embed_dim": 64, "char_filters": 64,
                       "char_kernels": [2, 3, 4, 5], "classifier_hidden": 128, "dropout": 0.3,
                       "hidden_dim": 256, "num_layers": 2,
                       "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
                       "label_smoothing": 0.1, "alpha": 0.5, "focal_gamma": 0.0},
    "FastClinicalCNN": {"word_embed_dim": 128, "char_embed_dim": 64, "char_filters": 64,
                        "char_kernels": [2, 3, 4, 5], "classifier_hidden": 128, "dropout": 0.3,
                        "filters": 128, "kernels": [2, 3, 4, 5],
                        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
                        "label_smoothing": 0.1, "alpha": 0.5, "focal_gamma": 0.0},
    "TinyClinicalFormer": {"word_embed_dim": 128, "char_embed_dim": 64, "char_filters": 64,
                           "char_kernels": [2, 3, 4, 5], "classifier_hidden": 128, "dropout": 0.3,
                           "num_layers": 2, "nhead": 4, "dim_ff": 256,
                           "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
                           "label_smoothing": 0.1, "alpha": 0.5, "focal_gamma": 0.0},
}
STACK_SPACE_LR = [0.01, 0.1, 1.0, 10.0]
STACK_SPACE_XGB = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (50, 100, 200, 300) for d in (2, 3, 4) for lr in (0.05, 0.1)
]


def sample_config(rng, model_name):
    cfg = {k: rng.choice(v) for k, v in SHARED_SPACE.items()}
    cfg.update({k: rng.choice(v) for k, v in TRAIN_SPACE.items()})
    cfg.update({k: rng.choice(v) for k, v in ARCH_SPACES[model_name].items()})
    if model_name == "TinyClinicalFormer":
        d_model = cfg["word_embed_dim"] + cfg["char_filters"] * len(cfg["char_kernels"])
        valid_heads = [h for h in ARCH_SPACES["TinyClinicalFormer"]["nhead"] if d_model % h == 0]
        cfg["nhead"] = rng.choice(valid_heads)
    return cfg


def build_model(model_name, vocab_size, char_vocab_size, cfg):
    return build_model_from_config(model_name, vocab_size, char_vocab_size, cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    core = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n)
    return total, core


def item_label(item):
    return item.get("llm_label", item.get("label", 0))


def focal_or_ce_loss(logits, targets, weight, smoothing, gamma):
    """Cross-entropy (gamma=0, with label smoothing) or focal loss (gamma>0,
    no smoothing: the two conflict, focal already down-weights easy rows)."""
    if gamma and gamma > 0:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        ce = F.nll_loss(log_probs, targets, weight=weight, reduction="none")
        return (((1.0 - pt) ** gamma) * ce).mean()
    return F.cross_entropy(logits, targets, weight=weight,
                           label_smoothing=smoothing)


def combined_losses(model, loader, device, class_weights, alpha, smoothing, gamma=0.0):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for word_ids, char_ids, lengths, hard_labels, soft_labels in loader:
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
            logits = model(word_ids, char_ids, lengths)
            h = focal_or_ce_loss(logits, hard_labels, class_weights, smoothing, gamma)
            s = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            total += (alpha * h + (1 - alpha) * s).item()
            n += 1
    return total / max(n, 1)


def fit_on_split(model_name, cfg, dataset, train_idx, dev_idx, epochs, seed, device, fold_tag,
                 batch_override=None):
    g = torch.Generator().manual_seed(seed)
    sub_idx, inner_idx = train_test_split(
        np.array(train_idx), test_size=0.15, random_state=1000 + seed,
        stratify=[item_label(dataset.data[i]) for i in train_idx],
    )
    sub_labels = [item_label(dataset.data[i]) for i in sub_idx]
    counts = [max(sub_labels.count(c), 1) for c in range(3)]
    weights = torch.FloatTensor([len(sub_labels) / c for c in counts]).to(device)
    weights = weights / weights.sum() * 3.0

    bs = batch_override or cfg["batch_size"]
    sub_loader = DataLoader(Subset(dataset, sub_idx), batch_size=bs, shuffle=True,
                            collate_fn=collate_fn, generator=g)
    inner_loader = DataLoader(Subset(dataset, inner_idx), batch_size=256, shuffle=False,
                              collate_fn=collate_fn)

    vocab_size = len(dataset.vocab.word2idx)
    char_size = len(dataset.vocab.char2idx)
    model = build_model(model_name, vocab_size, char_size, cfg).to(device)
    _, n_core = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    smooth, alpha = cfg["label_smoothing"], cfg["alpha"]
    gamma = cfg.get("focal_gamma", 0.0)
    if gamma:
        smooth = 0.0

    best_loss, best_state, t0 = float("inf"), None, time.time()
    epoch_iter = tqdm(range(epochs), desc=f"train {model_name} [{fold_tag}]",
                      unit="ep", leave=False, disable=not sys.stderr.isatty())
    for _ in epoch_iter:
        model.train()
        for word_ids, char_ids, lengths, hard_labels, soft_labels in sub_loader:
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
            opt.zero_grad()
            logits = model(word_ids, char_ids, lengths)
            hard_loss = focal_or_ce_loss(logits, hard_labels, weights, smooth, gamma)
            soft_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            (alpha * hard_loss + (1 - alpha) * soft_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        inner_loss = combined_losses(model, inner_loader, device, weights, alpha, smooth, gamma)
        if inner_loss < best_loss:
            best_loss = inner_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    dev_loader = DataLoader(Subset(dataset, dev_idx), batch_size=256, shuffle=False,
                            collate_fn=collate_nosort_fn)
    logits_out = []
    with torch.no_grad():
        for word_ids, char_ids, lengths, *_ in dev_loader:
            logits_out.append(model(word_ids.to(device), char_ids.to(device), lengths).cpu().numpy())
    return np.concatenate(logits_out), n_core, time.time() - t0


def dev_metrics(logits, labels):
    preds = logits.argmax(axis=1)
    y_bin = (labels > 0).astype(int)
    p_serious = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1:].sum(axis=1)
    try:
        auc = float(roc_auc_score(y_bin, p_serious))
    except ValueError:
        auc = 0.5
    return {"f1": float(f1_score(labels, preds, average="macro")),
            "acc": float(accuracy_score(labels, preds)), "auc": auc}


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def should_stop():
    return STOP_FILE.exists()


def load_data():
    vocab = SeverityVocab()
    vocab.load("cache/severity_vocab.json")
    dataset = SeverityDataset("cache/severity_train_hq.json", vocab)
    labels = np.array([item_label(d) for d in dataset.data])
    return dataset, labels


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stage_screen(args, dataset, labels, device, models):
    seed_everything(args.seed)
    fit_idx, dev_idx = train_test_split(
        np.arange(len(dataset)), test_size=300, random_state=7, stratify=labels)
    dev_labels = labels[np.array(dev_idx)]
    done = {(r["model"], r["trial"]) for r in read_jsonl(SCREEN_LOG)}
    def run_and_log(model, tag, cfg, seed, note=""):
        seed_everything(seed)
        bs_try = [None, max(cfg["batch_size"] // 2, 16), 16]
        last_err = None
        for attempt, bs in enumerate(bs_try):
            try:
                logits, n_core, secs = fit_on_split(
                    model, cfg, dataset, fit_idx, dev_idx, args.screen_epochs,
                    seed, device, fold_tag=0, batch_override=bs)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tqdm.write(f"[screen] {tag} OOM at batch "
                           f"{bs or cfg['batch_size']}, retrying smaller.")
        else:
            # Still OOM at batch 16: record the failure so one oversized
            # config can never kill the screen; refine will simply ignore it.
            model_obj = build_model(model, len(dataset.vocab.word2idx),
                                    len(dataset.vocab.char2idx), cfg)
            n_total, n_core = count_params(model_obj)
            del model_obj
            append_jsonl(SCREEN_LOG, {
                "model": model, "trial": tag, "config": cfg, "params_total": n_total,
                "params_core": n_core, "seconds": 0.0,
                "f1": 0.0, "acc": 0.0, "auc": 0.5, "note": "OOM-evict"})
            tqdm.write(f"[screen] {tag} OOM-evict (even at batch 16).")
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = dev_metrics(logits, dev_labels)
        model_obj = build_model(model, len(dataset.vocab.word2idx),
                                len(dataset.vocab.char2idx), cfg)
        n_total, _ = count_params(model_obj)
        del model_obj
        append_jsonl(SCREEN_LOG, {
            "model": model, "trial": tag, "config": cfg, "params_total": n_total,
            "params_core": n_core, "seconds": round(secs, 1),
            **{k: round(v, 4) for k, v in m.items()}})
        tqdm.write(f"[screen] {tag} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
                   f"core_params={n_core} ({secs:.0f}s){note}")
        return m["f1"]

    rng = random.Random(args.seed)
    for model in models:
        base_tag = f"{model}_baseline"
        if (model, base_tag) not in done:
            if should_stop():
                print("STOP requested, exiting cleanly.")
                return
            run_and_log(model, base_tag, BASELINES[model], args.seed, note=" (baseline)")
        else:
            tqdm.write(f"[screen] {base_tag} already done, skipping.")
        trial_iter = tqdm(range(args.n_configs), desc=f"[screen {model}]", unit="cfg")
        for i in trial_iter:
            tag = f"{model}_s{i:03d}"
            cfg = sample_config(rng, model)
            if (model, tag) in done:
                trial_iter.set_postfix_str("skipped (done)")
                continue
            if should_stop():
                print("STOP requested, exiting cleanly.")
                return
            f1 = run_and_log(model, tag, cfg, args.seed + i)
            trial_iter.set_postfix_str(f"F1={f1:.3f}")

    screen = read_jsonl(SCREEN_LOG)
    for model in models:
        rows = [r for r in screen if r["model"] == model and not r["trial"].endswith("_baseline")]
        base = next((r for r in screen if r["trial"] == f"{model}_baseline"), None)
        if rows and base:
            ranked = sorted(rows, key=lambda r: -r["f1"])
            mean_f1 = float(np.mean([r["f1"] for r in rows]))
            beaten = sum(r["f1"] > base["f1"] for r in rows)
            tqdm.write(f"[screen] {model} BASELINE F1={base['f1']:.4f} | best random "
                       f"F1={ranked[0]['f1']:.4f} ({ranked[0]['trial']}) | mean random "
                       f"F1={mean_f1:.4f} | random beats baseline: {beaten}/{len(rows)}")


def stage_refine(args, dataset, labels, device, models):
    seed_everything(args.seed)
    screen = [r for r in read_jsonl(SCREEN_LOG) if r["model"] in models]
    if len(screen) < len(models):
        raise SystemExit("Screen log is missing models; run --stage screen first.")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(len(dataset)), labels))
    done = {(r["model"], r["trial"]) for r in read_jsonl(REFINE_LOG)}
    for model in models:
        ranked = sorted([r for r in screen if r["model"] == model],
                        key=lambda r: (-r["f1"], r["params_core"]))[:args.top_k]
        candidates = [(r["trial"], r["config"]) for r in ranked] + [("baseline", BASELINES[model])]
        for tag, cfg in candidates:
            if (model, tag) in done:
                continue
            if should_stop():
                print("STOP requested, exiting cleanly.")
                return
            oof = np.zeros((len(dataset), 3))
            fold_f1s = []
            total_secs = 0.0
            fold_iter = tqdm(list(enumerate(folds)), desc=f"[refine {model}/{tag}]",
                             unit="fold", leave=False)
            for fi, (tr, dv) in fold_iter:
                seed_everything(args.seed + fi)
                dv_arr = np.array(dv)
                bs_try = [None, max(cfg["batch_size"] // 2, 16), 16]
                for bs in bs_try:
                    try:
                        logits, _, secs = fit_on_split(
                            model, cfg, dataset, tr, dv, args.full_epochs,
                            args.seed + fi, device, fold_tag=fi,
                            batch_override=bs)
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        tqdm.write(f"[refine] {model}/{tag} fold {fi} OOM, "
                                   f"retrying smaller.")
                else:
                    tqdm.write(f"[refine] {model}/{tag} fold {fi} OOM-evict; "
                               f"using zero logits for this fold.")
                    logits, secs = np.zeros((len(dv_arr), 3)), 0.0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oof[dv_arr] = logits
                fold_f1s.append(dev_metrics(logits, labels[dv_arr])["f1"])
                total_secs += secs
                fold_iter.set_postfix_str(f"F1={fold_f1s[-1]:.3f}")
                tqdm.write(f"[refine] {model}/{tag} fold {fi}: F1={fold_f1s[-1]:.4f} ({secs:.0f}s)")
            pooled = dev_metrics(oof, labels)
            model_obj = build_model(model, len(dataset.vocab.word2idx),
                                    len(dataset.vocab.char2idx), cfg)
            n_total, n_core = count_params(model_obj)
            del model_obj
            append_jsonl(REFINE_LOG, {
                "model": model, "trial": f"{model}_r_{tag}", "config": cfg,
                "params_total": n_total, "params_core": n_core,
                "oof_f1": round(pooled["f1"], 4), "oof_acc": round(pooled["acc"], 4),
                "oof_auc": round(pooled["auc"], 4),
                "fold_f1s": [round(v, 4) for v in fold_f1s],
                "seconds": round(total_secs, 1)})
            np.savez_compressed(SEARCH_DIR / f"oof_{model}_{tag}.npz", logits=oof, labels=labels)
            print(f"[refine] {model}/{tag} pooled OOF F1={pooled['f1']:.4f} "
                  f"AUC={pooled['auc']:.4f} folds={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}",
                  flush=True)


def pick_winners(models):
    refine = read_jsonl(REFINE_LOG)
    winners = {}
    for model in models:
        rows = [r for r in refine if r["model"] == model]
        if not rows:
            raise SystemExit(f"No refine rows for {model}; run --stage refine first.")
        best_f1 = max(r["oof_f1"] for r in rows)
        tied = [r for r in rows if best_f1 - r["oof_f1"] <= 0.002]
        winners[model] = min(tied, key=lambda r: r["params_core"])
    return winners


def calibrate_oof(logits, labels, folds):
    probs = np.zeros_like(logits, dtype=float)
    for tr, dv in folds:
        base = torch.tensor(logits[np.array(tr)])
        lab = torch.tensor(labels[np.array(tr)])

        def nll(t):
            return F.cross_entropy(base / t[0], lab).item()

        res = minimize(nll, [1.5], bounds=[(0.5, 5.0)], method="L-BFGS-B")
        probs[np.array(dv)] = torch.softmax(
            torch.tensor(logits[np.array(dv)]) / res.x[0], dim=1).numpy()
    return probs


def stage_stack(args, dataset, labels, models):
    winners = pick_winners(models)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
        np.zeros(len(dataset)), labels))
    feats = []
    for model in models:
        tag = winners[model]["trial"].split("_r_")[-1]
        z = np.load(SEARCH_DIR / f"oof_{model}_{tag}.npz")
        feats.append(calibrate_oof(z["logits"], labels, folds))
    X = np.concatenate(feats, axis=1)
    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(X, labels))
    results = []
    for c in STACK_SPACE_LR:
        f1s = []
        for tr, dv in outer:
            clf = LogisticRegression(max_iter=1000, C=c)
            clf.fit(X[tr], labels[tr])
            f1s.append(f1_score(labels[dv], clf.predict(X[dv]), average="macro"))
        results.append({"stacker": "logreg", "params": {"C": c},
                        "mean_f1": round(float(np.mean(f1s)), 4),
                        "std_f1": round(float(np.std(f1s)), 4)})
    for hp in STACK_SPACE_XGB:
        f1s = []
        for tr, dv in outer:
            clf = XGBClassifier(**hp, subsample=0.8, colsample_bytree=0.8,
                                eval_metric="mlogloss", random_state=42)
            clf.fit(X[tr], labels[tr])
            f1s.append(f1_score(labels[dv], clf.predict(X[dv]), average="macro"))
        results.append({"stacker": "xgb", "params": hp,
                        "mean_f1": round(float(np.mean(f1s)), 4),
                        "std_f1": round(float(np.std(f1s)), 4)})
    results.sort(key=lambda r: -r["mean_f1"])
    with open(SEARCH_DIR / "stack_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[stack] best: {results[0]['stacker']} {results[0]['params']} "
          f"F1={results[0]['mean_f1']:.4f}±{results[0]['std_f1']:.4f}", flush=True)


def stage_finalize(args, dataset, labels, device, models):
    seed_everything(args.seed)
    winners = pick_winners(models)
    val_dataset = SeverityDataset("cache/severity_val_hq.json", dataset.vocab)
    val_labels = np.array([item_label(d) for d in val_dataset.data])
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            collate_fn=collate_nosort_fn)

    summary = {"protocol": "CV-only selection on train1600; val400 used only for "
                           "checkpointing and a caveated final report.",
               "models": {}, "bootstrap": {}}
    for model in models:
        cfg = winners[model]["config"]
        bs = cfg["batch_size"]
        train_loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        sub_labels = labels.tolist()
        counts = [max(sub_labels.count(c), 1) for c in range(3)]
        weights = torch.FloatTensor([len(sub_labels) / c for c in counts]).to(device)
        weights = weights / weights.sum() * 3.0
        net = build_model(model, len(dataset.vocab.word2idx),
                          len(dataset.vocab.char2idx), cfg).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        smooth, alpha = cfg["label_smoothing"], cfg["alpha"]
        gamma = cfg.get("focal_gamma", 0.0)
        if gamma:
            smooth = 0.0
        best_loss, best_state = float("inf"), None
        final_iter = tqdm(range(args.full_epochs), desc=f"[finalize {model}]",
                          unit="ep", leave=False)
        for _ in final_iter:
            net.train()
            for word_ids, char_ids, lengths, hard_labels, soft_labels in train_loader:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
                opt.zero_grad()
                logits = net(word_ids, char_ids, lengths)
                hard_loss = focal_or_ce_loss(logits, hard_labels, weights, smooth, gamma)
                soft_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                (alpha * hard_loss + (1 - alpha) * soft_loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
            inner_loss = combined_losses(net, val_loader, device, weights, alpha, smooth, gamma)
            if inner_loss < best_loss:
                best_loss = inner_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save(best_state, SEARCH_DIR / f"final_ensemble_{model}.pt")
        val_logits = []
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            for word_ids, char_ids, lengths, *_ in val_loader:
                val_logits.append(net(word_ids.to(device), char_ids.to(device), lengths).cpu().numpy())
        vm = dev_metrics(np.concatenate(val_logits), val_labels)
        summary["models"][model] = {
            "winner_trial": winners[model]["trial"], "config": cfg,
            "params_total": winners[model]["params_total"],
            "params_core": winners[model]["params_core"],
            "oof_f1": winners[model]["oof_f1"], "oof_auc": winners[model]["oof_auc"],
            "fold_f1s": winners[model]["fold_f1s"],
            "val400_report_caveated": {k: round(v, 4) for k, v in vm.items()},
        }
        print(f"[finalize] {model} val400: F1={vm['f1']:.4f} AUC={vm['auc']:.4f} "
              f"(caveated: val picked the checkpoint)", flush=True)

    rng = np.random.default_rng(args.seed)
    for model in models:
        wtag = winners[model]["trial"].split("_r_")[-1]
        w = np.load(SEARCH_DIR / f"oof_{model}_{wtag}.npz")
        b = np.load(SEARCH_DIR / f"oof_{model}_baseline.npz")
        wpred, bpred = w["logits"].argmax(1), b["logits"].argmax(1)
        y, n = labels, len(labels)
        wf1 = [f1_score(y[i], wpred[i], average="macro") for i in
               (rng.integers(0, n, n) for _ in range(2000))]
        bf1 = [f1_score(y[i], bpred[i], average="macro") for i in
               (rng.integers(0, n, n) for _ in range(2000))]
        diffs = [a - b_ for a, b_ in zip(wf1, bf1)]
        summary["bootstrap"][model] = {
            "winner_f1_95ci": [round(float(np.percentile(wf1, 2.5)), 4),
                               round(float(np.percentile(wf1, 97.5)), 4)],
            "baseline_f1_95ci": [round(float(np.percentile(bf1, 2.5)), 4),
                                 round(float(np.percentile(bf1, 97.5)), 4)],
            "winner_minus_baseline_95ci": [round(float(np.percentile(diffs, 2.5)), 4),
                                           round(float(np.percentile(diffs, 97.5)), 4)],
            "p_winner_better": round(float(np.mean([d > 0 for d in diffs])), 4),
        }

    rows = []
    for r in read_jsonl(SCREEN_LOG) + read_jsonl(REFINE_LOG):
        if r["model"] in models:
            rows.append({"Model": r["model"], "NonEmbedParams": r["params_core"],
                         "F1": r["oof_f1"] if "oof_f1" in r else r["f1"],
                         "Kind": "trial"})
    for model in models:
        w = summary["models"][model]
        rows.append({"Model": model, "NonEmbedParams": w["params_core"],
                     "F1": w["oof_f1"], "Kind": "winner"})
        brow = next(r for r in read_jsonl(REFINE_LOG)
                    if r["model"] == model and r["trial"].endswith("baseline"))
        rows.append({"Model": model, "NonEmbedParams": brow["params_core"],
                     "F1": brow["oof_f1"], "Kind": "baseline"})
    plots.save_pareto_scatter(rows, SEARCH_DIR / "pareto.png")
    with open(SEARCH_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print("[finalize] wrote summary.json, pareto.png, final weights, bootstrap CIs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ensemble architecture + hyperparameter search")
    parser.add_argument("--stage", default="screen",
                        choices=["screen", "refine", "stack", "finalize", "all"])
    parser.add_argument("--model", default="all",
                        choices=["all"] + MODEL_NAMES)
    parser.add_argument("--n-configs", type=int, default=96)
    parser.add_argument("--screen-epochs", type=int, default=25)
    parser.add_argument("--full-epochs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true",
                        help="2 configs, 1 epoch, FastClinicalCNN only; validates the harness")
    args = parser.parse_args()

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.n_configs, args.screen_epochs, args.top_k = 2, 1, 1
        args.model = "FastClinicalCNN"
    models = MODEL_NAMES if args.model == "all" else [args.model]
    stages = {"screen": ["screen"], "refine": ["refine"], "stack": ["stack"],
              "finalize": ["finalize"],
              "all": ["screen", "refine", "stack", "finalize"]}[args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | stage={args.stage} models={models} "
          f"n_configs={args.n_configs} screen_epochs={args.screen_epochs} "
          f"full_epochs={args.full_epochs} top_k={args.top_k}", flush=True)
    dataset, labels = load_data()
    print(f"Train rows: {len(dataset)} | "
          f"label counts: {[int((labels == c).sum()) for c in range(3)]}", flush=True)

    if "screen" in stages:
        stage_screen(args, dataset, labels, device, models)
    if "refine" in stages:
        stage_refine(args, dataset, labels, device, models)
    if "stack" in stages:
        stage_stack(args, dataset, labels, models)
    if "finalize" in stages:
        stage_finalize(args, dataset, labels, device, models)


if __name__ == "__main__":
    main()
