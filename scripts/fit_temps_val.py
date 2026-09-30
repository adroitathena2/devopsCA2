"""Fit ensemble temperature scalars on severity_val_hq (held-out) NLL.

Replaces the train in-sample fit in evaluate_ensemble.py, which is a no-op
by construction (init 1.5 is already near-optimal in-sample, so temps never
move). Val picked the base checkpoints, so note the small reuse caveat: 3
scalars on 400 rows cannot meaningfully overfit, and train in-sample fits
nothing at all.

Writes cache/ensemble_temps.json (backs up the old file next to it).

Run with uv:
    uv run fit_temps_val.py
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.utils.data import DataLoader

from clinical_copilot.custom_severity_model import SeverityDataset, SeverityVocab, build_ensemble_model, collate_fn, ENSEMBLE_NAMES

MODEL_NAMES = list(ENSEMBLE_NAMES)


def gather_val_logits(device):
    vocab = SeverityVocab()
    vocab.load("cache/severity_vocab.json")
    ds = SeverityDataset("cache/severity_val_hq.json", vocab)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    out = {}
    labels = None
    for name in MODEL_NAMES:
        model = build_ensemble_model(name, len(vocab.word2idx), len(vocab.char2idx))
        model.load_state_dict(torch.load(f"cache/ensemble_{name}.pt", map_location=device))
        model = model.to(device)
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for word_ids, char_ids, lengths, hard_labels, _ in loader:
                logits = model(word_ids.to(device), char_ids.to(device), lengths)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(hard_labels.numpy())
        out[name] = np.concatenate(all_logits)
        if labels is None:
            labels = np.concatenate(all_labels)
    return out, labels


def fit_temp(logits, labels):
    base = torch.tensor(logits)
    lab = torch.tensor(labels)

    def nll(t):
        return F.cross_entropy(base / t[0], lab).item()

    res = minimize(nll, [1.5], bounds=[(0.5, 5.0)], method="L-BFGS-B")
    return float(res.x[0]), nll([1.5]), nll(res.x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    logits, labels = gather_val_logits(device)
    print(f"Val rows: {len(labels)}")

    old_path = Path("cache/ensemble_temps.json")
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    print(f"Old temps: {old}")

    new_temps = {}
    for name in MODEL_NAMES:
        t, nll_before, nll_after = fit_temp(logits[name], labels)
        new_temps[name] = round(t, 4)
        print(f"{name}: T={t:.4f} (was 1.5) val NLL {nll_before:.4f} -> {nll_after:.4f}")

    shutil.copy(old_path, old_path.with_suffix(".json.bak"))
    old_path.write_text(json.dumps(new_temps, indent=2))
    print(f"Wrote {old_path} (old saved as .bak): {new_temps}")


if __name__ == "__main__":
    main()
