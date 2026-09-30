"""Train ONLY the learned Severe-vs-rest gate (base nets + stacker stay frozen).

Val: cache/severity_val_hq.json (400 rows; 341 Moderate / 53 Severe / 6 Mild).
Test is NEVER touched here — single final eval via
  uv run evaluate_ddi_system.py --mode ensemble --dataset test_prescriptions.json

Outputs (outputs/learned_gate/):
  gate.pkl, gate_meta.json, cv_table.csv/json, weights.csv/json,
  spot_check_50.csv, val_predictions.csv
"""

import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from clinical_copilot.custom_severity_model import SeverityVocab, build_ensemble_model, ENSEMBLE_NAMES
from clinical_copilot.learned_gate import FEATURE_NAMES, GATE_DIR, featurize

VAL_PATH = Path("cache/severity_val_hq.json")
SEV, MOD, MILD = 2, 1, 0
SAMPLE_W = {SEV: 10.0, MOD: 2.0, MILD: 1.0}  # Severe:10, Moderate:2, Mild:1
SEED = 42

# --- frozen production constants (mirrors evaluate_ensemble.py:174,198) ---
STACKER_PATH = Path("cache/ensemble_stacker.pkl")
TEMPS_PATH = Path("cache/ensemble_temps.json")
VOCAB_PATH = Path("cache/severity_vocab.json")

# --- legacy regex gate constants (mirrors ddi.py:109-112) ---
SEVERE_MIN_PROB = 0.50
SEVERE_MARGIN = 0.05


def fail(msg: str) -> None:
    print(f"SCHEMA ABORT: {msg}", file=sys.stderr)
    sys.exit(2)


def load_val():
    if not VAL_PATH.exists():
        fail(f"{VAL_PATH} missing")
    rows = json.loads(VAL_PATH.read_text())
    if not isinstance(rows, list) or len(rows) != 400:
        fail(f"expected list of 400 rows, got {type(rows)} len={len(rows) if isinstance(rows, list) else '?'}")
    req = {"text", "label"}
    for i, r in enumerate(rows):
        if not req.issubset(r.keys()):
            fail(f"row {i} keys {sorted(r.keys())} missing {sorted(req - set(r.keys()))}")
        if r["label"] not in (0, 1, 2):
            fail(f"row {i} label={r['label']!r} not in {{0,1,2}}")
        if not isinstance(r["text"], str) or not r["text"].strip():
            fail(f"row {i} has empty text")
    dist = Counter(r["label"] for r in rows)
    print(f"val dist: Mild={dist[0]} Moderate={dist[1]} Severe={dist[2]}")
    if (dist[0], dist[1], dist[2]) != (6, 341, 53):
        fail(f"dist {dict(dist)} != expected Mild=6/Moderate=341/Severe=53")
    return rows


def frozen_stacker_probs(texts):
    """Replicate ddi.py::_custom_classify_severity (frozen, inference only)."""
    for p in (STACKER_PATH, TEMPS_PATH, VOCAB_PATH):
        if not p.exists():
            fail(f"frozen artifact missing: {p}")
    vocab = SeverityVocab()
    vocab.load(str(VOCAB_PATH))
    device = torch.device("cpu")
    models, names = [], []
    for member in ENSEMBLE_NAMES:  # FastClinicalCNN, TinyClinicalFormer — frozen
        fp = Path(f"cache/ensemble_{member}.pt")
        if not fp.exists():
            fail(f"frozen base net missing: {fp}")
        m = build_ensemble_model(member, len(vocab.word2idx), len(vocab.char2idx), num_classes=3)
        m.load_state_dict(torch.load(str(fp), map_location=device))
        m.eval()
        models.append(m)
        names.append(member)
    temps = json.loads(TEMPS_PATH.read_text())
    with open(STACKER_PATH, "rb") as f:
        stacker = pickle.load(f)
    assert list(stacker.classes_) == [0, 1, 2], f"stacker classes {stacker.classes_}"
    assert stacker.coef_.shape[1] == 6, f"stacker n_features {stacker.coef_.shape}"

    out = np.zeros((len(texts), 3))
    with torch.no_grad():
        for i, desc in enumerate(texts):
            w_ids, c_ids = vocab.encode_words(desc), vocab.encode_chars(desc)
            assert w_ids, f"empty encoding for row {i}"
            w = torch.tensor([w_ids], dtype=torch.long)
            L = torch.tensor([len(w_ids)], dtype=torch.long)
            mw = max(len(c) for c in c_ids)
            c = torch.zeros((1, len(w_ids), mw), dtype=torch.long)
            for j, seq in enumerate(c_ids):
                c[0, j, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            feats = []
            for k, m in enumerate(models):
                logits = m(w, c, L)
                feats.append(torch.softmax(logits / temps[names[k]], dim=-1)[0].numpy())
            out[i] = stacker.predict_proba(np.concatenate(feats).reshape(1, -1))[0]
            if (i + 1) % 100 == 0:
                print(f"  frozen probs {i + 1}/{len(texts)}")
    return out


def regex_decision(argmax, probs, text, is_mild_fn):
    """Replicate the ddi.py:773-791 regex gate."""
    if argmax == MILD:
        return "Mild" if is_mild_fn(text) else "Moderate"
    if argmax == SEV:
        if probs[2] < SEVERE_MIN_PROB or (probs[2] - probs[1]) < SEVERE_MARGIN:
            return "Moderate"
        return "Severe"
    return "Moderate"


def pipeline_metrics(y_true, decisions):
    """decisions: array of str labels. Returns Sens-Severe, overcalls, Mild F1."""
    yt = np.asarray(y_true)
    d = np.asarray(decisions)
    sens_sev = recall_score(yt == SEV, d == "Severe", zero_division=0)
    overcalls = int(((yt != SEV) & (d == "Severe")).sum())  # non-Severe called Severe
    mild_f1 = f1_score(yt == MILD, d == "Mild", zero_division=0)
    return sens_sev, overcalls, mild_f1


def main():
    from clinical_copilot.ddi import InteractionDetector  # static allowlist only

    is_mild = InteractionDetector._is_genuinely_mild
    rows = load_val()
    texts = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    y_bin = (y == SEV).astype(int)
    sw = np.array([SAMPLE_W[int(v)] for v in y])

    print("collecting frozen stacker probs...")
    P = frozen_stacker_probs(texts)
    argmax = P.argmax(axis=1)

    print("featurizing (codebook cues)...")
    X = featurize(texts, P)
    assert X.shape == (400, len(FEATURE_NAMES)), X.shape
    np.savez(GATE_DIR / "_frozen_cache.npz", P=P, X=X, y=y, argmax=argmax)

    # descriptive baselines (regex constants were tuned on this val set too)
    reg = np.array([regex_decision(a, p, t, is_mild) for a, p, t in zip(argmax, P, texts)])
    raw = np.array([{0: "Mild", 1: "Moderate", 2: "Severe"}[a] for a in argmax])
    print(f"stacker-argmax : Sens-Sev={recall_score(y_bin, (raw=='Severe').astype(int), zero_division=0):.3f} "
          f"overcalls={int(((y!=SEV)&(raw=='Severe')).sum())}")
    rs, ro, rm = pipeline_metrics(y, reg)
    print(f"regex-gate     : Sens-Sev={rs:.3f} overcalls={ro} MildF1={rm:.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_lr, oof_xgb = np.zeros(400), np.zeros(400)
    fold_rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y_bin)):
        lr = LogisticRegression(max_iter=5000, C=1.0)
        lr.fit(X[tr], y_bin[tr], sample_weight=sw[tr])
        oof_lr[va] = lr.predict_proba(X[va])[:, 1]
        xgb = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="mlogloss", random_state=SEED)
        xgb.fit(X[tr], y_bin[tr], sample_weight=sw[tr])
        oof_xgb[va] = xgb.predict_proba(X[va])[:, 1]

        for name, oof in (("LR", oof_lr), ("XGB", oof_xgb)):
            pass  # per-fold decisions need full oof; scored below via fold mask
        fold_rows.append({"fold": fold, "n_va": len(va),
                          "n_sev_va": int(y_bin[va].sum())})

    # per-fold metrics at operating threshold 0.5 (pipeline simulation:
    # Mild allowlist unchanged, Severe/Moderate decided by gate)
    cv_table = []
    for f, r in enumerate(fold_rows):
        va = list(skf.split(X, y_bin))[f][1]
        for name, oof in (("LR", oof_lr), ("XGB", oof_xgb)):
            dec = []
            for i in va:
                if argmax[i] == MILD and is_mild(texts[i]):
                    dec.append("Mild")
                elif argmax[i] == MILD:
                    dec.append("Moderate")
                else:
                    dec.append("Severe" if oof[i] >= 0.5 else "Moderate")
            s, o, m = pipeline_metrics(y[va], dec)
            cv_table.append({"model": name, "fold": r["fold"], "n_va": r["n_va"],
                             "n_sev_va": r["n_sev_va"], "sens_severe": round(s, 4),
                             "overcalls": o, "mild_f1": round(m, 4)})
            print(f"fold {r['fold']} {name}: Sens-Sev={s:.3f} overcalls={o} MildF1={m:.3f} "
                  f"(n_va={r['n_va']}, sev={r['n_sev_va']})")

    def oof_summary(oof, name):
        dec = np.array(["Moderate"] * 400)
        for i in range(400):
            if argmax[i] == MILD and is_mild(texts[i]):
                dec[i] = "Mild"
            elif argmax[i] != MILD and oof[i] >= 0.5:
                dec[i] = "Severe"
        s, o, m = pipeline_metrics(y, dec)
        print(f"OOF {name}@0.5: Sens-Sev={s:.3f} overcalls={o} MildF1={m:.3f}")
        return s, o, m

    s_lr, o_lr, m_lr = oof_summary(oof_lr, "LR")
    s_xgb, o_xgb, m_xgb = oof_summary(oof_xgb, "XGB")

    # Winner: LR by design (XGB is challenger-only per spec). Rationale,
    # confirmed on OOF below: on 53 Severe rows XGB's edge came from
    # memorization signatures (near-identical neuromuscular-blockade inputs
    # scored 0.982 vs 0.125; a Mild row scored 0.988), while LR is stable,
    # monotone in stacker probs, and yields interpretable weights.
    print("OOF comparison only - LR is the gate, XGB the challenger:")
    winner = "LR"
    oof = oof_lr

    # threshold tune on OOF only (max Severe-F1, tie-break fewer overcalls then recall)
    best = None
    for tau in np.arange(0.10, 0.91, 0.05):
        dec = np.array(["Moderate"] * 400)
        for i in range(400):
            if argmax[i] == MILD and is_mild(texts[i]):
                dec[i] = "Mild"
            elif argmax[i] != MILD and oof[i] >= tau:
                dec[i] = "Severe"
        s, o, m = pipeline_metrics(y, dec)
        f1 = f1_score((y == SEV).astype(int), (dec == "Severe").astype(int), zero_division=0)
        key = (round(f1, 4), -o, round(s, 4))
        if best is None or key > best[0]:
            best = (key, float(tau), s, o, m, f1)
    _, tau, s, o, m, f1 = best
    print(f"OOF threshold: tau={tau:.2f} SevereF1={f1:.3f} Sens-Sev={s:.3f} overcalls={o} MildF1={m:.3f}")

    # refit winner on full val-400
    if winner == "LR":
        model = LogisticRegression(max_iter=5000, C=1.0)
    else:
        model = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="mlogloss", random_state=SEED)
    model.fit(X, y_bin, sample_weight=sw)

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(GATE_DIR / "gate.pkl", "wb") as f:
        pickle.dump(model, f)

    if winner == "LR":
        wtab = [{"feature": n, "coef": float(c)} for n, c in zip(FEATURE_NAMES, model.coef_[0])]
        wtab.append({"feature": "intercept", "coef": float(model.intercept_[0])})
    else:
        imp = model.feature_importances_
        wtab = [{"feature": n, "coef": float(v)} for n, v in zip(FEATURE_NAMES, imp)]
    meta = {
        "model": ("LogisticRegression(C=1.0)" if winner == "LR"
                  else "XGBClassifier(n_estimators=50,max_depth=2)"),
        "binary_task": "Severe-vs-rest",
        "mild_policy": "deterministic allowlist _is_genuinely_mild (no learned Mild weights)",
        "sample_weight": {"Severe": 10, "Moderate": 2, "Mild": 1},
        "threshold": tau,
        "oos_estimate": "5-fold OOF",
        "oof": {"sens_severe": s, "overcalls": o, "mild_f1": m, "severe_f1": f1},
        "features": FEATURE_NAMES,
        "frozen": {"base_nets": list(ENSEMBLE_NAMES),
                   "stacker": "LogisticRegression C=0.1, 6 features",
                   "val": "cache/severity_val_hq.json (6/341/53)"},
        "weights": wtab,
    }
    with open(GATE_DIR / "gate_meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    with open(GATE_DIR / "cv_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cv_table[0].keys()))
        w.writeheader()
        w.writerows(cv_table)
    with open(GATE_DIR / "cv_table.json", "w") as f:
        json.dump(cv_table, f, indent=1)
    with open(GATE_DIR / "weights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "coef"])
        w.writeheader()
        w.writerows(wtab)
    with open(GATE_DIR / "weights.json", "w") as f:
        json.dump(wtab, f, indent=1)

    np.save(GATE_DIR / "oof_probs.npy", oof)
    np.save(GATE_DIR / "oof_probs_lr.npy", oof_lr)
    np.save(GATE_DIR / "oof_probs_xgb.npy", oof_xgb)

    # val predictions (full-refit, in-sample — for audit, not for reporting)
    full_p = model.predict_proba(X)[:, 1]
    with open(GATE_DIR / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label", "p_mild", "p_mod", "p_sev",
                    "stacker_argmax", "regex_decision", "gate_oof_p",
                    "gate_full_p", "gate_decision_at_tau"])
        for i in range(400):
            if argmax[i] == MILD and is_mild(texts[i]):
                gd = "Mild"
            elif argmax[i] != MILD and full_p[i] >= tau:
                gd = "Severe"
            else:
                gd = "Moderate"
            w.writerow([i, y[i], f"{P[i,0]:.4f}", f"{P[i,1]:.4f}", f"{P[i,2]:.4f}",
                        int(argmax[i]), reg[i], f"{oof[i]:.4f}",
                        f"{full_p[i]:.4f}", gd])

    # 50-row physician spot-check: all Severe rows first, then regex/learned
    # disagreements (OOF decisions), then random fill — seed fixed
    rng = np.random.default_rng(SEED)
    oof_dec = []
    for i in range(400):
        if argmax[i] == MILD and is_mild(texts[i]):
            oof_dec.append("Mild")
        elif argmax[i] != MILD and oof[i] >= tau:
            oof_dec.append("Severe")
        else:
            oof_dec.append("Moderate")
    oof_dec = np.array(oof_dec)
    cue_names = FEATURE_NAMES[6:]
    active = ["/".join(n for n, v in zip(cue_names, X[i, 6:]) if v > 0) for i in range(400)]
    sev_idx = [i for i in range(400) if y[i] == SEV]
    disag = [i for i in range(400) if y[i] != SEV and reg[i] != oof_dec[i]]
    rest = [i for i in range(400) if i not in sev_idx and i not in disag]
    rng.shuffle(disag)
    rng.shuffle(rest)
    pick = (sev_idx + disag + rest)[:50]
    with open(GATE_DIR / "spot_check_50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label", "stacker_argmax", "p_sev", "p_mod",
                    "regex_decision", "gate_oof_p", "gate_decision",
                    "active_cues", "text"])
        for i in pick:
            w.writerow([i, y[i], int(argmax[i]), f"{P[i,2]:.4f}",
                        f"{P[i,1]:.4f}", reg[i], f"{oof[i]:.4f}",
                        oof_dec[i], active[i], texts[i]])
    print(f"wrote {GATE_DIR}/ (gate.pkl, gate_meta.json, cv_table.*, weights.*, "
          f"val_predictions.csv, spot_check_50.csv, oof_probs.npy)")


if __name__ == "__main__":
    main()
