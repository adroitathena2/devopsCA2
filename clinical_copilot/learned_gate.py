"""Learned codebook gate for DDI severity (Severe-vs-rest).

Replaces the brittle regex gate in clinical_copilot/ddi.py
(DDIClassifier._is_genuinely_mild + SEVERE_MIN_PROB/SEVERE_MARGIN).

DESIGN CHOICE — option (a) from the task spec:
  * The learned model is Severe-vs-rest ONLY (binary LogisticRegression).
  * The Mild path is untouched: DDIClassifier._is_genuinely_mild stays a
    deterministic allowlist. With only 6 Mild rows in val-400, learning
    Mild-side weights would be noise; no Mild decision boundary is learned.
  * Per-class sample weights {Severe:10, Moderate:2, Mild:1} encode that the
    gate exists to recover Severe misses (10 in production confusion) while
    barely nudging the Moderate majority.

Features (17 dims): 6 numeric from the FROZEN stacker path
(p_mild, p_mod, p_sev, p_sev-p_mod, entropy, len(words)) + 10 binary
keyword cues matching the ddi.py:558-561 severity codebook.
Base nets (FastClinicalCNN + TinyClinicalFormer) and the production
LogReg stacker (C=0.1, 6 features) are never retrained here.
"""

import json
import re
from pathlib import Path

import numpy as np

GATE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "learned_gate"
GATE_PATH = GATE_DIR / "gate.pkl"
GATE_META_PATH = GATE_DIR / "gate_meta.json"

FEATURE_NAMES = [
    "p_mild", "p_mod", "p_sev", "p_sev_minus_p_mod", "entropy", "n_words",
    "cue_qtc_torsades", "cue_bleed_event", "cue_bleed_risk",
    "cue_arrhythmia", "cue_rhabdo", "cue_serotonin", "cue_renal_failure",
    "cue_organ_tox", "cue_excretion_serum", "cue_gi_only",
    "cue_severe_event",
]

_RX = {
    "qtc_torsades": re.compile(r"qtc|torsad", re.I),
    "bleed": re.compile(r"bleed\w*|hemorrhag\w*|hematemesis|melena|hemoptysis", re.I),
    "bleed_risk": re.compile(r"bleed\w*.{0,30}risk|risk.{0,30}bleed\w*|hemorrhag\w*.{0,30}risk|risk.{0,30}hemorrhag\w*", re.I),
    "arrhythmia": re.compile(r"arrhythmia|ventricular|cardiac arrest|fibrillation|bradycardia|tachycardia", re.I),
    "rhabdo": re.compile(r"rhabdomyolysis|myopathy|myoglobinuria", re.I),
    "serotonin": re.compile(r"serotonin syndrome", re.I),
    "renal_failure": re.compile(r"renal failure|kidney failure|acute renal", re.I),
    "organ_tox": re.compile(r"nephrotoxic|cardiotoxic|neurotoxic|hepatotoxic|ototoxic|"
                            r"pulmonary toxicity|liver damage|\btoxicity\b|\btoxic\b", re.I),
    "excretion": re.compile(r"excretion", re.I),
    "serum": re.compile(r"serum level|serum concentration", re.I),
    "gi_only": re.compile(r"gastrointestinal irritation|gastric irritation", re.I),
    # Aggregate OR of individually-rare Severe-codebook events (ddi.py:559):
    # each is too rare in val-400 for its own weight, but as a class they are
    # near-pathognomonic for Severe. Keeps dim count at 17.
    "severe_event": re.compile(
        r"neuromuscular block|paralysis|apnea|anaphylaxis|hypersensitivity|"
        r"respiratory depress|respiratory failure|coma|seizure|convulsion|"
        r"thrombosis|thromboembol|methemoglobinemia|angioedema", re.I),
}


def extract_gate_features(description: str, probs) -> np.ndarray:
    """Build the 17-dim gate feature vector.

    Args:
        description: interaction description text.
        probs: frozen stacker [p_mild, p_mod, p_sev].
    """
    t = description or ""
    p = np.asarray(probs, dtype=float).ravel()
    assert p.shape == (3,), f"expected 3 stacker probs, got {p.shape}"
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    n_words = len(t.split())

    has_bleed = bool(_RX["bleed"].search(t))
    bleed_risk = bool(_RX["bleed_risk"].search(t))

    row = [
        float(p[0]), float(p[1]), float(p[2]),
        float(p[2] - p[1]), ent, float(n_words),
        float(bool(_RX["qtc_torsades"].search(t))),
        float(has_bleed and not bleed_risk),   # bleeding as an EVENT (severe cue)
        float(has_bleed and bleed_risk),        # bleeding RISK only (moderate cue)
        float(bool(_RX["arrhythmia"].search(t))),
        float(bool(_RX["rhabdo"].search(t))),
        float(bool(_RX["serotonin"].search(t))),
        float(bool(_RX["renal_failure"].search(t))),
        float(bool(_RX["organ_tox"].search(t))),
        float(bool(_RX["excretion"].search(t)) and bool(_RX["serum"].search(t))),
        float(bool(_RX["gi_only"].search(t))),
        float(bool(_RX["severe_event"].search(t))),
    ]
    return np.array(row, dtype=float)


def featurize(descriptions, prob_matrix) -> np.ndarray:
    return np.vstack([extract_gate_features(t, p) for t, p in zip(descriptions, prob_matrix)])


def load_gate():
    """Load (model, threshold, meta). Raises FileNotFoundError if untrained."""
    import pickle

    with open(GATE_PATH, "rb") as f:
        model = pickle.load(f)
    with open(GATE_META_PATH, "r") as f:
        meta = json.load(f)
    return model, float(meta["threshold"]), meta


def gate_severe_prob(description: str, probs) -> float:
    """P(Severe) from the learned gate for one description."""
    model, _, _ = load_gate()
    x = extract_gate_features(description, probs).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])
