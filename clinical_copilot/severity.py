"""Single source of truth for DDI severity labels and mappings.

The severity classifier's class space follows the ensemble training data:
Mild (0), Moderate (1), Severe (2). "No Interaction" is not a classifier class;
it is the prescription-level outcome when no interacting pair is found.
"""

import re

NO_INTERACTION = "No Interaction"
SEVERITY_LABELS = ("Mild", "Moderate", "Severe")
OUTCOME_LABELS = (NO_INTERACTION,) + SEVERITY_LABELS

RAW_RANKS = {
    NO_INTERACTION: 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "VERY SEVERE": 3,
}

RISK_SCORES = {
    NO_INTERACTION: 0.05,
    "Mild": 0.25,
    "Moderate": 0.75,
    "Severe": 0.95,
}


def _clean(raw) -> str:
    if raw is None:
        return ""
    text = str(raw).strip().replace("\n", " ").upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(raw, strict: bool = False):
    """Map any raw severity string to a severity label, or to the no-interaction outcome."""
    text = _clean(raw)
    if text == "MILD":
        return "Mild"
    if text == "MODERATE":
        return "Moderate"
    if text in ("SEVERE", "VERY SEVERE"):
        return "Severe"
    if text in ("", "NONE", "NO", "NO INTERACTION", "NO INTERACTIONS"):
        return None if strict else NO_INTERACTION
    return None if strict else NO_INTERACTION


def rank(raw) -> int:
    """Severity rank (0 = no interaction, 3 = severe)."""
    return RAW_RANKS.get(normalize(raw), 0)


def class_index(raw) -> int:
    """Outcome index (0 = No Interaction, 1..3 = severity classes)."""
    return OUTCOME_LABELS.index(normalize(raw))


def risk_score(raw) -> float:
    """Serious-risk score used by the binary DDI metrics (Mild is non-serious)."""
    return RISK_SCORES[normalize(raw)]


def max_label(labels) -> str:
    """Highest-ranked outcome across labels; empty input means no interaction."""
    best = NO_INTERACTION
    for label in labels:
        candidate = normalize(label)
        if rank(candidate) > rank(best):
            best = candidate
    return best
