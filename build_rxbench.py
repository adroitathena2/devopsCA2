"""RxBench-India + Rx-Noise benchmark builder.

READ-ONLY w.r.t. test_prescriptions.json, cache/, and production weights:
this script never writes to those paths. It only reads
  - test_prescriptions.json (233 Rx, GT pairs with ingredients)
  - data/indian_brands.json (brand -> ingredient(s) lexicon)
  - cache/medicine_data.json (DrugBank canonical names, for normalization)
and writes everything under outputs/rxbench/.

GT-PRESERVATION RULE (Rx-Noise):
  The interacting drug-pair tokens themselves are NEVER dropped or truncated.
  L1/L2 apply only character-level perturbations (which may touch drug tokens
  by a single character, exactly as a typo/OCR error would); L3 structural
  ops (dose-drop, truncation) remove only dose strengths and surrounding
  instruction text. Every L3 output is post-checked: each drug surface listed
  in the Rx `drugs` field must still occur verbatim (case-insensitive) in the
  L3 text, otherwise that Rx falls back to the dose-dropped-only variant and
  the fallback is logged. GT `interactions` / `has_interaction` fields are
  copied unchanged into every split.

Brand substitution (RxBench-India):
  Each generic slot is substituted independently with the top verified:true
  SINGLE-ingredient brand for that ingredient, sorted by n_rows (desc,
  brand-name tiebreak for determinism). Multi-ingredient (combo) brands are
  never used to cover a generic slot, so e.g. an amoxicillin slot never
  becomes Augmentin (amoxicillin+clavulanic acid); each ingredient maps to
  its own single-ingredient brand. Tokens that are already brands in the
  lexicon are left as-is; generics with no verified single-ingredient brand
  are left as-is and logged in coverage.json.

Noise levels (all derived from the BRAND text, fixed seed recorded in file):
  L1 single-char : seeded <=2 single-char substitutions from
                  {e->c, rn->m, 0->O} at eligible sites.
  L2 glued/OCR   : case-strip (lowercase) first, then glue drug-dose gaps
                  ("Simvastatin 20mg" -> "simvastatin20mg"), then OCR word
                  maps (daily->daiIy, vitamin->v1tam1n). Order matters:
                  lowercasing first keeps the OCR case markers intact.
  L3 structural  : Hinglish prefix (seeded choice, e.g. "bukhar ke liye "),
                  dose-drop (strength tokens like 10mg removed, drug names
                  kept), truncation of trailing instruction text on
                  drug-bearing lines (drug surfaces always preserved).

Usage:
  uv run build_rxbench.py [--seed 20260924]
Outputs (all under outputs/rxbench/):
  rxbench-india.json      evaluator-ready list (brand-substituted texts)
  rxbench-india.meta.json meta + per-Rx linking map + coverage counts
  rxbench-noise.json      {meta:{seed,...}, splits:{L1:[...],L2:[...],L3:[...]}}
  split_clean.json / split_brand.json / split_L1.json / split_L2.json /
    split_L3.json         evaluator-ready per-split files
  spotcheck_50.csv        50-row clean -> brand / L1 / L2 / L3 side-by-side
  coverage.json           substituted / already-brand / no-verified-brand lists
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_indian_brands import normalize_ingredient  # noqa: E402 (read-only import)

TEST_RX = ROOT / "test_prescriptions.json"
BRANDS = ROOT / "data" / "indian_brands.json"
MED_DATA = ROOT / "cache" / "medicine_data.json"
OUT_DIR = ROOT / "outputs" / "rxbench"

SEED_DEFAULT = 20260924
HINGLISH_PREFIXES = [
    "bukhar ke liye ",
    "dard ke liye ",
    "doctor ne likha hai: ",
    "bukhar aur dard ke liye ",
]
STRENGTH_RE = re.compile(
    r"\s*\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|units?|%)\b",
    re.IGNORECASE,
)
L1_PATTERNS = [("e", "c"), ("rn", "m"), ("0", "O")]


def prettify_brand(key: str) -> str:
    """'telma-h' -> 'Telma-H', 'dolo' -> 'Dolo' (deterministic surface form)."""
    return "-".join(p[:1].upper() + p[1:] if p else p for p in key.split("-"))


def load_inputs():
    with open(TEST_RX, encoding="utf-8") as f:
        test_rx = json.load(f)
    with open(BRANDS, encoding="utf-8") as f:
        brands = json.load(f)
    with open(MED_DATA, encoding="utf-8") as f:
        known = set(json.load(f)["medicine_names"])
    return test_rx, brands, known


def build_ingredient_map(brands):
    """verified:true AND single-ingredient only -> {ingredient: [(n_rows, key)]}."""
    ingmap = {}
    n_verified = sum(1 for v in brands.values() if v.get("verified"))
    for key, v in brands.items():
        if not v.get("verified"):
            continue
        ings = [str(i).strip().lower() for i in v.get("ingredients", []) if str(i or "").strip()]
        if len(ings) != 1:  # combo brands never cover a single generic slot
            continue
        ingmap.setdefault(ings[0], []).append((v.get("n_rows", 0) or 0, key))
    for ing in ingmap:
        ingmap[ing].sort(key=lambda t: (-t[0], t[1]))
    return ingmap, n_verified


def brand_key_set(brands):
    keys = set()
    for key, v in brands.items():
        keys.add(key.strip().lower())
        for var in v.get("variants", []):
            keys.add(str(var).strip().lower())
    return keys


def classify_drugs(test_rx, brands, known, ingmap, bkeys):
    """Return per-unique-drug plan: substituted / already_brand / no_brand."""
    from collections import Counter

    uniq = Counter()
    for r in test_rx:
        for g in r.get("drugs", []):
            uniq[g.strip().lower()] += 1
    plan = {}
    for g in uniq:
        canon = normalize_ingredient(g, known) or g
        canon = canon.strip().lower()
        if canon in ingmap:
            n_rows, bkey = ingmap[canon][0]
            plan[g] = {
                "status": "substituted",
                "canonical": canon,
                "brand_key": bkey,
                "brand_text": prettify_brand(bkey),
                "n_rows": n_rows,
                "count": uniq[g],
            }
        elif g in bkeys:
            plan[g] = {"status": "already_brand", "canonical": canon,
                       "brand_key": None, "brand_text": None,
                       "n_rows": None, "count": uniq[g]}
        else:
            plan[g] = {"status": "no_verified_brand", "canonical": canon,
                       "brand_key": None, "brand_text": None,
                       "n_rows": None, "count": uniq[g]}
    return plan


def substitute_text(text, plan):
    """Replace generic mentions with brand surfaces (longest generic first)."""
    subs = [(g, p["brand_text"]) for g, p in plan.items()
            if p["status"] == "substituted"]
    subs.sort(key=lambda t: -len(t[0]))
    n_applied = 0
    missing = []
    for generic, brand in subs:
        # Trailing (?![A-Za-z]) instead of \b so glued doses still match:
        # 'Simvastatin20mg' -> 'Simvotin20mg' (dose kept). The lookahead
        # still blocks prefix collisions ('warf' inside 'warfarin').
        pat = re.compile(r"\b" + re.escape(generic) + r"(?![A-Za-z])",
                         re.IGNORECASE)
        text, n = pat.subn(brand, text)
        n_applied += n
        if n == 0:
            missing.append(generic)
    return text, n_applied, missing


def build_india_split(test_rx, plan):
    india = []
    text_missing = []
    for r in test_rx:
        new_text, n_applied, missing = substitute_text(
            r.get("prescription_text", ""), plan)
        new_drugs = []
        linking_rows = []
        for g in r.get("drugs", []):
            p = plan[g.strip().lower()]
            surf = p["brand_text"] if p["status"] == "substituted" else g
            new_drugs.append(surf)
            linking_rows.append({
                "original_generic": g,
                "canonical_ingredient": p["canonical"],
                "brand_text": surf,
                "status": p["status"],
            })
        entry = dict(r)  # GT interactions / has_interaction copied unchanged
        entry["prescription_text"] = new_text
        entry["drugs"] = new_drugs
        entry["original_prescription_text"] = r.get("prescription_text", "")
        entry["original_drugs"] = list(r.get("drugs", []))
        entry["linking"] = {
            "rx_id": r.get("prescription_id"),
            "brand_text": new_drugs,
            "ingredients": [x["canonical_ingredient"] for x in linking_rows],
            "original_generics": list(r.get("drugs", [])),
            "rows": linking_rows,
        }
        india.append(entry)
        for m in missing:
            if m in [g.lower() for g in r.get("drugs", [])]:
                text_missing.append({"rx_id": r.get("prescription_id"),
                                     "generic": m})
    return india, text_missing


# ── noise ────────────────────────────────────────────────────────────────

def noise_l1(text, rng):
    sites = []
    for i, (src, dst) in enumerate(L1_PATTERNS):
        for m in re.finditer(re.escape(src), text):
            sites.append((m.start(), src, dst))
    rng.shuffle(sites)
    edits = []
    out = text
    offset = 0
    for pos, src, dst in sites[:2]:
        p = pos + offset
        out = out[:p] + dst + out[p + len(src):]
        offset += len(dst) - len(src)
        edits.append(f"{src}->{dst}@{pos}")
    return out, edits


def noise_l2(text, drug_surfaces):
    ops = ["case-strip"]
    t = text.lower()
    for surf in sorted(set(drug_surfaces), key=len, reverse=True):
        if not surf:
            continue
        pat = re.compile(re.escape(surf) + r"\s+(?=\d)", re.IGNORECASE)
        if pat.search(t):
            t = pat.sub(surf.lower(), t)
            ops.append(f"glue:{surf}")
    # general drug-dose glue for any remaining "<word> <dose>" gaps
    t2, n = re.subn(r"(?<=[A-Za-z])\s+(?=\d+\s*(?:mg|mcg|g|ml|iu)\b)", "", t)
    if n:
        ops.append(f"glue:dose x{n}")
        t = t2
    ocr = {"daily": "daiIy", "vitamin": "v1tam1n"}
    for src, dst in ocr.items():
        if src in t:
            t = t.replace(src, dst)
            ops.append(f"ocr:{src}->{dst}")
    return t, ops


def noise_l3(text, drug_surfaces, rng):
    ops = []
    t = rng.choice(HINGLISH_PREFIXES) + text
    ops.append("hinglish-prefix")
    t, n = STRENGTH_RE.subn("", t)
    if n:
        ops.append(f"dose-drop x{n}")
    # truncation: on drug-bearing lines keep through last drug + 3 words;
    # drug-free lines are cut to their first 3 words. Drug surfaces kept.
    surfs = sorted({s for s in drug_surfaces if s}, key=len, reverse=True)
    lines = t.split("\n")
    cut = []
    for line in lines:
        low = line.lower()
        hits = [m for s in surfs for m in
                [low.find(s.lower())] if m != -1]
        if hits:
            end = max(m + len(s) for s in surfs
                      for m in [low.find(s.lower())] if m != -1)
            tail = line[end:].split()
            if len(tail) > 3:
                line = line[:end] + " ".join(tail[:3]) + " ..."
                ops.append("truncate:tail")
        else:
            words = line.split()
            if len(words) > 3:
                line = " ".join(words[:3]) + " ..."
                ops.append("truncate:line")
    t = "\n".join(cut or lines)
    # GT-PRESERVATION post-check (verbatim, case-insensitive)
    low = t.lower()
    lost = [s for s in surfs if s.lower() not in low]
    fallback = False
    if lost:
        t = rng.choice(HINGLISH_PREFIXES) + STRENGTH_RE.sub("", text)
        ops = ops + [f"PRESERVATION-FALLBACK lost={lost}"]
        fallback = True
    return t, ops, fallback


def build_noise_splits(india, seed):
    l1_rows, l2_rows, l3_rows = [], [], []
    l1_edits_total = 0
    fallbacks = []
    for idx, r in enumerate(india):
        rng = random.Random(seed * 1000 + idx)
        base = r["prescription_text"]
        surfs = r["drugs"]
        t1, e1 = noise_l1(base, rng)
        l1_edits_total += len(e1)
        e1r = dict(r)
        e1r["prescription_text"] = t1
        e1r["noise_level"] = "L1"
        e1r["noise_edits"] = e1
        l1_rows.append(e1r)
        t2, ops2 = noise_l2(base, surfs)
        e2r = dict(r)
        e2r["prescription_text"] = t2
        e2r["noise_level"] = "L2"
        e2r["noise_edits"] = ops2
        l2_rows.append(e2r)
        t3, ops3, fb = noise_l3(base, surfs, rng)
        if fb:
            fallbacks.append(r.get("prescription_id"))
        e3r = dict(r)
        e3r["prescription_text"] = t3
        e3r["noise_level"] = "L3"
        e3r["noise_edits"] = ops3
        l3_rows.append(e3r)
    return l1_rows, l2_rows, l3_rows, l1_edits_total, fallbacks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()
    seed = args.seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_rx, brands, known = load_inputs()
    ingmap, n_verified = build_ingredient_map(brands)
    bkeys = brand_key_set(brands)
    print(f"brands: total={len(brands)} verified={n_verified} "
          f"single-ingredient verified ingredients={len(ingmap)}")
    print(f"test Rx: {len(test_rx)}")

    plan = classify_drugs(test_rx, brands, known, ingmap, bkeys)
    n_sub = sum(1 for p in plan.values() if p["status"] == "substituted")
    n_already = sum(1 for p in plan.values() if p["status"] == "already_brand")
    n_nosub = sum(1 for p in plan.values() if p["status"] == "no_verified_brand")
    rx_sub = sum(p["count"] for p in plan.values()
                 if p["status"] == "substituted")
    rx_tot = sum(p["count"] for p in plan.values())
    print(f"unique generics: substituted={n_sub} already-brand={n_already} "
          f"no-verified-brand={n_nosub} (slots {rx_sub}/{rx_tot} substituted)")

    india, text_missing = build_india_split(test_rx, plan)
    rx_changed = sum(1 for a, b in zip(india, test_rx)
                     if a["prescription_text"] != b["prescription_text"])
    print(f"Rx texts changed by substitution: {rx_changed}/{len(india)}")
    if text_missing:
        print(f"[WARN] {len(text_missing)} drug slots not found verbatim "
              f"in their Rx text (abbrev/mismatch, left as-is).")

    l1_rows, l2_rows, l3_rows, l1_total, fallbacks = build_noise_splits(
        india, seed)
    print(f"L1 edits applied: {l1_total}; L3 preservation fallbacks: "
          f"{len(fallbacks)} {fallbacks[:10]}")

    # spot-check CSV (seeded 50)
    rng = random.Random(seed + 999)
    sample = sorted(rng.sample(range(len(test_rx)), min(50, len(test_rx))))
    with open(OUT_DIR / "spotcheck_50.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["rx_id", "clean", "brand", "L1", "L2", "L3"])
        for i in sample:
            w.writerow([test_rx[i].get("prescription_id"),
                        test_rx[i]["prescription_text"],
                        india[i]["prescription_text"],
                        l1_rows[i]["prescription_text"],
                        l2_rows[i]["prescription_text"],
                        l3_rows[i]["prescription_text"]])

    # write bench files (evaluator-ready lists stay bare lists for --dataset)
    with open(OUT_DIR / "rxbench-india.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in india], f,
                  ensure_ascii=False, indent=1)
    noise_file = {
        "meta": {
            "seed": seed,
            "n_rx": len(india),
            "base": "brand-substituted text from build_rxbench.py",
            "gt_preservation_rule": (
                "Interacting drug-pair tokens are never dropped/truncated; "
                "L3 dose-drop/truncation touch only strengths and "
                "surrounding instruction text (verified per-Rx)."),
            "l3_preservation_fallbacks": fallbacks,
            "l1_single_char_edits_total": l1_total,
        },
        "splits": {"L1": l1_rows, "L2": l2_rows, "L3": l3_rows},
    }
    with open(OUT_DIR / "rxbench-noise.json", "w", encoding="utf-8") as f:
        json.dump(noise_file, f, ensure_ascii=False, indent=1)

    def strip(entry):
        return {k: v for k, v in entry.items()
                if k not in ("noise_edits",)}

    splits = {
        "split_clean.json": test_rx,
        "split_brand.json": india,
        "split_L1.json": [strip(r) for r in l1_rows],
        "split_L2.json": [strip(r) for r in l2_rows],
        "split_L3.json": [strip(r) for r in l3_rows],
    }
    for name, rows in splits.items():
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    coverage = {
        "substituted": sorted([g for g, p in plan.items()
                               if p["status"] == "substituted"]),
        "already_brand": sorted([g for g, p in plan.items()
                                 if p["status"] == "already_brand"]),
        "no_verified_brand": sorted([g for g, p in plan.items()
                                     if p["status"] == "no_verified_brand"]),
        "counts_unique": {"substituted": n_sub, "already_brand": n_already,
                          "no_verified_brand": n_nosub},
        "counts_slots": {"substituted": rx_sub, "total": rx_tot},
        "text_slots_missing_verbatim": text_missing,
        "top_brand_per_substituted": {
            g: {"brand_key": p["brand_key"],
                "brand_text": p["brand_text"], "n_rows": p["n_rows"]}
            for g, p in sorted(plan.items()) if p["status"] == "substituted"},
    }
    with open(OUT_DIR / "coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=1)
    meta = {
        "seed": seed,
        "n_rx": len(test_rx),
        "brands_total": len(brands),
        "brands_verified": n_verified,
        "rx_texts_changed": rx_changed,
        "linking": [
            {"rx_id": r["linking"]["rx_id"],
             "brand_text": r["linking"]["brand_text"],
             "ingredients": r["linking"]["ingredients"],
             "original_generics": r["linking"]["original_generics"]}
            for r in india
        ],
    }
    with open(OUT_DIR / "rxbench-india.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # validate JSON reload
    for name in ["rxbench-india.json", "rxbench-noise.json",
                 "split_clean.json", "split_brand.json", "split_L1.json",
                 "split_L2.json", "split_L3.json"]:
        with open(OUT_DIR / name, encoding="utf-8") as f:
            json.load(f)
    print(f"Wrote {OUT_DIR} (all bench JSON re-validated). "
          f"test_prescriptions.json untouched.")


if __name__ == "__main__":
    main()
