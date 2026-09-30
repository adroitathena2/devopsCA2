"""
Build data/indian_brands.json: Indian brand -> ingredient(s) mapping.

Sources (download first, see UPDATE_3.md):
  data/raw/cdci/*.txt                  Common Drug Codes for India flat files
                                       (NRCeS / C-DAC Pune, CC BY 4.0 - attribute
                                       C-DAC/NRCeS. Obtained via the ohcnetwork/cdci
                                       mirror of the official Flat Files Package.)
  data/raw/indian_medicine/*.csv       junioralive/Indian-Medicine-Dataset (254k rows).
                                       Community-scraped, GREY license: internal
                                       resolution only, never redistribute the raw CSV.

Pipeline:
  1. CDCI join  BrandMaster -> ProductMaster (short brand)
                        -> GenericMaster -> SubstanceMaster (ingredients, '+'-joined)
  2. junioralive gap-fill: brand token from `name`, ingredients from
     short_composition1/2, frequency-ranked (market presence).
  3. Ingredient normalization to DrugBank canonical names (synonym table +
     salt stripping, self-verified against cache/medicine_data.json).
  4. Brand token collapsing (release modifiers only; combo suffixes like -H/-AM
     are NEVER collapsed - telma-h is a different drug than telma).
  5. Emission with source/verified/priority flags. Seeded target list of known
     Indian market brands guarantees the clinically important ones ship.

Usage:
    python build_indian_brands.py [--max-brands 2000] [--all]

Output is merged into synonym_map by clinical_copilot.brands (used by
build_cache.py and the runtime pipeline).
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CDCI_DIR = ROOT / 'data' / 'raw' / 'cdci'
IMD_DIR = ROOT / 'data' / 'raw' / 'indian_medicine'
OUT_FILE = ROOT / 'data' / 'indian_brands.json'
MEDICINE_DATA = ROOT / 'cache' / 'medicine_data.json'

# ── Release/strength modifiers safe to collapse into the base brand. ──
# The base brand must have the SAME active ingredient(s). NEVER put combo or
# formula-changing suffixes here (h/am/m/d/plus/forte/mr/p/spas/tx):
# telma-h != telma, pan-d != pan, meftal-forte != meftal (forte is a combo).
SAFE_MODIFIERS = {
    'xl', 'xr', 'sr', 'er', 'cr', 'ds', 'odt', 'la', 'pr',
}

# Salt/excipient words safe to strip when the stem is a known DrugBank drug.
SALT_WORDS = {
    'hydrochloride', 'hcl', 'potassium', 'sodium', 'succinate', 'tartrate',
    'dihydrate', 'hydrate', 'mesylate', 'maleate', 'fumarate', 'citrate',
    'sulphate', 'sulfate', 'phosphate', 'besylate', 'arginine', 'trometamol',
    'tromethamine', 'calcium', 'magnesium', 'strontium', 'medoxomil',
    'pamoate', 'embonate', 'valerate', 'butyrate', 'propionate', 'acetate',
    'stearate', 'palmitate', 'ascorbate', 'gluconate', 'syclate', 'toxilate',
    'di', 'bis', 'tert', 'as', 'in',
}

# Explicit ingredient synonyms -> DrugBank canonical (lowercase).
INGREDIENT_SYNONYMS = {
    'paracetamol': 'acetaminophen',
    'acetaminophen': 'acetaminophen',
    'aspirin': 'acetylsalicylic acid',
    'acetylsalicylic acid': 'acetylsalicylic acid',
    'asa': 'acetylsalicylic acid',
    'amoxycillin': 'amoxicillin',
    'salbutamol': 'albuterol',
    'lignocaine': 'lidocaine',
    'lignocaine hydrochloride': 'lidocaine',
    'adrenaline': 'epinephrine',
    'clavulanate potassium': 'clavulanic acid',
    'clavulanate': 'clavulanic acid',
    'clavulanic acid': 'clavulanic acid',
    'potassium clavulanate': 'clavulanic acid',
    'acetylsalicylic acid': 'aspirin',
    'hydrochlorothiazide': 'hydrochlorothiazide',
    'hctz': 'hydrochlorothiazide',
    'vitamin d3': 'cholecalciferol',
    'vitamin d': 'cholecalciferol',
    'cholecalciferol': 'cholecalciferol',
    'ergocalciferol': 'ergocalciferol',
    'vitamin b1': 'thiamine',
    'vitamin b2': 'riboflavin',
    'vitamin b6': 'pyridoxine',
    'vitamin b12': 'cyanocobalamin',
    'vitamin c': 'ascorbic acid',
    'vitamin e': 'tocopherol',
    'folic acid': 'folic acid',
    'folate': 'folic acid',
    'niacinamide': 'niacinamide',
    'nicotinamide': 'niacinamide',
    'zinc sulphate': 'zinc sulfate',
    'zinc gluconate': 'zinc gluconate',
    'povidone iodine': 'povidone-iodine',
    'povidone-iodine': 'povidone-iodine',
    'iodine': 'iodine',
    'calcitonin': 'salmon calcitonin',
    'calcitonin salmon': 'salmon calcitonin',
    'salmon calcitonin': 'salmon calcitonin',
    'ferrous fumarate': 'ferrous fumarate',
    'ferrous sulphate': 'ferrous sulfate',
    'iron sucrose': 'iron sucrose',
    'sodium chloride': 'sodium chloride',
    'dextrose': 'dextrose',
    'mannitol': 'mannitol',
    'sucralfate': 'sucralfate',
    'activated charcoal': 'charcoal',
    'chlorpheniramine maleate': 'chlorpheniramine',
    'phenylephrine': 'phenylephrine',
    'ambroxol': 'ambroxol',
    'terbutaline': 'terbutaline',
    'bromhexine': 'bromhexine',
    'guaifenesin': 'guaifenesin',
    'dextromethorphan': 'dextromethorphan',
    'pseudoephedrine': 'pseudoephedrine',
    'pheniramine maleate': 'pheniramine',
    'diphenhydramine': 'diphenhydramine',
    'promethazine': 'promethazine',
    'oxymetazoline': 'oxymetazoline',
    'xylometazoline': 'xylometazoline',
    'sodium cromoglycate': 'cromolyn sodium',
    'sodium cromoglicate': 'cromolyn sodium',
    'sodium feredetate': 'sodium feredetate',
    'haemodial': 'haemodial',
}

# Dosage-form words stripped when extracting brand tokens from product names.
# Forms ONLY: variant words (forte/duo/advance/adult/junior) mark different
# formulas and must survive into the brand key (meftal-forte != meftal).
FORM_WORDS = {
    'tablet', 'tablets', 'tab', 'tabs', 'capsule', 'capsules', 'cap', 'caps',
    'syrup', 'suspension', 'susp', 'injection', 'inj', 'drops', 'drop',
    'cream', 'ointment', 'gel', 'solution', 'soln', 'lotion', 'powder',
    'sachet', 'patch', 'spray', 'inhaler', 'rotacap', 'rotacaps', 'respule',
    'respules', 'suppository', 'infusion', 'oral', 'gastro', 'resistant',
    'release', 'extended', 'sustained', 'prolonged', 'modified', 'dispersible',
    'chewable', 'effervescent', 'granules', 'pellets', 'sachets', 'liquid',
    'syr', 'inhalation', 'topical', 'dermal', 'nasal', 'ophthalmic', 'ot',
    'dt', 'softgel', 'caplet', 'lozenge', 'enema', 'implant', 'vial', 'ampoule',
}

STRENGTH_RE = re.compile(
    r'\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*\s*(?:mg|mcg|µg|ug|g|ml|l|iu|k|'
    r'units?|%|w/v|v/v)?\b', re.IGNORECASE)
PAREN_RE = re.compile(r'\s*\([^)]*\)\s*')
MULTISPACE_RE = re.compile(r'\s+')
ING_SPLIT_RE = re.compile(r'\s+and\s+|\s*\+\s*|/', re.IGNORECASE)

# Clinically important brands: always emitted, cross-checked in QA report.
# Ingredients are the expected canonical names (QA only - data wins on conflict).
SEED_TARGETS = {
    # brief failure cases
    'dolo': ['acetaminophen'], 'crocin': ['acetaminophen'], 'calpol': ['acetaminophen'],
    'paracip': ['acetaminophen'], 'augmentin': ['amoxicillin', 'clavulanic acid'],
    'azithral': ['azithromycin'], 'telma': ['telmisartan'],
    'telma-h': ['telmisartan', 'hydrochlorothiazide'],
    'telma-am': ['telmisartan', 'amlodipine'],
    'ecosprin': ['acetylsalicylic acid'], 'loprin': ['acetylsalicylic acid'],
    'ramistar': ['ramipril'], 'ramistar-am': ['ramipril', 'amlodipine'],
    'ramistar-h': ['ramipril', 'hydrochlorothiazide'],
    'storvas': ['atorvastatin'], 'storvas-ez': ['atorvastatin', 'ezetimibe'],
    'glycomet': ['metformin'], 'glycomet-gp': ['metformin', 'glimepiride'],
    'glycomet-sr': ['metformin'],
    'starpress': ['metoprolol'], 'starpress-xl': ['metoprolol'],
    # analgesic / antipyretic
    'combiflam': ['ibuprofen', 'acetaminophen'],     'brufen': ['ibuprofen'],
    'ibugesic': ['ibuprofen'], 'meftal': ['mefenamic acid'],
    'zerodol': ['aceclofenac'], 'hifenac': ['aceclofenac'],
    'nise': ['nimesulide'], 'nimulid': ['nimesulide'],
    'dolonex': ['piroxicam'], 'zydol': ['tramadol'], 'tramazac': ['tramadol'],
    'ultracet': ['tramadol', 'acetaminophen'], 'diclonac': ['diclofenac'],
    'voveran': ['diclofenac'], 'cataflam': ['diclofenac'],
    # antibiotics
    'mox': ['amoxicillin'], 'amox': ['amoxicillin'], 'novamox': ['amoxicillin'],
    'augmentin-duo': ['amoxicillin', 'clavulanic acid'],
    'azithrocin': ['azithromycin'], 'zithromax': ['azithromycin'],
    'cifran': ['ciprofloxacin'], 'ciplox': ['ciprofloxacin'],
    'taxim': ['cefotaxime'], 'zifi': ['cefixime'], 'cefix': ['cefixime'],
    'mahacef': ['cefixime'], 'cepodem': ['cefpodoxime'],
    'monocef': ['ceftriaxone'], 'monocef-o': ['cefixime', 'ofloxacin'],
    'oflox': ['ofloxacin'], 'zanocin': ['ofloxacin'],
    'levocip': ['levofloxacin'], 'levoflox': ['levofloxacin'],
    'metrogyl': ['metronidazole'], 'flagyl': ['metronidazole'],
    'doxy-1': ['doxycycline'], 'minicycline': ['doxycycline'],
    # cardiac / antihypertensive / antiplatelet
    'telsartan': ['telmisartan'], 'telmikind': ['telmisartan'],
    'losar': ['losartan'], 'losar-h': ['losartan', 'hydrochlorothiazide'],
    'losar-am': ['losartan', 'amlodipine'], 'repace': ['losartan'],
    'repace-h': ['losartan', 'hydrochlorothiazide'],
    'olmy': ['olmesartan'], 'olmy-h': ['olmesartan', 'hydrochlorothiazide'],
    'cresar': ['telmisartan'], 'amlong': ['amlodipine'],
    'amlong-a': ['amlodipine', 'atenolol'], 'amlopres': ['amlodipine'],
    'amlopres-at': ['amlodipine', 'atenolol'], 'amlokind': ['amlodipine'],
    'cardace': ['ramipril'], 'metxl': ['metoprolol'],
    'metxl-a': ['metoprolol', 'amlodipine'], 'betaloc': ['metoprolol'],
    'concor': ['bisoprolol'], 'biso': ['bisoprolol'],
    'clopilet': ['clopidogrel'], 'deplatt': ['clopidogrel'],
    'rozavel': ['rosuvastatin'], 'rozavel-ez': ['rosuvastatin', 'ezetimibe'],
    'rosuvas': ['rosuvastatin'], 'rosuvas-ez': ['rosuvastatin', 'ezetimibe'],
    'atorva': ['atorvastatin'], 'lipvas': ['atorvastatin'],
    'lasix': ['furosemide'], 'frusenex': ['furosemide'],
    'aldactone': ['spironolactone'], 'spiron': ['spironolactone'],
    'nicardia': ['nifedipine'], 'calcigard': ['nifedipine'],
    'isoxy': ['isoxsuprine'], 'monotrate': ['isosorbide mononitrate'],
    'sorbitrate': ['isosorbide dinitrate'],
    # diabetes
    'obimet': ['metformin'], 'forminal': ['metformin'],
    'glizide': ['gliclazide'], 'reclide': ['gliclazide'], 'gliclaz': ['gliclazide'],
    'zoryl': ['glimepiride'], 'glimisave': ['glimepiride'], 'diaglime': ['glimepiride'],
    'janumet': ['metformin', 'sitagliptin'], 'istamet': ['metformin', 'sitagliptin'],
    'galvus': ['vildagliptin'], 'galvus-met': ['metformin', 'vildagliptin'],
    'zomelis': ['vildagliptin'], 'zomelis-met': ['metformin', 'vildagliptin'],
    'forxiga': ['dapagliflozin'], 'oxra': ['dapagliflozin'],
    'jardiance': ['empagliflozin'], 'glyxambi': ['empagliflozin', 'linagliptin'],
    'glaritus': ['insulin glargine'], 'lantus': ['insulin glargine'],
    'humulin': ['insulin human'], 'actrapid': ['insulin human'],
    'voglibose': ['voglibose'], 'volix': ['voglibose'],
    # GI
    'pan': ['pantoprazole'], 'pan-d': ['pantoprazole', 'domperidone'],
    'pantop': ['pantoprazole'], 'pantocid': ['pantoprazole'],
    'pantodac': ['pantoprazole'], 'omez': ['omeprazole'], 'ocid': ['omeprazole'],
    'razo': ['rabeprazole'], 'rabemac': ['rabeprazole'], 'rabonik': ['rabeprazole'],
    'nexpro': ['esomeprazole'], 'sompraz': ['esomeprazole'], 'esogress': ['esomeprazole'],
    'emesis': ['ondansetron'], 'emeset': ['ondansetron'], 'ondem': ['ondansetron'],
    'domstal': ['domperidone'], 'motilium': ['domperidone'],
    'vizylac': ['lactobacillus'], 'econorm': ['saccharomyces boulardii'],
    'ucdl': ['bismuth'],     'digene': ['carboxymethylcellulose', 'magnesium hydroxide', 'simethicone'],
    # respiratory
    'montair': ['montelukast'], 'montek': ['montelukast'], 'telekast': ['montelukast'],
    'montair-lc': ['montelukast', 'levocetirizine'],
    'telekast-l': ['montelukast', 'levocetirizine'],
    'ascoril': ['ambroxol', 'terbutaline', 'guaifenesin'],
    'grilinctus': ['ammonium chloride', 'chlorpheniramine', 'dextromethorphan'],
    'tusq': ['bromhexine', 'chlorpheniramine', 'dextromethorphan'],
    'asmatil': ['salbutamol'],
    'duolin': ['levosalbutamol', 'ipratropium'], 'asthalin': ['albuterol'],
    'levolin': ['levosalbutamol'], 'foracort': ['budesonide', 'formoterol'],
    'budecort': ['budesonide'],     'seroflo': ['fluticasone propionate', 'salmeterol'],
    'theobid': ['theophylline'], 'accuhaler': ['salmeterol'],
    'azulix': ['glimepiride'],
    # allergy / cold
    'cetzine': ['cetirizine'], 'okacet': ['cetirizine'], 'alerid': ['cetirizine'],
    'alcet': ['levocetirizine'], 'levocet': ['levocetirizine'],
    'allegra': ['fexofenadine'], 'fexofen': ['fexofenadine'],
    'nasivion': ['oxymetazoline'], 'otrivin': ['xylometazoline'],
    'zykast': ['montelukast', 'fexofenadine'], 'citrizine': ['cetirizine'],
    'sinarest': ['paracetamol', 'phenylephrine', 'chlorpheniramine'],
    # vitamins / minerals
    'zincovit': ['multivitamin'], 'shelcal': ['calcium carbonate', 'cholecalciferol'],
    'neurobion': ['thiamine', 'pyridoxine', 'riboflavin', 'niacinamide', 'cyanocobalamin'],
    'supradyn': ['multivitamin'], 'becosules': ['vitamin b complex'],
    'polybion': ['vitamin b complex'], 'folvite': ['folic acid'],
    'fefol': ['ferrous fumarate', 'folic acid'],
    'autrin': ['cyanocobalamin', 'ferrous fumarate', 'folic acid'],
    'orofer': ['iron sucrose'], 'gemcal': ['calcium carbonate', 'cholecalciferol'],
    'd-rise': ['cholecalciferol'], 'uprise': ['cholecalciferol'],
    'calcimax': ['calcium carbonate'], 'ossica': ['calcium carbonate'],
    'd3-60k': ['cholecalciferol'], 'calcirol': ['cholecalciferol'],
    # thyroid / steroids / derm / misc
    'thyronorm': ['levothyroxine'], 'eltroxin': ['levothyroxine'],
    'thyronorm': ['levothyroxine'], 'wysolone': ['prednisolone'],
    'omnacortil': ['prednisolone'], 'zempred': ['methylprednisolone'],
    'wysolone': ['prednisolone'], 'decadron': ['dexamethasone'],
    'betnesol': ['betamethasone'], 'candid': ['clotrimazole'],
    'lobate': ['clobetasol propionate'], 'dermadex': ['dexamethasone'],
    'onabet': ['sertaconazole'], 'moiz': ['cetyl alcohol'],
    'hhs': ['hydrocortisone'], 'soframycin': ['framycetin'],
    'neosporin': ['bacitracin', 'neomycin', 'polymyxin b'],     'betadine': ['povidone-iodine'],
    'sofradex': ['framycetin', 'dexamethasone'],
    'combiflam': ['ibuprofen', 'acetaminophen'],
    'crocin-advance': ['acetaminophen'],
    'augmentin-1000': ['amoxicillin', 'clavulanic acid'],
    'augmentin-duo': ['amoxicillin', 'clavulanic acid'],
}

# Brand names reused across companies/products for DIFFERENT drugs. Data is
# right for each product; a single mapping is a documented market ambiguity.
AMBIGUOUS_BRANDS = {'alcet', 'gemcal'}


# ── helpers ────────────────────────────────────────────────────────────

def _read_tsv(path):
    """Read a CDCI master file. Handles the trailing-tab quirk (rows carry an
    extra empty field) and the 2 ragged BrandMaster rows."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        n = len(header)
        for row in reader:
            if len(row) < n:
                continue
            yield dict(zip(header, row[:n]))


def _norm_space(s: str) -> str:
    return MULTISPACE_RE.sub(' ', (s or '').strip().lower())


def normalize_ingredient(raw: str, known: set) -> str | None:
    """
    Map a raw ingredient string to a DrugBank canonical name (lowercase).

    Handles parenthetical word order ('Calcitonin (Salmon)' -> 'salmon
    calcitonin'), hyphen/spacing variants ('povidone iodine' -> 'povidone-iodine'),
    and trailing salt words. Only SALT_WORDS are ever stripped: stripping any
    trailing word wrongly mapped 'povidone iodine' -> 'povidone'.
    """
    s = _norm_space(raw)
    parens = []
    for chunk in re.findall(r'\(([^)]+)\)', s):
        chunk = re.sub(r'[^a-z0-9\s\-/]', ' ', chunk)
        chunk = _norm_space(chunk).strip(' -/')
        # drop pure strengths ("10% w/w", "500mg")
        if chunk and not re.fullmatch(r'[\d\s\.%/a-z]{0,6}', chunk):
            parens.append(chunk)
        elif chunk and not re.search(r'\d', chunk):
            parens.append(chunk)
    outer = PAREN_RE.sub(' ', s)
    outer = re.sub(r'[^a-z0-9\s\-/]', ' ', outer)
    outer = _norm_space(outer).strip(' -/')

    candidates = []
    if outer:
        candidates.append(outer)
        for pw in parens:
            candidates.append(f'{outer} {pw}')
            candidates.append(f'{pw} {outer}')
    for cand in list(candidates):
        if ' ' in cand:
            candidates.append(cand.replace(' ', '-'))
        if '-' in cand:
            candidates.append(cand.replace('-', ' '))
    for cand in candidates:
        hit = _lookup_ingredient(cand, known)
        if hit:
            return hit
    return None


def _lookup_ingredient(s: str, known: set) -> str | None:
    if not s:
        return None
    s = s.strip(' -/')
    if not s:
        return None
    if s in INGREDIENT_SYNONYMS:
        c = INGREDIENT_SYNONYMS[s]
        return c if (c in known or not c) else c
    if s in known:
        return s
    words = s.split()
    while len(words) > 1 and words[-1] in SALT_WORDS:
        words.pop()
        cand = ' '.join(words)
        if cand in INGREDIENT_SYNONYMS:
            c = INGREDIENT_SYNONYMS[cand]
            return c if (c in known or not c) else c
        if cand in known:
            return cand
    return None


def collapse_token(token: str) -> str:
    """
    Canonical brand key. Hyphen-joined so 'telma h' == 'telma-h'; release
    modifiers and strength-ish segments dropped ('starpress xl' -> 'starpress',
    'dolo-650' -> 'dolo'). Combo suffixes (h/am/m/d/plus) are NEVER dropped:
    telma-h is a different drug than telma.
    """
    parts = [p for p in re.split(r'[-_\s]+', (token or '').strip().lower()) if p]
    parts = [p for p in parts if not re.fullmatch(r'\d+(?:\.\d+)?[a-z]{0,3}', p)]
    while len(parts) > 1 and parts[-1] in SAFE_MODIFIERS:
        parts.pop()
    return '-'.join(parts)


def surface_forms(key: str) -> set:
    forms = {key}
    if '-' in key:
        forms.add(key.replace('-', ' '))
        forms.add(key.replace('-', ''))
    if ' ' in key:
        forms.add(key.replace(' ', '-'))
        forms.add(key.replace(' ', ''))
    return {f for f in forms if f}


def extract_brand_token(name: str) -> str:
    """'Augmentin 625 Duo Tablet' -> 'augmentin duo'; 'Telma-H 40/12.5' -> 'telma-h'."""
    s = _norm_space(name)
    s = PAREN_RE.sub(' ', s)
    for _ in range(3):
        stripped = STRENGTH_RE.sub(' ', s)
        if stripped == s:
            break
        s = stripped
    words = [w for w in s.split()
             if w not in FORM_WORDS and not re.fullmatch(r'[-/]+', w)]
    s = _norm_space(' '.join(words))
    return s.strip(' -/')


def fsn_brand_token(bname: str) -> str:
    """'Dolo (paracetamol) 650 mg oral tablet' -> 'dolo'."""
    return extract_brand_token(bname.split('(')[0])


def fsn_ingredients(bname: str) -> list:
    """Parse the parenthetical ingredient list of a CDCI FSN brand name."""
    m = re.search(r'\(([^)]+)\)', bname)
    if not m:
        return []
    inner = re.sub(r'\b(as|in)\s+\S+', ' ', m.group(1))
    parts = ING_SPLIT_RE.split(inner)
    return [p.strip() for p in parts if p and p.strip()]


# ── ingestion ──────────────────────────────────────────────────────────

class BrandRec:
    __slots__ = ('tokens', 'ings', 'suppliers', 'sources', 'rows')

    def __init__(self):
        self.tokens = Counter()
        self.ings = defaultdict(Counter)  # source -> Counter[tuple[str, ...]]
        self.suppliers = set()
        self.sources = set()
        self.rows = 0


def load_cdci(cdci_dir: Path, known: set, records: dict) -> int:
    files = ['SubstanceMaster.txt', 'GenericMaster.txt', 'ProductMaster.txt',
             'BrandMaster.txt', 'SupplierMaster.txt']
    for fn in files:
        if not (cdci_dir / fn).exists():
            print(f"[WARN] CDCI file missing: {fn} - skipping CDCI ingest.")
            return 0

    sub = {}
    for r in _read_tsv(cdci_dir / 'SubstanceMaster.txt'):
        sid = (r.get('Identifier') or '').strip()
        sname = (r.get('Substance Name') or '').strip()
        if sid and sname:
            sub[sid] = sname

    gen = {}
    for r in _read_tsv(cdci_dir / 'GenericMaster.txt'):
        gid = (r.get('Identifier') or '').strip()
        sids = [x.strip() for x in (r.get('Substance Identifier') or '').split('+')]
        if gid:
            gen[gid] = sids

    prod = {}
    for r in _read_tsv(cdci_dir / 'ProductMaster.txt'):
        pid = (r.get('Identifier') or '').strip()
        pname = (r.get('Product Name') or '').strip()
        if pid and pname:
            prod[pid] = pname

    supp = {}
    for r in _read_tsv(cdci_dir / 'SupplierMaster.txt'):
        sid = (r.get('Identifier') or '').strip()
        sname = (r.get('Supplier Name') or '').strip()
        if sid and sname:
            supp[sid] = sname

    n = 0
    for r in _read_tsv(cdci_dir / 'BrandMaster.txt'):
        bname = (r.get('Brand Name') or '').strip()
        if not bname:
            continue
        short = prod.get((r.get('Product Identifier') or '').strip())
        token_raw = (short or fsn_brand_token(bname))
        token = extract_brand_token(token_raw) or fsn_brand_token(bname)
        if not token or len(token) < 2:
            continue

        sids = gen.get((r.get('Generic Identifier') or '').strip(), [])
        ing_raw = [sub[s] for s in sids if s in sub]
        if not ing_raw:
            ing_raw = fsn_ingredients(bname)
        ings = tuple(sorted({norm for norm in
                             (normalize_ingredient(x, known) for x in ing_raw)
                             if norm}))
        if not ings:
            continue

        key = collapse_token(token)
        rec = records[key]
        rec.tokens[token] += 1
        rec.ings['cdci'][ings] += 1
        rec.sources.add('cdci')
        rec.rows += 1
        s = supp.get((r.get('Supplier Identifier') or '').strip())
        if s:
            rec.suppliers.add(s)
        n += 1
    return n


def load_indian_medicine(csv_path: Path, known: set, records: dict) -> int:
    n = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name') or ''
            if not name:
                continue
            token = extract_brand_token(name)
            if not token or len(token) < 2:
                continue
            raw = [(row.get('short_composition1') or ''),
                   (row.get('short_composition2') or '')]
            ings = tuple(sorted({norm for norm in
                                 (normalize_ingredient(x, known) for x in raw)
                                 if norm}))
            if not ings:
                continue
            key = collapse_token(token)
            rec = records[key]
            rec.tokens[token] += 1
            rec.ings['junioralive'][ings] += 1
            rec.sources.add('junioralive')
            rec.rows += 1
            m = (row.get('manufacturer_name') or '').strip()
            if m:
                rec.suppliers.add(m)
            n += 1
    return n


# ── emission ───────────────────────────────────────────────────────────

# Seed keys normalized through the same collapse as data keys, so
# 'starpress-xl'/'dolo-650'/'augmentin-duo' merge into their base concepts.
SEED_CANON = {}
for _k, _v in SEED_TARGETS.items():
    _ck = collapse_token(_k)
    SEED_CANON[_ck] = sorted(set(SEED_CANON.get(_ck, [])) | set(_v))


def build(records: dict, known: set, max_brands: int | None) -> tuple:
    lex = {}
    qa = {'mismatch': [], 'incomplete': [], 'kept': 0, 'skipped': 0}

    # Alias-quality guard: short/junk surfaces pollute the embedding alias
    # pool (e.g. 'ole'->olanzapine matching 'omeprazole' queries). Short
    # surfaces must be curated (priority seed list); junk patterns dropped.
    _junk = [re.compile(r'^ns(-|%| )?'), re.compile(r'[%\[\(/:+]'),
             re.compile(r'\d{5,}-\d{2}-\d'), re.compile(r'^[^a-z]+$'),
             re.compile(r'stayhappi')]

    def _ok_surface(s: str, pri: bool) -> bool:
        t = (s or '').strip().lower()
        if not t or len(t) <= 2 or len(t) > 40:
            return False
        if any(p.search(t) for p in _junk):
            return False
        if len(t) <= 4 and not pri:
            return False
        return True

    ranked = sorted(records.items(), key=lambda kv: (-kv[1].rows, kv[0]))
    for key, rec in ranked:
        if not key:
            continue
        if len(key) > 40:
            qa['skipped'] += 1
            continue

        # Vote across sources. CDCI is the authoritative structured join
        # (full '+'-joined substance chains); junioralive short_composition
        # only holds 2 slots and silently truncates 3+ ingredient combos,
        # so it is weighted lower and longer tuples win near-ties.
        raw_votes, scored = Counter(), Counter()
        for src, ctr in rec.ings.items():
            w = 3 if src == 'cdci' else 1
            for ing_tup, n in ctr.items():
                raw_votes[ing_tup] += n
                scored[ing_tup] += n * w
        if not scored:
            qa['skipped'] += 1
            continue
        top = max(scored.values())
        contenders = [t for t, s in scored.items() if s >= 0.5 * top]
        winner = max(contenders, key=lambda t: (len(t), scored[t]))
        ings = list(winner)
        votes = raw_votes[winner]

        verified = rec.rows >= 2 and votes >= max(2, rec.rows // 2)
        is_pri = key in SEED_CANON
        if not _ok_surface(key, is_pri):
            qa['skipped'] += 1
            continue
        entry = {
            'ingredients': ings,
            'variants': sorted({v for v in
                                (surface_forms(key) | set(rec.tokens))
                                if _ok_surface(v, is_pri)})[:10],
            'sources': sorted(rec.sources),
            'priority': is_pri,
            'verified': bool(verified),
            'combo': len(ings) > 1,
            'supplier': sorted(rec.suppliers)[0] if rec.suppliers else '',
            'n_rows': rec.rows,
            'notes': '',
        }
        if key in SEED_CANON and key not in AMBIGUOUS_BRANDS:
            exp = set(SEED_CANON[key])
            got = set(ings)
            if exp != got:
                if got < exp:
                    entry['notes'] = f'INCOMPLETE vs expected {sorted(exp)}'
                    qa['incomplete'].append((key, sorted(got), sorted(exp)))
                else:
                    entry['notes'] = f'MISMATCH vs expected {sorted(exp)}'
                    qa['mismatch'].append((key, sorted(got), sorted(exp)))
        elif key in AMBIGUOUS_BRANDS:
            entry['notes'] = 'AMBIGUOUS brand name (reused across products)'
        lex[key] = entry
        qa['kept'] += 1

    # guarantee every seed target ships (even with zero data rows)
    for key, exp in SEED_CANON.items():
        if key not in lex:
            ings = [x for x in (normalize_ingredient(i, known) for i in exp) if x] or exp
            lex[key] = {
                'ingredients': ings or exp,
                'variants': sorted({v for v in surface_forms(key)
                                    if _ok_surface(v, True)}),
                'sources': ['seed'],
                'priority': True,
                'verified': False,
                'combo': len(exp) > 1,
                'supplier': '',
                'n_rows': 0,
                'notes': 'seed only - no data source hit',
            }
            qa['kept'] += 1

    if max_brands and len(lex) > max_brands:
        pri = [k for k, v in lex.items() if v['priority']]
        rest = sorted((k for k in lex if k not in pri),
                      key=lambda k: (-lex[k]['n_rows'], k))
        keep = set(pri) | set(rest[:max(0, max_brands - len(pri))])
        dropped = set(lex) - keep
        for k in dropped:
            del lex[k]
        qa['skipped'] += len(dropped)

    return lex, qa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-brands', type=int, default=2000,
                    help='cap on emitted brand concepts (seed targets always kept)')
    ap.add_argument('--all', action='store_true', help='emit everything (no cap)')
    args = ap.parse_args()

    if not MEDICINE_DATA.exists():
        sys.exit(f"Need {MEDICINE_DATA} (run build_cache.py first) for name validation.")
    with open(MEDICINE_DATA, encoding='utf-8') as f:
        known = set(json.load(f)['medicine_names'])
    print(f"Loaded {len(known)} DrugBank canonical names for validation.")

    records = defaultdict(BrandRec)
    n1 = load_cdci(CDCI_DIR, known, records) if CDCI_DIR.exists() else 0
    print(f"CDCI ingest:          {n1} brand rows -> {len(records)} concepts")

    imd_csv = next(iter(IMD_DIR.glob('*.csv')), None) if IMD_DIR.exists() else None
    n2 = load_indian_medicine(imd_csv, known, records) if imd_csv else 0
    print(f"Indian-Medicine ingest: {n2} product rows -> {len(records)} concepts")

    if not n1 and not n2:
        sys.exit("No source data found under data/raw/ - see UPDATE_3.md for URLs.")

    lex, qa = build(records, known, None if args.all else args.max_brands)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(lex.items())), f, ensure_ascii=False, indent=1)

    n_combo = sum(1 for v in lex.values() if v['combo'])
    n_ver = sum(1 for v in lex.values() if v['verified'])
    n_pri = sum(1 for v in lex.values() if v['priority'])
    print(f"\nWrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.0f} KB)")
    print(f"  concepts: {len(lex)}  (priority {n_pri}, verified {n_ver}, combo {n_combo})")
    print(f"  skipped:  {qa['skipped']}")

    missing_seed = [k for k in SEED_CANON if k not in lex]
    if missing_seed:
        print(f"  [WARN] seed targets dropped by cap/quality: {missing_seed}")
    if qa['mismatch']:
        print(f"  [QA] ingredient MISMATCH vs expected ({len(qa['mismatch'])}):")
        for key, got, exp in qa['mismatch'][:15]:
            print(f"       {key}: got {got} expected {exp}")
    if qa['incomplete']:
        print(f"  [QA] ingredient INCOMPLETE ({len(qa['incomplete'])}):")
        for key, got, exp in qa['incomplete'][:15]:
            print(f"       {key}: got {got} expected {exp}")
    probe = ['dolo', 'telma', 'telma-h', 'telma-am', 'augmentin', 'ramistar-am',
             'starpress', 'crocin', 'ecosprin', 'glycomet', 'storvas', 'betadine']
    print("\n  probe:")
    for p in probe:
        e = lex.get(p) or lex.get(p.replace('-', ' '))
        if e:
            print(f"    {p:14s} -> {e['ingredients']}  src={e['sources']} rows={e['n_rows']}")
        else:
            print(f"    {p:14s} -> MISSING")


if __name__ == '__main__':
    main()
