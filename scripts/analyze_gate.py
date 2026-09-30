import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, json

z = np.load('outputs/learned_gate/_frozen_cache.npz')
P, X, y, argmax = z['P'], z['X'], z['y'], z['argmax']
lr = np.load('outputs/learned_gate/oof_probs_lr.npy')
from clinical_copilot.ddi import InteractionDetector
is_mild = InteractionDetector._is_genuinely_mild
rows = json.load(open('cache/severity_val_hq.json'))
texts = [r['text'] for r in rows]

gated = [i for i in range(400) if not (argmax[i] == 0 and is_mild(texts[i]))]
sev = [i for i in gated if y[i] == 2]
non = [i for i in gated if y[i] != 2]
print(f'gated rows: {len(gated)} (excluded {400 - len(gated)} Mild-allowlist)')
print(f'min Severe OOF P: {min(lr[sev]):.4f}')
print(f'max non-Severe OOF P: {max(lr[non]):.4f}')
print('top-8 non-Severe OOF P:')
for i in sorted(non, key=lambda i: -lr[i])[:8]:
    print(f'  {i:3d} y={y[i]} P={lr[i]:.4f} | {texts[i][:100]}')
print('bottom-5 Severe OOF P:')
for i in sorted(sev, key=lambda i: lr[i])[:5]:
    print(f'  {i:3d} P={lr[i]:.4f} | {texts[i][:100]}')
meta = json.load(open('outputs/learned_gate/gate_meta.json'))
print('weights:')
for w in meta['weights']:
    print(f"  {w['feature']:22s} {w['coef']:+.4f}")
