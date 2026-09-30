import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import glob
from collections import Counter

def merge():
    val_path = 'cache/severity_val_2k.json'
    with open(val_path, 'r') as f:
        val_data = json.load(f)

    label_map = {}
    for path in sorted(glob.glob('chunks/chunk_*_labeled.json')):
        with open(path, 'r') as f:
            labels = json.load(f)
        for item in labels:
            label_map[item['id']] = item['severity']

    severity_to_int = {
        "Mild": 0,
        "Moderate": 1,
        "Severe": 2,
        "VERY SEVERE": 2,
    }

    updated = 0
    dist = Counter()
    for i, item in enumerate(val_data):
        if i in label_map:
            sev = label_map[i]
            item['llm_severity'] = sev
            item['llm_label'] = severity_to_int.get(sev, 0)
            dist[sev] += 1
            updated += 1

    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"Updated {updated}/{len(val_data)} items with LLM severity labels")
    print("\nLabel distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({100*count/updated:.1f}%)")

if __name__ == "__main__":
    merge()
