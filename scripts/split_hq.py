import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import random

def split_hq_data():
    with open('cache/severity_val_2k.json', 'r') as f:
        data = json.load(f)
        
    # We'll use the 'llm_label' as the ground truth
    # We need to make sure 'label' key exists for the training script
    for item in data:
        item['label'] = item['llm_label']
        
    random.seed(42)
    random.shuffle(data)
    
    train_hq = data[:1600]
    val_hq = data[1600:]
    
    with open('cache/severity_train_hq.json', 'w') as f:
        json.dump(train_hq, f, indent=2)
        
    with open('cache/severity_val_hq.json', 'w') as f:
        json.dump(val_hq, f, indent=2)
        
    print(f"Created HQ Train: {len(train_hq)} examples")
    print(f"Created HQ Val: {len(val_hq)} examples")

if __name__ == "__main__":
    split_hq_data()
