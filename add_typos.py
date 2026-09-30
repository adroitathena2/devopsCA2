import json

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

data = load_data('test_prescriptions.json')

typo_map = {
    "Capecitabine": "Capcitabine",     # missing e
    "Cefpodoxime": "Cefpodoxim",       # missing e
    "Tridihexethyl": "Tridihexethil",  # y -> i
    "Mofebutazone": "Mofebutasone",    # z -> s
    "Mepindolol": "Mependolol",        # i -> e
    "Icosapent": "Icosopent"           # a -> o
}

count = 0
for rx in data:
    text = rx['prescription_text']
    
    for real, typo in typo_map.items():
        if real in text:
            # Replace in text
            text = text.replace(real, typo)
            rx['prescription_text'] = text
            count += 1
            
save_data(data, 'test_prescriptions.json')
print(f"Added {count} realistic typos back.")
