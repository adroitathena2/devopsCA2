import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import os

def split_dataset():
    os.makedirs('chunks', exist_ok=True)
    with open('cache/severity_val_2k.json', 'r') as f:
        data = json.load(f)

    # 40 files -> 50 items each
    chunk_size = len(data) // 40

    for i in range(40):
        start_idx = i * chunk_size
        # Get the rest if it's the last chunk
        end_idx = (i + 1) * chunk_size if i < 39 else len(data)

        chunk = data[start_idx:end_idx]

        # Only keep 'text' to save context space for the agents
        clean_chunk = [{"id": start_idx + j, "text": item["text"]} for j, item in enumerate(chunk)]

        with open(f'chunks/chunk_{i+1}.json', 'w') as f:
            json.dump(clean_chunk, f, indent=2)

    print(f"Successfully split {len(data)} items into 40 chunks in the 'chunks/' directory.")

if __name__ == "__main__":
    split_dataset()
