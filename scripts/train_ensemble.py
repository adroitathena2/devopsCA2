import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clinical_copilot.custom_severity_model import SeverityVocab, SeverityDataset, build_ensemble_model, collate_fn

def train_ensemble(seed: int = 7):
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (seed={seed})")

    subset_train_path = 'cache/severity_train_hq.json'
    subset_val_path = 'cache/severity_val_hq.json'

    vocab = SeverityVocab()
    vocab.load('cache/severity_vocab.json')

    train_dataset = SeverityDataset(subset_train_path, vocab)
    val_dataset = SeverityDataset(subset_val_path, vocab)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, generator=loader_gen)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # Note: Using num_classes=3

    # Calculate class weights dynamically based on the actual training subset
    from collections import Counter
    label_counts = Counter([item.get('llm_label', item.get('label', 0)) for item in train_dataset.data])
    total_samples = len(train_dataset.data)
    class_counts = [label_counts.get(i, 1) for i in range(3)] # Classes 0, 1, 2
    class_weights = torch.FloatTensor([total_samples / c for c in class_counts]).to(device)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * 3.0

    # Winner architectures (see ENSEMBLE_ARCHS in custom_severity_model.py).
    models_to_train = {
        name: build_ensemble_model(name, len(vocab.word2idx), len(vocab.char2idx),
                                   num_classes=3)
        for name in ('FastClinicalCNN', 'TinyClinicalFormer')
    }

    epochs = 20
    alpha = 0.5 # Distillation weight

    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name} with Label Smoothing...")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for word_ids, char_ids, lengths, hard_labels, soft_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                word_ids, char_ids, hard_labels, soft_labels = word_ids.to(device), char_ids.to(device), hard_labels.to(device), soft_labels.to(device)

                optimizer.zero_grad()
                logits = model(word_ids, char_ids, lengths)

                #  Label Smoothing (0.1)
                hard_loss = F.cross_entropy(logits, hard_labels, weight=class_weights, label_smoothing=0.1)

                log_probs = F.log_softmax(logits, dim=-1)
                soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()

                loss = alpha * hard_loss + (1 - alpha) * soft_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for word_ids, char_ids, lengths, hard_labels, soft_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    word_ids, char_ids, hard_labels, soft_labels = word_ids.to(device), char_ids.to(device), hard_labels.to(device), soft_labels.to(device)

                    logits = model(word_ids, char_ids, lengths)
                    h_loss = F.cross_entropy(logits, hard_labels, weight=class_weights, label_smoothing=0.1)
                    s_loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    val_loss += (alpha * h_loss + (1 - alpha) * s_loss).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'cache/ensemble_{model_name}.pt')

if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser()
    _p.add_argument("--seed", type=int, default=7)
    train_ensemble(seed=_p.parse_args().seed)
