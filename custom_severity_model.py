import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import random
from collections import Counter

class SeverityVocab:
    def __init__(self, max_vocab_size=10000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        self.max_vocab_size = max_vocab_size

    def build_vocab(self, texts):
        word_counts = Counter()
        char_counts = Counter()

        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
            for char in text:
                char_counts.update(char)

        # Most common words
        for word, _ in word_counts.most_common(self.max_vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        # All characters
        for char, _ in char_counts.most_common():
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode_words(self, text):
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in text.lower().split()]

    def encode_chars(self, text):
        return [[self.char2idx.get(c, self.char2idx["<UNK>"]) for c in w] for w in text.lower().split()]

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'char2idx': self.char2idx}, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.char2idx = data['char2idx']
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
            self.idx2char = {int(v): k for k, v in self.char2idx.items()}


class SeverityDataset(Dataset):
    def __init__(self, data_path, vocab, max_samples=None, seed=7):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        if max_samples and len(self.data) > max_samples:
            # Subsample for faster training (local RNG: no global mutation)
            self.data = random.Random(seed).sample(self.data, max_samples)

        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        word_ids = self.vocab.encode_words(text)
        char_ids = self.vocab.encode_chars(text)

        # Use llm_label if available (for validation set), otherwise label (for training set)
        label = item.get('llm_label', item.get('label', 0))
        soft_label = item.get('soft_label', [0.0]*3) # fallback to zero if missing
        if len(soft_label) == 4:
            # Merge No Interaction (0) and Mild (1) into a single low-risk class
            # to match the 3-class scheme used by the ensemble: [Low-risk, Moderate, Severe].
            # Sum is the correct marginalization: P(A∪B) = P(A) + P(B) for disjoint classes.
            soft_label = [soft_label[0]+soft_label[1], soft_label[2], soft_label[3]]

        return {
            'word_ids': word_ids,
            'char_ids': char_ids,
            'hard_label': label,
            'soft_label': soft_label
        }

def collate_fn(batch):
    # Sort batch by length for packing
    batch.sort(key=lambda x: len(x['word_ids']), reverse=True)

    word_ids = [torch.tensor(x['word_ids']) for x in batch]
    hard_labels = torch.tensor([x['hard_label'] for x in batch], dtype=torch.long)
    soft_labels = torch.tensor([x['soft_label'] for x in batch], dtype=torch.float)

    lengths = torch.tensor([len(w) for w in word_ids])

    # Pad word ids
    padded_word_ids = nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=0)

    # Pad char ids (requires 3D tensor: batch x words x chars)
    max_word_len = max(max(len(w) for w in x['char_ids']) if x['char_ids'] else 1 for x in batch)
    padded_char_ids = torch.zeros((len(batch), padded_word_ids.size(1), max_word_len), dtype=torch.long)

    for i, item in enumerate(batch):
        for j, char_seq in enumerate(item['char_ids']):
            if j < padded_word_ids.size(1):
                padded_char_ids[i, j, :len(char_seq)] = torch.tensor(char_seq)

    return padded_word_ids, padded_char_ids, lengths, hard_labels, soft_labels


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim=64, filters=64, kernels=[2, 3, 4, 5]):
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, filters, k) for k in kernels
        ])
        self.output_dim = filters * len(kernels)

    def forward(self, char_ids):
        # char_ids: (batch, seq_len, max_word_len)
        batch_size, seq_len, max_word_len = char_ids.size()

        # Reshape to treat words as independent sequences
        # (batch * seq_len, max_word_len)
        x = char_ids.view(-1, max_word_len)

        # (batch * seq_len, max_word_len, char_embed_dim) -> (batch * seq_len, char_embed_dim, max_word_len)
        x = self.embed(x).transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            if x.size(2) < conv.kernel_size[0]:
                # Pad if word is shorter than kernel size
                pad = torch.zeros(x.size(0), x.size(1), conv.kernel_size[0] - x.size(2)).to(x.device)
                x_padded = torch.cat([x, pad], dim=2)
                out = F.relu(conv(x_padded))
            else:
                out = F.relu(conv(x))

            # Max pool over characters
            pooled = F.max_pool1d(out, out.size(2)).squeeze(2)
            conv_outs.append(pooled)

        # (batch * seq_len, num_filters * num_kernels)
        x = torch.cat(conv_outs, dim=1)

        # Back to (batch, seq_len, char_cnn_dim)
        return x.view(batch_size, seq_len, -1)


class MedSeverityNet(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, word_embed_dim=128,
                 hidden_dim=256, num_classes=3, dropout=0.3,
                 char_embed_dim=64, char_filters=64, char_kernels=(2, 3, 4, 5),
                 classifier_hidden=128, num_layers=2):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_embed_dim=char_embed_dim,
                                filters=char_filters, kernels=list(char_kernels))

        lstm_input_dim = word_embed_dim + self.char_cnn.output_dim

        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if (dropout > 0 and num_layers > 1) else 0
        )

        # Self-Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, word_ids, char_ids, lengths, return_attention=False):
        # Embeddings
        word_embedded = self.word_embed(word_ids)  # (B, S, E_w)
        char_embedded = self.char_cnn(char_ids)    # (B, S, E_c)

        x = torch.cat([word_embedded, char_embedded], dim=-1)  # (B, S, E_w + E_c)

        # Pack sequence for LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # (B, S, H*2)

        # Attention Pooling
        attn_weights = self.attention(out).squeeze(-1) # (B, S)

        # Mask padded elements
        mask = torch.arange(out.size(1)).unsqueeze(0).to(word_ids.device) < lengths.unsqueeze(1).to(word_ids.device)
        attn_weights[~mask] = -1e9

        attn_weights = F.softmax(attn_weights, dim=-1) # (B, S)

        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1) # (B, H*2)

        # Classification
        logits = self.classifier(context) # (B, C)

        if return_attention:
            return logits, attn_weights
        return logits

class FastClinicalCNN(nn.Module):
    """
    Alternative 2: TextCNN for blazing fast inference and n-gram detection.
    """
    def __init__(self, vocab_size, char_vocab_size, word_embed_dim=128,
                 num_classes=3, dropout=0.3, filters=128, kernels=(2, 3, 4, 5),
                 char_embed_dim=64, char_filters=64, char_kernels=(2, 3, 4, 5),
                 classifier_hidden=128):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_embed_dim=char_embed_dim,
                                filters=char_filters, kernels=list(char_kernels))
        kernels = list(kernels)

        cnn_input_dim = word_embed_dim + self.char_cnn.output_dim

        # 1D Convolutions for text (acting on words)
        self.convs = nn.ModuleList([
            nn.Conv1d(cnn_input_dim, filters, k) for k in kernels
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters * len(kernels), classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, word_ids, char_ids, lengths, return_attention=False):
        # Embeddings
        word_embedded = self.word_embed(word_ids)  # (B, S, E_w)
        char_embedded = self.char_cnn(char_ids)    # (B, S, E_c)

        x = torch.cat([word_embedded, char_embedded], dim=-1)  # (B, S, E_w + E_c)
        x = x.transpose(1, 2)  # (B, E_w + E_c, S) for Conv1d

        conv_outs = []
        for conv in self.convs:
            if x.size(2) < conv.kernel_size[0]:
                pad = torch.zeros(x.size(0), x.size(1), conv.kernel_size[0] - x.size(2)).to(x.device)
                x_padded = torch.cat([x, pad], dim=2)
                out = F.relu(conv(x_padded))
            else:
                out = F.relu(conv(x))

            # Global max pooling over the sequence
            pooled = F.max_pool1d(out, out.size(2)).squeeze(2)  # (B, filters)
            conv_outs.append(pooled)

        x = torch.cat(conv_outs, dim=1)  # (B, filters * len(kernels))
        logits = self.classifier(x)      # (B, C)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return x

class TinyClinicalFormer(nn.Module):
    """
    Alternative 1: Lightweight Transformer for deep context without RNN overhead.
    """
    def __init__(self, vocab_size, char_vocab_size, word_embed_dim=128,
                 num_classes=3, dropout=0.3, num_layers=2, nhead=4, dim_feedforward=256,
                 char_embed_dim=64, char_filters=64, char_kernels=(2, 3, 4, 5),
                 classifier_hidden=128):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_embed_dim=char_embed_dim,
                                filters=char_filters, kernels=list(char_kernels))

        d_model = word_embed_dim + self.char_cnn.output_dim

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, word_ids, char_ids, lengths, return_attention=False):
        # Embeddings
        word_embedded = self.word_embed(word_ids)  # (B, S, E_w)
        char_embedded = self.char_cnn(char_ids)    # (B, S, E_c)

        x = torch.cat([word_embedded, char_embedded], dim=-1)  # (B, S, E_w + E_c)

        # Transformer expects (S, B, E) if batch_first=False, but we used batch_first=True
        # Positional encoding expects (S, B, E), so transpose, encode, transpose back
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # back to (B, S, E)

        # Create mask for padding tokens
        # key_padding_mask is True for padded positions
        mask = torch.arange(x.size(1)).unsqueeze(0).to(word_ids.device) >= lengths.unsqueeze(1).to(word_ids.device)

        # (B, S, E)
        transformer_out = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over valid positions
        # Mask out padding tokens for pooling
        mask_float = (~mask).float().unsqueeze(-1)
        sum_embeddings = torch.sum(transformer_out * mask_float, dim=1)
        mean_embeddings = sum_embeddings / torch.clamp(mask_float.sum(dim=1), min=1.0)

        logits = self.classifier(mean_embeddings)
        return logits


# Production ensemble: CNN + Transformer winners from the extended search
# (post physician-relabel, 96 configs/model, 5-fold refine, top-5 shootout).
# MedSeverityNet retired. All production code must construct ensemble
# members through build_ensemble_model so layer shapes always match
# cache/ensemble_*.pt.
ENSEMBLE_NAMES = ("FastClinicalCNN", "TinyClinicalFormer")
ENSEMBLE_ARCHS = {
    'FastClinicalCNN': {'word_embed_dim': 512, 'char_embed_dim': 64, 'char_filters': 64,
                        'char_kernels': [2, 3, 4, 5], 'classifier_hidden': 512,
                        'dropout': 0.4, 'filters': 512, 'kernels': [2, 3, 4, 5]},
    'TinyClinicalFormer': {'word_embed_dim': 64, 'char_embed_dim': 32, 'char_filters': 256,
                           'char_kernels': [2, 3, 4], 'classifier_hidden': 512,
                           'dropout': 0.3, 'num_layers': 2, 'nhead': 2, 'dim_ff': 512},
}


def build_model_from_config(model_name, vocab_size, char_vocab_size, cfg, num_classes=3):
    """Build any ensemble member from a config dict holding the architecture keys."""
    common = dict(
        word_embed_dim=cfg["word_embed_dim"], dropout=cfg["dropout"],
        char_embed_dim=cfg["char_embed_dim"], char_filters=cfg["char_filters"],
        char_kernels=cfg["char_kernels"], classifier_hidden=cfg["classifier_hidden"],
    )
    if model_name == "MedSeverityNet":
        return MedSeverityNet(vocab_size, char_vocab_size, hidden_dim=cfg["hidden_dim"],
                              num_layers=cfg["num_layers"], num_classes=num_classes, **common)
    if model_name == "FastClinicalCNN":
        return FastClinicalCNN(vocab_size, char_vocab_size, filters=cfg["filters"],
                               kernels=cfg["kernels"], num_classes=num_classes, **common)
    return TinyClinicalFormer(vocab_size, char_vocab_size, num_layers=cfg["num_layers"],
                              nhead=cfg["nhead"], dim_feedforward=cfg["dim_ff"],
                              num_classes=num_classes, **common)


def build_ensemble_model(name, vocab_size, char_vocab_size, num_classes=3):
    """Build a production ensemble member with its locked-in architecture."""
    if name not in ENSEMBLE_ARCHS:
        raise ValueError(f"Unknown ensemble member {name!r}; retired or misspelled?"
                         f" Active members: {ENSEMBLE_NAMES}")
    return build_model_from_config(name, vocab_size, char_vocab_size,
                                   ENSEMBLE_ARCHS[name], num_classes=num_classes)
