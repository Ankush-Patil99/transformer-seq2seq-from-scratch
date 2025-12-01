import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

# --------------------------
# CLEAN TEXT
# --------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", text)  # English + Hindi
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------
# TOKENIZATION
# --------------------------

def build_vocab(sentences, min_freq=2):
    from collections import Counter

    counter = Counter()
    for s in sentences:
        counter.update(s.split())

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def encode(sentence, vocab):
    tokens = sentence.split()
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    return [vocab["<sos>"]] + ids + [vocab["<eos>"]]

def decode_ids(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    tokens = []
    for i in ids:
        if i == inv["<eos>"]:
            break
        if inv.get(i) not in ["<pad>", "<sos>"]:
            tokens.append(inv.get(i, "<unk>"))
    return " ".join(tokens)

# --------------------------
# MASKS
# --------------------------

def create_padding_mask(seq, pad_id=0):
    mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    return mask  # (B,1,1,L)

def create_decoder_mask(seq, pad_id=0):
    B, L = seq.size()
    pad_mask = create_padding_mask(seq, pad_id)
    look_ahead = torch.tril(torch.ones((L, L), device=seq.device)).bool()
    return pad_mask & look_ahead  # (B,1,L,L)

# --------------------------
# POS ENCODING
# --------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
