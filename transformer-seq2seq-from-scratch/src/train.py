# src/train.py
"""
Train & evaluate the Transformer model.
Usage:
    - Configure dataset path and hyperparameters in config.py or edit below variables.
    - Run: python src/train.py
"""

import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sacrebleu import corpus_bleu

# local imports (ensure src is on PYTHONPATH or run from project root)
from config import D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, MAX_LEN, LR, EPOCHS, LABEL_SMOOTHING, BEAM_WIDTH, PAD_ID, SOS_ID, EOS_ID
from utils import clean_text, build_vocab, encode, decode_ids, create_padding_mask, create_decoder_mask
from model import Transformer

# -------------------------
# Paths / dataset
# -------------------------
# Edit these according to your repo layout
DATA_CSV = "data/parallel/hindi_english_small.csv"   # expected columns: 'en', 'hi'
VOCAB_DIR = "vocab"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
IM_DIR = os.path.join(RESULTS_DIR, "images")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(VOCAB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Simple label smoothing loss
# -------------------------
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, label_smoothing, trg_vocab_size, ignore_index=0):
        super().__init__()
        self.smoothing = label_smoothing
        self.vocab_size = trg_vocab_size
        self.confidence = 1.0 - label_smoothing
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # pred: (B, L, V); target: (B, L)
        pred = pred.reshape(-1, pred.size(-1))
        target = target.reshape(-1)

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        log_probs = F.log_softmax(pred, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
        return loss

# -------------------------
# Data helpers (lightweight)
# -------------------------
def load_dataset(csv_path, max_rows=10000):
    df = pd.read_csv(csv_path)
    assert 'en' in df.columns and 'hi' in df.columns, "CSV must have 'en' and 'hi' columns"
    df['en'] = df['en'].astype(str).apply(clean_text)
    df['hi'] = df['hi'].astype(str).apply(clean_text)
    df = df[df['en'].str.len() < 150]
    df = df[df['hi'].str.len() < 150]
    df = df.head(max_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def collate_batch(batch, vocab_en, vocab_hi):
    import torch
    en_seqs, hi_seqs = zip(*batch)
    def pad(l):
        max_len = max(len(x) for x in l)
        return torch.tensor([x + [PAD_ID] * (max_len - len(x)) for x in l], dtype=torch.long)
    return pad(en_seqs), pad(hi_seqs)

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab_en, vocab_hi):
        self.en = df['en'].tolist()
        self.hi = df['hi'].tolist()
        self.vocab_en = vocab_en
        self.vocab_hi = vocab_hi

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return encode(self.en[idx], self.vocab_en), encode(self.hi[idx], self.vocab_hi)

# -------------------------
# Decoding: greedy + beam
# -------------------------
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len=40, start_id=SOS_ID, end_id=EOS_ID):
    model.eval()
    device = src.device
    batch = src.size(0)
    trg = torch.ones(batch, 1, dtype=torch.long, device=device) * start_id

    for _ in range(max_len):
        trg_mask = create_decoder_mask(trg).to(device)
        logits, _ = model(src, trg, src_mask, trg_mask)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        trg = torch.cat([trg, next_token], dim=1)
        if (next_token == end_id).all():
            break
    return trg

@torch.no_grad()
def beam_search(model, src, src_mask, beam_width=3, max_len=40, start_id=SOS_ID, end_id=EOS_ID):
    # Simple beam_search for a single example (batch size 1)
    device = src.device
    sequences = [[ [start_id], 0.0 ]]
    for _ in range(max_len):
        all_cands = []
        for seq, score in sequences:
            trg = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            trg_mask = create_decoder_mask(trg).to(device)
            logits, _ = model(src, trg, src_mask, trg_mask)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            topk = torch.topk(logp, beam_width)
            for i in range(beam_width):
                token = int(topk.indices[0, i].item())
                new_seq = seq + [token]
                new_score = score + float(topk.values[0, i].item())
                all_cands.append([new_seq, new_score])
        sequences = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(s[0][-1] == end_id for s in sequences):
            break
    return torch.tensor(sequences[0][0], dtype=torch.long, device=device)

# -------------------------
# Training / evaluation
# -------------------------
def train_epoch(model, dataloader, optimizer, loss_fn, pad_id=PAD_ID):
    model.train()
    total_loss = 0.0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)
        trg_mask_full = create_decoder_mask(trg, pad_id).to(device)

        trg_input = trg[:, :-1]
        trg_target = trg[:, 1:]

        # Use trg_mask for trg_input (remove last timestep from mask)
        trg_mask = trg_mask_full[:, :, :-1, :-1]

        logits, _ = model(src, trg_input, src_mask, trg_mask)
        loss = loss_fn(logits, trg_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, pad_id=PAD_ID):
    model.eval()
    total_loss = 0.0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)
        trg_mask_full = create_decoder_mask(trg, pad_id).to(device)

        trg_input = trg[:, :-1]
        trg_target = trg[:, 1:]
        trg_mask = trg_mask_full[:, :, :-1, :-1]

        logits, _ = model(src, trg_input, src_mask, trg_mask)
        loss = loss_fn(logits, trg_target)
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def compute_bleu(model, dataloader, vocab_hi, max_eval=500, beam_width=3):
    model.eval()
    hyps = []
    refs = []
    count = 0
    for src, trg in tqdm(dataloader, desc="BLEU eval"):
        src = src.to(device)
        trg = trg.to(device)
        for b in range(src.size(0)):
            src_b = src[b].unsqueeze(0)
            trg_b = trg[b].unsqueeze(0)
            src_mask = create_padding_mask(src_b, PAD_ID).to(device)

            pred_ids = beam_search(model, src_b, src_mask, beam_width=beam_width)
            pred_text = decode_ids(pred_ids.tolist(), vocab_hi)
            tgt_text = decode_ids(trg_b[0].tolist(), vocab_hi)

            hyps.append(pred_text)
            refs.append([tgt_text])

            count += 1
            if count >= max_eval:
                break
        if count >= max_eval:
            break

    bleu = corpus_bleu(hyps, list(zip(*refs)))
    return bleu.score

# -------------------------
# Save / plotting helpers
# -------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def plot_losses(train_losses, val_losses, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------
# Main training script
# -------------------------
def main():
    # Load dataset
    df = load_dataset(DATA_CSV, max_rows=10000)
    split = int(0.8 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)

    # Build vocabularies (or load if saved)
    # Build English vocab from train
    vocab_en = build_vocab(train_df['en'].tolist(), min_freq=1)
    vocab_hi = build_vocab(train_df['hi'].tolist(), min_freq=1)

    save_vocab(vocab_en, os.path.join(VOCAB_DIR, "vocab_en.json"))
    save_vocab(vocab_hi, os.path.join(VOCAB_DIR, "vocab_hi.json"))

    # Create dataloaders
    train_ds = TranslationDataset(train_df, vocab_en, vocab_hi)
    val_ds = TranslationDataset(val_df, vocab_en, vocab_hi)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=lambda b: collate_batch(b, vocab_en, vocab_hi))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch(b, vocab_en, vocab_hi))

    # Instantiate model / optimizer / loss / scheduler
    model = Transformer(src_vocab_size=len(vocab_en), trg_vocab_size=len(vocab_hi),
                        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                        d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = LabelSmoothingLoss(LABEL_SMOOTHING, trg_vocab_size=len(vocab_hi), ignore_index=PAD_ID)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

    # Save artifacts
    save_model(model, os.path.join(MODEL_DIR, "transformer_model.pth"))
    plot_losses(train_losses, val_losses, os.path.join(IM_DIR, "loss_curve.png"))

    # BLEU (sample)
    bleu = compute_bleu(model, val_loader, vocab_hi, max_eval=300, beam_width=3)
    print("BLEU (sample):", bleu)

    # Save sample translations (few)
    src_batch, trg_batch = next(iter(val_loader))
    src_batch = src_batch.to(device)
    src_mask = create_padding_mask(src_batch, PAD_ID).to(device)
    preds = greedy_decode(model, src_batch, src_mask, max_len=40)
    samples = []
    for i in range(min(10, src_batch.size(0))):
        samples.append({
            "en": decode_ids(src_batch[i].tolist(), vocab_en),
            "hi_true": decode_ids(trg_batch[i].tolist(), vocab_hi),
            "hi_pred": decode_ids(preds[i].tolist(), vocab_hi)
        })
    pd.DataFrame(samples).to_csv(os.path.join(METRICS_DIR, "sample_translations.csv"), index=False)

if __name__ == "__main__":
    main()
