# src/model.py
"""
Transformer encoder-decoder model.
Relies on:
 - layers.py (MultiHeadAttention, FeedForward, ResidualBlock)
 - utils.py (PositionalEncoding, create_padding_mask, create_decoder_mask)
"""

import torch
import torch.nn as nn
from layers import MultiHeadAttention, FeedForward, ResidualBlock
from utils import PositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.res1 = ResidualBlock(d_model, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.res2 = ResidualBlock(d_model, dropout)

    def forward(self, x, src_mask=None):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.res1(x, attn_out)

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.res2(x, ffn_out)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.res1 = ResidualBlock(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.res2 = ResidualBlock(d_model, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.res3 = ResidualBlock(d_model, dropout)

    def forward(self, x, enc_out, trg_mask=None, src_mask=None):
        # Masked self-attention
        self_attn_out, _ = self.self_attn(x, x, x, trg_mask)
        x = self.res1(x, self_attn_out)

        # Cross-attention (queries from decoder, keys/values from encoder)
        cross_attn_out, attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.res2(x, cross_attn_out)

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.res3(x, ffn_out)

        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (B, L)
        x = self.embedding(src) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x  # (B, L, D)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, trg, enc_out, trg_mask=None, src_mask=None):
        # trg: (B, L)
        x = self.embedding(trg) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_enc(x)
        x = self.dropout(x)

        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, enc_out, trg_mask, src_mask)
            attn_maps.append(attn_weights)

        x = self.layernorm(x)
        return x, attn_maps


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 d_model=384,
                 num_heads=6,
                 num_layers=4,
                 d_ff=768,
                 dropout=0.1,
                 max_len=200):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.output_layer = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # src: (B, L_src)
        # trg: (B, L_trg)
        enc_out = self.encoder(src, src_mask)
        dec_out, attn_maps = self.decoder(trg, enc_out, trg_mask, src_mask)
        logits = self.output_layer(dec_out)  # (B, L_trg, vocab)
        return logits, attn_maps
