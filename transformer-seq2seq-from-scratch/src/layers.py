import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------------------
# SCALED DOT-PRODUCT ATTENTION
# --------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V, attn

# --------------------------
# MULTI-HEAD ATTENTION
# --------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, D = Q.size()

        q = self.W_q(Q)
        k = self.W_k(K)
        v = self.W_v(V)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        out, attn = self.attention(q, k, v, mask)

        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.linear(out)

        return out, attn

# --------------------------
# FEED FORWARD NETWORK
# --------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# RESIDUAL CONNECTION + NORM
# --------------------------

class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer))
