"""RoboBytes models over a byte stream (vocabulary 258 = 256 bytes + BOS + SEP).

- TokenLM: embedding -> causal Transformer -> softmax. Used for Byte (one position per byte) and BPE (one per token).
- ByteHNet: 1-stage H-Net (Hwang, Wang & Gu 2025). byte embedding -> causal conv encoder -> router -> downsample ->
  main Transformer on chunks -> EMA dechunk + causal upsample (STE) + residual -> causal conv decoder -> softmax.
  `boundaries` given = Fixed / Entropy baselines (same architecture, external boundaries).
The router, ratio loss, STE and refractory rule are RoboChunk's (robochunk/model.py), unchanged in maths.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BYTES, BOS, SEP = 256, 256, 257
VOCAB = 258


def rope(x, base=10000.0):
    """Rotary position embedding on (B, h, L, dh), rotating pairs (even, odd) of each head dimension."""
    L, dh = x.shape[-2], x.shape[-1]
    inv = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    ang = torch.arange(L, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    """Pre-norm Transformer block; attention through scaled_dot_product_attention with is_causal=True (no L x L mask)."""

    def __init__(self, d, heads, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Main(nn.Module):
    """Causal Transformer with a final norm (shared by every method; only its size varies). Positions: learned absolute
    embeddings ("learned") or rotary embeddings in attention ("rope")."""

    def __init__(self, d, layers, heads, max_len=1024, pos="learned"):
        super().__init__()
        self.pos = nn.Embedding(max_len, d) if pos == "learned" else None
        self.blocks = nn.ModuleList([Block(d, heads, use_rope=pos == "rope") for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        if self.pos is not None:
            h = h + self.pos(torch.arange(h.shape[1], device=h.device))
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


def gpt2_init(main, embeddings=()):
    """GPT-2 initialisation for the main Transformer and the given embeddings: N(0, 0.02) weights, zero biases, output
    projections of each block scaled by 1 / sqrt(2 x layers). The H-Net router, residual and conv weights keep theirs."""
    L = len(main.blocks)
    for mod in list(main.modules()) + list(embeddings):
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, std=0.02)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)
    for b in main.blocks:
        nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * L))
        nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * L))


class TokenLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, max_len=1024, pos="learned", init="default"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.main = Main(d, layers, heads, max_len, pos)
        if init == "gpt2":
            gpt2_init(self.main, [self.emb])
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return dict(logits=self.head(self.main(self.emb(ids))))


# ---- from RoboChunk (unchanged maths) ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Causal depthwise conv (kernel 4) + gated MLP: a cheap stand-in for a Mamba layer (Mamba kernels do not run on MPS)."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv_w = nn.Parameter(torch.randn(k, d) / math.sqrt(k))
        self.conv_b = nn.Parameter(torch.zeros(d))
        self.w1, self.w3 = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)
        self.w2 = nn.Linear(2 * d, d)

    def forward(self, x):
        h = self.n1(x)
        T = h.shape[1]
        hp = F.pad(h, (0, 0, self.k - 1, 0))
        x = x + sum(hp[:, j:j + T] * self.conv_w[j] for j in range(self.k)) + self.conv_b
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


def ste(c):
    """Forward value 1, gradient of c (straight-through)."""
    return c + (1 - c).detach()


def ratio_loss(b, p, N):
    """H-Net ratio loss: N/(N-1) * ((N-1) F G + (1-F)(1-G)), F = boundary fraction, G = mean p."""
    Fr, G = b.float().mean(), p.mean()
    return N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))


def suppress(b, r):
    """Causal refractory period (RoboChunk v2): drop a boundary at t if one was kept in t-r..t-1; position 0 is kept.
    Non-differentiable, so it runs on CPU in numpy (a per-position loop of small MPS kernels is slow)."""
    a = b.cpu().numpy().copy()
    a[:, 0] = True
    last = np.zeros(a.shape[0], dtype=np.int64)
    for t in range(1, a.shape[1]):
        keep = a[:, t] & (t - last > r)
        a[:, t] = keep
        last = np.where(keep, t, last)
    return torch.from_numpy(a).to(b.device)
# --------------------------------------------------------------------------------------------------------------------


def ema(z, P):
    """h_j = P_j z_j + (1 - P_j) h_{j-1}, computed in parallel: h = W z with W[j,i] = P_i prod_{i<k<=j} (1 - P_k).
    Equal to RoboChunk's sequential loop up to the 1e-6 clamp on (1 - P)."""
    C = torch.log((1 - P).clamp(min=1e-6)).cumsum(1)                     # (B, K)
    K = P.shape[1]
    tri = torch.ones(K, K, dtype=torch.bool, device=z.device).tril()
    W = torch.exp((C.unsqueeze(2) - C.unsqueeze(1)).masked_fill(~tri, -float("inf"))) * P.unsqueeze(1)
    return W @ z


class ByteHNet(nn.Module):
    def __init__(self, d_main, layers, heads, d_outer=256, outer_blocks=4, refractory=0, max_chunks=1024, k_bucket=64,
                 pos="learned", init="default"):
        super().__init__()
        self.refractory = refractory
        self.k_bucket = k_bucket  # chunk count rounded up to a multiple of this: few distinct shapes, no MPS fragmentation
        self.emb = nn.Embedding(VOCAB, d_outer)
        self.encoder = nn.Sequential(*[ConvBlock(d_outer) for _ in range(outer_blocks)], nn.LayerNorm(d_outer))
        self.router_norm = nn.LayerNorm(d_outer)
        self.wq, self.wk = nn.Linear(d_outer, d_outer, bias=False), nn.Linear(d_outer, d_outer, bias=False)
        with torch.no_grad():  # identity init (RoboChunk)
            self.wq.weight.copy_(torch.eye(d_outer))
            self.wk.weight.copy_(torch.eye(d_outer))
        self.up_proj, self.down_proj = nn.Linear(d_outer, d_main), nn.Linear(d_main, d_outer)
        self.main = Main(d_main, layers, heads, max_chunks, pos)
        if init == "gpt2":
            gpt2_init(self.main, [self.emb])
        self.residual = nn.Linear(d_outer, d_outer)
        with torch.no_grad():  # H-Net initialises the residual projection close to 0
            self.residual.weight.mul_(1e-3)
            self.residual.bias.zero_()
        self.decoder = nn.Sequential(*[ConvBlock(d_outer) for _ in range(outer_blocks)], nn.LayerNorm(d_outer))
        self.head = nn.Linear(d_outer, VOCAB)

    def route(self, xh):
        r = self.router_norm(xh)
        q, k = self.wq(r), self.wk(r)
        cos = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p = (0.5 * (1 - cos)).clamp(0, 1)
        return torch.cat([torch.ones_like(p[:, :1]), p], 1)

    def forward(self, ids, boundaries=None):
        """ids: (B, L) symbols. boundaries: (B, L) bool for Fixed/Entropy, or None for the learned router.
        Returns logits (B, L, VOCAB) predicting ids[:, 1:] from position t, plus p, b and K (chunks per row)."""
        xh = self.encoder(self.emb(ids))
        if boundaries is None:
            p = self.route(xh)
            b = p >= 0.5
            if self.refractory:
                b = suppress(b, self.refractory)
        else:
            b = boundaries.bool()
            p = b.float()
        b = b.clone()
        b[:, 0] = True
        idx = b.long().cumsum(1) - 1
        nk = b.sum(1)
        K = min(ids.shape[1], -(-int(nk.max()) // self.k_bucket) * self.k_bucket) if self.k_bucket else int(nk.max())
        bi, ti = b.nonzero(as_tuple=True)
        chunks = xh.new_zeros(ids.shape[0], K, xh.shape[-1])
        chunks[bi, idx[bi, ti]] = xh[bi, ti]
        P = xh.new_zeros(ids.shape[0], K)
        P[bi, idx[bi, ti]] = p[bi, ti]
        zh = self.down_proj(self.main(self.up_proj(chunks)))
        zbar = zh if boundaries is not None else ema(zh, P)
        up = torch.gather(zbar, 1, idx.unsqueeze(-1).expand(-1, -1, zbar.shape[-1]))
        c = torch.where(b, p, 1 - p)
        up = up * ste(c).unsqueeze(-1)
        h = self.decoder(up + self.residual(xh))
        return dict(logits=self.head(h), p=p, b=b, n_chunks=nk)


def n_params(m, exclude=("pos",)):
    return sum(p.numel() for n, p in m.named_parameters() if not any(e in n for e in exclude))
