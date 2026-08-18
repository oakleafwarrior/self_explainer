"""A tiny pure-torch replica of the Qwen3 block structure, for offline testing.

`rotate.py` navigates a model by attribute name (`model.layers[i].self_attn.q_proj`,
`model.norm`, `lm_head`, ...). This module reproduces that structure — including GQA
and Qwen3's head-dimension QK-norms — in a few hundred lines with no `transformers`
dependency, so `test_rotate.py` can validate the invariance math anywhere.

It is a test fixture, not a model. Nothing in the experiment pipeline imports it.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class Attention(nn.Module):
    """Qwen3-style attention: GQA, no biases, RMSNorm on head-dim slices of q and k.

    q_norm/k_norm sit downstream of the read matrices and act inside a head, so a
    residual-stream rotation leaves them alone — the point of including them here.
    """

    def __init__(self, d, n_heads, n_kv, head_dim):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.q_proj = nn.Linear(d, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d, ff, n_heads, n_kv, head_dim):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = Attention(d, n_heads, n_kv, head_dim)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = MLP(d, ff)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniQwen3Model(nn.Module):
    def __init__(self, vocab, d, ff, n_layers, n_heads, n_kv, head_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [Block(d, ff, n_heads, n_kv, head_dim) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d)


class MiniQwen3ForCausalLM(nn.Module):
    """Same attribute layout as Qwen3ForCausalLM, so rotate.py cannot tell the difference."""

    def __init__(self, vocab=256, d=64, ff=128, n_layers=3, n_heads=4, n_kv=2,
                 head_dim=16, tie_word_embeddings=False):
        super().__init__()
        self.model = MiniQwen3Model(vocab, d, ff, n_layers, n_heads, n_kv, head_dim)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.config = SimpleNamespace(hidden_size=d, vocab_size=vocab,
                                      tie_word_embeddings=tie_word_embeddings)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kw):
        x = self.model.embed_tokens(input_ids)
        hidden = [x]
        for layer in self.model.layers:
            x = layer(x)
            hidden.append(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden) if output_hidden_states else None,
        )


def build(d=64, tied=False, seed=0, dtype=torch.float64):
    """A tiny model with *random* norm gains — g == 1 everywhere would make folding trivial."""
    torch.manual_seed(seed)
    m = MiniQwen3ForCausalLM(d=d, ff=2 * d, head_dim=d // 4, tie_word_embeddings=tied)
    for name, p in m.named_parameters():
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            p.data = 1.0 + 0.3 * torch.randn_like(p.data)
    return m.to(dtype).eval()
