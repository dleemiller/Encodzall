import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from local_attention.local_attention import LocalAttention
from local_attention.transformer import LocalMHA, FeedForward, DynamicPositionBias


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal=False,
        local_attn_window_size=128,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ignore_index=-1,
        use_xpos=False,
        xpos_scale_base=None,
        use_dynamic_pos_bias=False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.word_start_emb = nn.Embedding(2, dim)
        self.word_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=causal,
                            window_size=local_attn_window_size,
                            use_xpos=use_xpos,
                            xpos_scale_base=xpos_scale_base,
                            use_rotary_pos_emb=not use_dynamic_pos_bias,
                            prenorm=True,
                            **kwargs
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.ignore_index = ignore_index
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False)
        )

    def forward(self, x, mask, word_start):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        # embed the position of the embedding vector
        x = x + self.word_start_emb(word_start.long())

        # embed the word positions themselves
        x = x + self.word_emb(torch.cumsum(word_start.long(), dim=-1))

        assert n <= self.max_seq_len
        x = x + self.pos_emb(torch.arange(n, device=device))

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        return self.to_logits(x)
