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
    def __init__(self, config, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.local_attn_window_size = config.word_encoder.local_attn_window_size
        self.dynamic_pos_bias = None
        if config.word_encoder.use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(
                dim=config.embedding.embedding_dim // 2, heads=config.word_encoder.heads
            )

        for _ in range(config.word_encoder.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=config.embedding.embedding_dim,
                            dim_head=config.word_encoder.head_dim,
                            heads=config.word_encoder.heads,
                            dropout=config.word_encoder.attention_dropout,
                            causal=False,
                            window_size=config.word_encoder.local_attn_window_size,
                            use_xpos=False,
                            xpos_scale_base=None,
                            use_rotary_pos_emb=not config.word_encoder.use_dynamic_pos_bias,
                            prenorm=config.word_encoder.prenorm,
                            look_backward=False,
                            look_forward=True,
                            **kwargs
                        ),
                        FeedForward(
                            dim=config.embedding.embedding_dim,
                            mult=config.word_encoder.ff_mult,
                            dropout=config.word_encoder.ff_dropout,
                        ),
                    ]
                )
            )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(config.embedding.embedding_dim),
            nn.Linear(
                config.embedding.embedding_dim,
                config.embedding.embedding_dim,
                bias=False,
            ),
        )

    def forward(self, x, mask):
        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        return self.to_logits(x)
