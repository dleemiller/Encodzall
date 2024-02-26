import torch
from torch import nn
from typing import Optional


class ContextEncoder(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.embedding_dim = config.word_encoder.embedding_dim

        self.seq_emb = nn.Embedding(config.tokenizer.max_words, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            dim_feedforward=int(config.word_encoder.ff_mult * self.embedding_dim),
            nhead=config.context_encoder.heads,
            batch_first=True,
            norm_first=config.context_encoder.prenorm,
            activation="gelu",
            dropout=config.context_encoder.attention_dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.context_encoder.depth
        )
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x, mask):
        word_num = torch.cumsum(mask, dim=1) * mask
        x += self.seq_emb(word_num)
        return self.transformer(
            self.norm(x), src_key_padding_mask=mask.logical_not().float()
        )
