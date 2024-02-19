import torch
from torch import nn

from .local_transformer import LocalTransformer
from ..utils.gather_vectors import gather_word_starts


class WordEncoder(nn.Module):
    """
    Encodes character-level tokens into vectors using local windowed attention
    from lucidrains.
    """

    def __init__(self):
        super().__init__()

        self.word_encoder = LocalTransformer(
            num_tokens=256,
            dim=256,
            depth=6,
            max_seq_len=8192,
            local_attn_window_size=64,
            look_forward=True,
        )

    def forward(self, input_ids, attention_mask, word_start):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)
            attention_mask = attention_mask.unsqueeze(dim=0)
            word_start = word_start.unsqueeze(dim=0)

        input_ids = input_ids.long()
        x = self.word_encoder(input_ids, attention_mask, word_start)
        return gather_word_starts(x, word_start)
