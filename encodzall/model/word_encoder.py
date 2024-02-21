import torch
from torch import nn

from .local_transformer import LocalTransformer
from .word_embedding import WordEmbedding
from ..utils.gather_vectors import gather_word_starts


class WordEncoder(nn.Module):
    """
    Encodes character-level tokens into vectors using local windowed attention
    from lucidrains.
    """

    def __init__(self, config):
        super().__init__()

        self.word_embedding = WordEmbedding(config)
        self.word_encoder = LocalTransformer(config)

    def forward(self, input_ids, attention_mask, word_start, max_words=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)
            attention_mask = attention_mask.unsqueeze(dim=0)
            word_start = word_start.unsqueeze(dim=0)

        input_ids = input_ids.long()
        x = self.word_embedding(input_ids, word_start)
        x = self.word_encoder(x, attention_mask)
        return gather_word_starts(x, word_start, max_words=max_words)
