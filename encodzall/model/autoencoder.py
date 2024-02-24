import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from encodzall import Tokenizer
from encodzall.model import WordEncoder, Decoder


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_encoder = WordEncoder(config)
        self.decoder = Decoder(
            config, embed_weight=self.word_encoder.word_embedding.token_emb.weight
        )
        self.max_words = config.tokenizer.max_words
        self.to_word_batch = Rearrange("b s w -> (b s) w")

    def forward(
        self, input_ids, attention_mask, word_start, target_ids, target_mask, **kwargs
    ):
        embeddings, word_mask = self.word_encoder(
            input_ids, attention_mask, word_start, max_words=self.max_words
        )
        embeddings = self.to_word_batch(embeddings)
        target_ids = self.to_word_batch(target_ids)
        target_mask = self.to_word_batch(target_mask)
        return self.decoder(target_ids, embeddings, target_mask)
