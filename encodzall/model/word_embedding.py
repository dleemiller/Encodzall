import torch
from torch import nn


class WordEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_seq_length = config.tokenizer.max_seq_length
        self.n_vocab = config.tokenizer.n_vocab
        self.initialize(config.word_encoder)

    def initialize(self, config):
        self.token_emb = nn.Embedding(self.n_vocab, config.embedding_dim)
        self.pos_emb = nn.Embedding(self.max_seq_length, config.embedding_dim)
        self.word_start_emb = nn.Embedding(2, config.embedding_dim)
        self.word_emb = nn.Embedding(self.max_seq_length, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, input_ids: torch.Tensor, word_start: torch.Tensor):
        n = input_ids.shape[1]
        device = input_ids.device
        assert n <= self.max_seq_length

        x = self.token_emb(input_ids)
        x += self.word_start_emb(word_start.long())
        x += self.word_emb(torch.cumsum(word_start.long(), dim=-1))
        x += self.pos_emb(torch.arange(n, device=device))
        return self.norm(x)
