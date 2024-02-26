import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from encodzall import Tokenizer
from encodzall.model import WordEncoder, ContextEncoder, Decoder


class ElasticNetLoss(nn.Module):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        """
        Initializes the ElasticNetLoss module.
        :param alpha: The overall regularization strength.
        :param l1_ratio: The balance between L1 and L2 regularization, with 0 < l1_ratio < 1.
                         l1_ratio = 1 corresponds to L1 regularization only, and l1_ratio = 0 corresponds to L2 regularization only.
        """
        super(ElasticNetLoss, self).__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, weights):
        """
        Computes the Elastic Net loss given the weights.
        :param weights: The weights of the model parameters you want to regularize.
        :return: The Elastic Net regularization term.
        """
        l1_term = torch.sum(torch.abs(weights))
        l2_term = torch.sum(weights**2)
        div = torch.ones_like(weights).sum()
        elastic_net_loss = (
            self.alpha
            * (self.l1_ratio * l1_term + (1 - self.l1_ratio) * 0.5 * l2_term)
            / div
        )
        return elastic_net_loss


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_encoder = WordEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.decoder = Decoder(
            config, embed_weight=self.word_encoder.word_embedding.token_emb.weight
        )
        self.max_words = config.tokenizer.max_words
        self.to_word_batch = Rearrange("b s w -> (b s) w")
        self.norm = nn.LayerNorm(config.word_encoder.embedding_dim)
        self.elasticnet_loss = ElasticNetLoss(alpha=1.0, l1_ratio=0.5)

    def forward(
        self, input_ids, attention_mask, word_start, target_ids, target_mask, **kwargs
    ):
        embeddings, word_mask = self.word_encoder(
            input_ids, attention_mask, word_start, max_words=self.max_words
        )

        context = self.context_encoder(embeddings, word_mask)
        embeddings = self.norm(embeddings + context)
        embeddings = self.to_word_batch(embeddings)
        if kwargs.get("skip_decode", False):
            return embeddings

        target_ids = self.to_word_batch(target_ids)
        target_mask = self.to_word_batch(target_mask)
        preds, target, loss = self.decoder(target_ids, embeddings, target_mask)

        en_loss = self.elasticnet_loss(context)
        return (preds, target, loss + en_loss), embeddings
