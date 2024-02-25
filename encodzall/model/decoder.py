import torch
from torch import nn
from typing import Optional


class Decoder(nn.Module):

    def __init__(
        self,
        config,
        embed_weight=None,
    ):
        super().__init__()
        self.n_vocab = config.tokenizer.n_vocab
        self.max_word_length = config.tokenizer.max_word_length
        self.embedding_dim = config.word_encoder.embedding_dim
        self.bos_id = config.tokenizer.bos_id
        self.pad_id = config.tokenizer.pad_id

        self.token_emb = nn.Embedding(self.n_vocab, self.embedding_dim)
        self.pos_emb = nn.Embedding(self.max_word_length + 1, self.embedding_dim)
        # self.embedding_bias = nn.Parameter(torch.zeros(self.embedding_dim))
        self.register_buffer("bos_idx", torch.Tensor([self.bos_id]))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            dim_feedforward=int(config.word_encoder.ff_mult * self.embedding_dim) // 2,
            nhead=1,
            batch_first=True,
            norm_first=config.word_encoder.prenorm,
            activation="gelu",
            dropout=config.word_encoder.attention_dropout,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.to_logits = nn.Linear(
            config.word_encoder.embedding_dim,
            config.tokenizer.n_vocab,
            bias=False,
        )
        if embed_weight is not None:
            self.tie_embed_weights(embed_weight)

    @staticmethod
    def generate_square_subsequent_mask(x):
        """
        Generates attention mask for single shot decoding. Prevents lookahead
        during attention, since transformers are not by nature autoregressive.

        For decoding in inference, use autoregressive seq2seq method. This function
        from here, with good details about transformer decoder usage:

        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        sz = x.shape[1]
        mask = (torch.triu(torch.ones((sz, sz), device=x.device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def tie_embed_weights(self, embed_weight):
        # decoder is shared with embedding layer
        self.to_logits.weight = embed_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mem: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ):
        bs = mem.shape[0]
        device = mem.device
        mem = mem.unsqueeze(dim=1)

        # prepend bos token
        # embed tokens
        shifted_input_ids = torch.cat(
            [self.bos_idx.long().repeat(bs, 1), input_ids], dim=1
        )
        tgt = (
            self.token_emb(shifted_input_ids)
            + self.pos_emb(torch.arange(self.max_word_length + 1, device=device))
            # + self.embedding_bias
        )

        if pad_mask is not None:
            pad_mask = torch.cat(
                [
                    torch.ones(bs, 1, device=pad_mask.device).type(pad_mask.dtype),
                    pad_mask,
                ],
                dim=1,
            )

        # generate position mask
        tgt_mask = Decoder.generate_square_subsequent_mask(tgt)

        # decode and produce logits
        logits = self.to_logits(
            self.transformer(
                tgt,
                mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask.logical_not().float(),
            )
        )[:, :-1, :]

        # get valid word indices
        word_idx = torch.where(torch.any(pad_mask[:, 1:], dim=-1))[0]

        # reconstruction loss
        loss = nn.functional.cross_entropy(
            logits[word_idx].permute(0, 2, 1),
            input_ids[word_idx].long(),
            ignore_index=self.pad_id,
        )
        return logits[word_idx].permute(0, 2, 1), input_ids[word_idx], loss
