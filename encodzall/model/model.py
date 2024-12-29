# model.py
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from ..config import TransformerConfig
from .encoder import TransformerEncoder
from .positional_encoding import SinusoidalPositionalEncoding
from .word_pooling import WordPooling
from .unpad_sequences import UnpadSequences


class Encodzall(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Initializes the Encodzall model.

        Args:
            config (TransformerConfig): Configuration parameters for the model.
        """
        super(Encodzall, self).__init__()
        self.config = config
        self.d_model = config.d_model

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=self.d_model
        )

        # Encoder layers
        self.encoder1 = TransformerEncoder(
            num_layers=config.num_encoder1_layers,
            d_model=self.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads_encoder1 or config.nhead // 2,
            head_dim=self.d_model // config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            attn_dropout=config.dropout,  # Adjust as needed
            max_seq_len=config.max_seq_length_encoder1,
            is_causal=False,
        )
        self.encoder2 = TransformerEncoder(
            num_layers=config.num_encoder2_layers,
            d_model=self.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads_encoder2 or config.nhead // 2,
            head_dim=self.d_model // config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            attn_dropout=config.dropout,  # Adjust as needed
            max_seq_len=config.max_seq_length_encoder2,
            is_causal=False,
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        # Positional encodings for memory and target sequences
        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model)

        # Word pooling and unpadding
        self.word_pooling = WordPooling(pooling_type=config.pooling_type)
        self.unpad_sequences = UnpadSequences()

        # Output layer
        self.output_layer = nn.Linear(self.d_model, config.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        word_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Encodzall model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length) with token indices.
            sequence_ids (torch.Tensor): Sequence IDs for unpadding.
            target_ids (torch.Tensor): Target tensor of shape (batch_size, target_seq_length) with token indices.
            target_key_padding_mask (torch.Tensor): Padding mask for the target.
            attention_mask (torch.Tensor, optional): Attention mask for the encoder of shape (batch_size, seq_length, seq_length).
            word_boundaries (List[List[Tuple[int, int]]], optional): Word boundaries for word pooling.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, target_seq_length - 1, vocab_size).
        """
        batch_size, seq_length = x.size()
        device = x.device

        # Embed input tokens
        embedded = self.embedding(x.long())  # Shape: (batch_size, seq_length, d_model)

        # Pass through the first encoder
        encoder1_output = self.encoder1(embedded, attn_mask=attention_mask)

        # Perform word pooling
        pooled_word_vectors = self.word_pooling(
            encoder1_output, word_boundaries
        )  # Shape: (total_words, d_model)

        # Unpad sequences to form memory
        encoder1_output, memory_key_padding_mask = self.unpad_sequences(
            pooled_word_vectors, sequence_ids
        )

        # Pass through the second encoder
        memory = self.encoder2(
            encoder1_output, attn_mask=memory_key_padding_mask.unsqueeze(1) == 0
        )
        memory += self.positional_encoding(seq_length=memory.size(1), device=device)

        # Embed target tokens
        target_embedded = self.embedding(
            target_ids.long()
        )  # Shape: (batch_size, target_seq_length, d_model)
        target_embedded += self.positional_encoding(
            seq_length=target_ids.size(1), device=device
        )

        # Generate causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_ids.size(1)
        ).to(device)

        # Pass through the decoder
        decoder_output = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )

        # Compute logits
        logits = self.output_layer(
            decoder_output
        )  # Shape: (batch_size, target_seq_length, vocab_size)

        return logits[:, :-1, :].contiguous()  # Exclude last token for prediction
