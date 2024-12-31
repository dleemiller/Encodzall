# model.py
from contextlib import contextmanager
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

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

        # Word Decoder
        word_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.word_decoder = nn.TransformerDecoder(word_decoder_layer, num_layers=1)

        # Positional encodings for memory and target sequences
        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model)

        # Word pooling and unpadding
        self.word_pooling = WordPooling(pooling_type=config.pooling_type)
        self.unpad_sequences = UnpadSequences()

        # Output layer
        self.output_layer = nn.Linear(self.d_model, config.vocab_size)

    @contextmanager
    def set_dropout(self, dropout: float):
        """
        Temporarily updates the dropout rate for all layers during a forward pass.

        Args:
            dropout (float): The dropout rate to use temporarily.
        """
        original_dropout_rates = {}
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                original_dropout_rates[module] = module.p
                module.p = dropout
        try:
            yield
        finally:
            for module, original_rate in original_dropout_rates.items():
                module.p = original_rate

    @staticmethod
    def key_padding_to_attention_mask(key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a key padding mask into an attention mask.

        Args:
            key_padding_mask (torch.Tensor): Tensor of shape (batch_size, seq_length)
                                             where True indicates padding tokens.

        Returns:
            torch.Tensor: Attention mask of shape (batch_size, seq_length, seq_length)
                          with 1 for allowed attention and 0 for blocked attention.
        """
        # Invert the mask: True -> 0, False -> 1
        inverted_mask = ~key_padding_mask  # Shape: (batch_size, seq_length)

        # Expand and multiply to create pairwise mask
        attention_mask = inverted_mask.unsqueeze(1) & inverted_mask.unsqueeze(2)
        return attention_mask.bool()

    @staticmethod
    def flatten_and_filter(tensor, key_padding_mask):
        """
        Flattens the first two dimensions of the tensor and filters out padding tokens.

        Args:
            tensor (torch.Tensor): Input tensor of shape (b, s, d_model).
            key_padding_mask (torch.Tensor): Mask tensor of shape (b, s), where
                                             True indicates padding and False indicates valid tokens.

        Returns:
            torch.Tensor: Filtered tensor of shape (N, d_model), where N is the number of valid tokens.
        """
        # Ensure the mask is of boolean type
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()

        # Invert the mask to get valid tokens (False -> True, True -> False)
        valid_mask = ~key_padding_mask  # Shape: (b, s)

        # Flatten the tensor and mask
        b, s, d_model = tensor.shape
        flattened_tensor = tensor.view(-1, d_model)  # Shape: (b*s, d_model)
        flattened_mask = valid_mask.view(-1)  # Shape: (b*s)

        # Select only the valid vectors
        filtered_tensor = flattened_tensor[flattened_mask]  # Shape: (N, d_model)

        return filtered_tensor.unsqueeze(dim=1)

    def average_pool_memory(
        self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs average pooling on the memory vectors, respecting the padding mask.

        Args:
            memory (torch.Tensor): Memory tensor from the second encoder of shape (batch_size, seq_length, d_model).
            memory_key_padding_mask (torch.Tensor): Padding mask for the memory of shape (batch_size, seq_length),
                                                    where True indicates padding tokens.

        Returns:
            torch.Tensor: Average pooled embeddings of shape (batch_size, d_model).
        """
        # Invert mask: True for valid tokens, False for padding
        valid_mask = ~memory_key_padding_mask  # Shape: (batch_size, seq_length)

        # Expand mask for multiplication
        valid_mask_expanded = valid_mask.unsqueeze(
            -1
        ).float()  # Shape: (batch_size, seq_length, 1)

        # Zero out the padded tokens
        memory_masked = (
            memory * valid_mask_expanded
        )  # Shape: (batch_size, seq_length, d_model)

        # Sum the memory vectors
        sum_memory = memory_masked.sum(dim=1)  # Shape: (batch_size, d_model)

        # Count the number of valid (non-padded) tokens
        counts = valid_mask_expanded.sum(dim=1)  # Shape: (batch_size, 1)

        # Avoid division by zero
        counts = counts.clamp(min=1e-9)

        # Compute the average pooled embeddings
        embeddings = sum_memory / counts  # Shape: (batch_size, d_model)

        return embeddings

    @staticmethod
    def get_eos_mask(sequence_ids):
        """
        Generates a mask and indices indicating the last occurrence of each sequence ID.

        Args:
            sequence_ids (list or tuple): List of sequence IDs.

        Returns:
            tuple: (mask tensor, indices tensor)
        """
        tensor = torch.tensor(sequence_ids)
        mask = tensor[:-1] != tensor[1:]
        mask = torch.cat([mask, torch.tensor([True]).to(sequence_ids.device)])
        return ~mask

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        seq_target_ids: torch.Tensor,
        seq_key_padding_mask: torch.Tensor,
        word_target_ids: torch.Tensor,
        word_key_padding_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        word_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        return_embeddings_only: bool = False,  # Added flag
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Encodzall model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length) with token indices.
            sequence_ids (torch.Tensor): Sequence IDs for unpadding.
            target_ids (torch.Tensor): Target tensor of shape (batch_size, target_seq_length) with token indices.
            target_key_padding_mask (torch.Tensor): Padding mask for the target.
            attention_mask (torch.Tensor, optional): Attention mask for the encoder of shape (batch_size, seq_length, seq_length).
            word_boundaries (List[List[Tuple[int, int]]], optional): Word boundaries for word pooling.
            return_embeddings_only (bool, optional): If set to True, returns only the pooled embeddings. Defaults to False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If `return_embeddings_only` is True, returns the pooled embeddings tensor of shape (batch_size, d_model).
                - Otherwise, returns a tuple containing:
                    1. Pooled embeddings tensor of shape (batch_size, d_model).
                    2. Logits tensor of shape (batch_size, target_seq_length - 1, vocab_size).
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
        memory_attn_mask = self.key_padding_to_attention_mask(memory_key_padding_mask)

        # Pass through the second encoder
        memory = self.encoder2(encoder1_output, attn_mask=memory_attn_mask)
        # Note: Positional encoding will be added after average pooling

        # ------------------------- Average Pooling Step Moved to Separate Function -------------------------
        # Average pool the memory vectors, respecting the memory_key_padding_mask
        embeddings = self.average_pool_memory(
            memory, memory_key_padding_mask
        )  # Shape: (batch_size, d_model)
        # -----------------------------------------------------------------------------------------------

        # Check if only embeddings should be returned
        if return_embeddings_only:
            return embeddings  # Early exit with embeddings

        # Add positional encodings to memory
        memory_positions = self.positional_encoding(
            seq_length=memory.size(1), device=device
        )

        # Embed target tokens
        seq_target_embedded = self.embedding(
            seq_target_ids.long()
        )  # Shape: (batch_size, target_seq_length, d_model)
        seq_target_positions = self.positional_encoding(
            seq_length=seq_target_ids.size(1), device=device
        )

        # Generate causal mask for decoder
        seq_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_target_ids.size(1)
        ).to(device)

        # Pass through the decoder
        seq_decoder_output = self.decoder(
            tgt=seq_target_embedded + seq_target_positions,
            memory=memory + memory_positions,
            tgt_mask=seq_causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=seq_key_padding_mask,
        )

        # Embed target tokens
        word_target_embedded = self.embedding(
            word_target_ids.long()
        )  # Shape: (batch_size, target_seq_length, d_model)
        word_target_positions = self.positional_encoding(
            seq_length=word_target_ids.size(1), device=device
        )

        # Generate causal mask for decoder
        word_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            word_target_ids.size(1)
        ).to(device)

        word_memory = self.flatten_and_filter(memory, memory_key_padding_mask)

        # remove sequence EOS tokens
        eos_mask = self.get_eos_mask(sequence_ids)
        word_memory = word_memory[eos_mask]

        # Pass through the decoder
        print(
            word_target_embedded.shape,
            word_target_positions.shape,
            word_memory.shape,
            word_key_padding_mask.shape,
        )
        word_decoder_output = self.decoder(
            tgt=word_target_embedded + word_target_positions,
            memory=word_memory,
            tgt_mask=word_causal_mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=word_key_padding_mask,
        )

        # Compute logits
        seq_logits = self.output_layer(seq_decoder_output)
        word_logits = self.output_layer(word_decoder_output)

        # Exclude last token for prediction
        seq_logits = seq_logits[:, :-1, :].contiguous()
        word_logits = word_logits[:, :-1, :].contiguous()

        # Return both embeddings and logits
        return embeddings, seq_logits, word_logits
