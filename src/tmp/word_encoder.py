import torch
import torch.nn as nn
import math
from typing import List, Tuple

from encoder import TransformerEncoder
from unpad_sequences import UnpadSequences


class WordPooling(nn.Module):
    def __init__(self, pooling_type: str = "average"):
        """
        Initializes the WordPooling module.

        Args:
            pooling_type (str): Type of pooling to perform ('average' or 'max').
        """
        super(WordPooling, self).__init__()
        assert pooling_type in [
            "average",
            "max",
        ], "pooling_type must be 'average' or 'max'"
        self.pooling_type = pooling_type

    def forward(
        self, hidden_states: torch.Tensor, word_boundaries: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Performs pooling over word vectors based on the word boundaries.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dim).
            word_boundaries (List[List[Tuple[int, int]]]): List of word boundaries per batch.
                Each element in the list corresponds to a batch element and contains a list of (start, end) tuples.

        Returns:
            torch.Tensor: Tensor of shape (total_words, hidden_dim) containing pooled word vectors.
        """
        pooled_words = []

        for batch_idx, boundaries in enumerate(word_boundaries):
            try:
                for start, end in boundaries:
                    word_vectors = hidden_states[
                        batch_idx, start:end, :
                    ]  # Shape: (word_len, hidden_dim)
                    if word_vectors.size(0) == 0:
                        continue  # Skip if no vectors to pool
                    if self.pooling_type == "average":
                        pooled = word_vectors.mean(dim=0)
                    else:
                        pooled, _ = word_vectors.max(dim=0)
                    pooled_words.append(pooled)
            except Exception as e:
                print(boundaries)
                raise e

        if pooled_words:
            return torch.stack(pooled_words, dim=0)  # Shape: (total_words, hidden_dim)
        else:
            return torch.empty(
                0, hidden_states.size(-1), device=hidden_states.device
            )  # Return empty tensor if no words


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        """
        Sinusoidal positional encoding module.

        Args:
            d_model (int): Dimension of the embeddings.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional encodings for a sequence length.

        Args:
            seq_length (int): Length of the input sequence.
            device (torch.device): Device where the tensor will be placed.

        Returns:
            torch.Tensor: Positional encoding tensor of shape (seq_length, d_model).
        """
        position = torch.arange(
            seq_length, dtype=torch.float32, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pos_encoding = torch.zeros((seq_length, self.d_model), device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding


class WordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        activation="gelu",
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 768,
        pooling_type: str = "average",
    ):
        """
        Initializes the TransformerEncoderModule with WordPooling.

        Args:
            vocab_size (int): Size of the tokenizer vocabulary. Default is 256.
            d_model (int): Dimension of the embeddings and Transformer. Default is 512.
            nhead (int): Number of attention heads. Default is 8.
            num_layers (int): Number of Transformer encoder layers. Default is 6.
            dim_feedforward (int): Dimension of the feedforward network. Default is 2048.
            dropout (float): Dropout rate. Default is 0.1.
            max_seq_length (int): Maximum sequence length for RoPE. Default is 512.
            activation (str): Activation function in the feedforward network. Default is "relu".
            batch_first (bool): If True, input and output tensors are (batch, seq, feature). Default is True.
            pooling_type (str): Type of pooling to perform ('average' or 'max'). Default is 'average'.
        """
        super(WordTransformer, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        assert d_model % 2 == 0, "d_model must be even for RoPE."

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.encoder = TransformerEncoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Positional encodings for target sequences
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)

        # Initialize WordPooling
        self.word_pooling = WordPooling(pooling_type=pooling_type)
        self.unpad_sequences = UnpadSequences()
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
        word_boundaries: List[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length) with token indices.
            sequence_ids (torch.Tensor): Sequence IDs for unpadding.
            target_ids (torch.Tensor): Target tensor of shape (batch_size, target_seq_length) with token indices.
            attention_mask (torch.Tensor, optional): Attention mask for encoder of shape (batch_size, seq_length, seq_length).
            word_boundaries (List[List[Tuple[int, int]]], optional): Word boundaries for word pooling.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        batch_size, seq_length = x.size()

        # Shape: (batch_size, seq_length, d_model)
        embedded = self.embedding(x.long()) * math.sqrt(self.d_model)
        transformer_output = self.encoder(embedded, attention_mask)

        # pooled_word_vectors shape: (total_words, hidden_dim)
        pooled_word_vectors = self.word_pooling(transformer_output, word_boundaries)

        # arrange to block vector
        memory, memory_key_padding_mask = self.unpad_sequences(
            pooled_word_vectors, sequence_ids
        )
        memory.contiguous()

        # Decoder
        target_seq_length = target_ids.size(1)
        target_embedded = self.embedding(target_ids.long()) + self.positional_encoding(
            seq_length=target_seq_length, device=target_ids.device
        )

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_seq_length
        ).to(target_ids.device)
        decoder_output = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_is_causal=True,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )

        logits = self.output_layer(decoder_output)
        print(target_ids[:, 1:])
        print(torch.argmax(logits, dim=-1)[:, :-1], logits.shape)
        return logits[:, :-1, :].contiguous()


# Example Usage
if __name__ == "__main__":
    from tokenizer import ByteLevelTokenizer

    # Initialize the tokenizer and pooling module
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    pooling_type = "average"  # Change to 'max' for max pooling

    # Initialize the Transformer Encoder with WordPooling
    model = WordTransformer(
        vocab_size=256,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=512,
        activation="gelu",
        pooling_type=pooling_type,
    )

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sample text
    sample_text1 = (
        "Hello world! This is a sample text to encode. "
        "Tokenize the input text into byte token IDs and create the corresponding attention mask."
    )
    sample_text2 = (
        "Hello world! This is a sample text to encode. "
        "Tokenize the input text into byte token IDs and create the corresponding attention mask."
        "Assuming tokenizer.tokenize returns tokens as a list of lists for batch processing"
    )
    target_ids, target_key_padding_mask = tokenizer.create_targets(
        [sample_text1, sample_text2]
    )

    batch = []
    tokens = []
    mask = []
    boundaries = []
    sequence_ids = []
    for i, text in enumerate([sample_text1, sample_text2]):
        # Tokenize the sample text
        tokens_i, mask_i, boundaries_i = tokenizer.tokenize(text)
        print(f"Token IDs:\n{tokens_i}\nToken IDs shape: {tokens_i.shape}")
        print(f"\nAttention Mask:\n{mask_i}\nMask Shape: {mask_i.shape}")
        print(f"\nWord Boundaries:\n {boundaries_i}")

        tokens.append(tokens_i)
        mask.append(mask_i)
        boundaries.extend(boundaries_i)
        sequence_ids.extend([i] * sum([len(x) for x in boundaries_i]))

    # Convert tokens and mask to tensors
    tokens_tensor = torch.cat(tokens, dim=0).to(
        device
    )  # Shape: (batch_size, seq_length)
    attention_mask_tensor = torch.cat(mask, dim=0).to(
        device
    )  # Shape: (seq_length, seq_length)
    sequence_ids = torch.tensor(sequence_ids).to(device)
    target_ids = target_ids.to(device)
    target_key_padding_mask = target_key_padding_mask.to(device)
    word_boundaries_list = boundaries
    print(tokens_tensor.shape, attention_mask_tensor.shape)
    print("Sequences: ", sequence_ids, sequence_ids.shape)

    # Forward pass through the model
    output = model(
        tokens_tensor,
        sequence_ids=sequence_ids,
        target_ids=target_ids,
        target_key_padding_mask=target_key_padding_mask,
        attention_mask=attention_mask_tensor,
        word_boundaries=word_boundaries_list,
    )

    print(output)
