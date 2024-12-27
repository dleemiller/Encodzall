import torch
import torch.nn as nn
import math
from typing import List, Tuple

# Assuming ByteLevelTokenizer is defined in tokenizer.py
from tokenizer import ByteLevelTokenizer
from sequence_padding import LearnedPadding


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
            print(batch_idx, boundaries)
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

        if pooled_words:
            return torch.stack(pooled_words, dim=0)  # Shape: (total_words, hidden_dim)
        else:
            return torch.empty(
                0, hidden_states.size(-1), device=hidden_states.device
            )  # Return empty tensor if no words


class TransformerEncoderModule(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        activation: str = "gelu",
        batch_first: bool = True,
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
        super(TransformerEncoderModule, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        assert d_model % 2 == 0, "d_model must be even for RoPE."

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Initialize Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Initialize WordPooling
        self.word_pooling = WordPooling(pooling_type=pooling_type)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        word_boundaries: List[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the TransformerEncoderModule with WordPooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length) with token indices.
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length, seq_length),
                                                    where `True` indicates positions to be masked.
            word_boundaries (List[List[Tuple[int, int]]], optional):
                List of word boundaries per batch. Each element corresponds to a batch element and contains
                a list of (start, end) tuples indicating the token indices for each word.

        Returns:
            torch.Tensor: Tensor of shape (total_words, hidden_dim) containing pooled word vectors.
        """
        batch_size, seq_length = x.size()

        # Step 1: Embed the input tokens
        # Shape: (batch_size, seq_length, d_model)
        embedded = self.embedding(x.long()) * math.sqrt(self.d_model)

        # Step 2: Apply Rotary Positional Encoding (RoPE)
        # Shape: (batch_size, seq_length, d_model)
        embedded = self.apply_rope(embedded)

        # Step 3: Pass through Transformer Encoder with per-sample attention masks
        transformer_outputs = []
        for i in range(batch_size):
            if attention_mask is not None:
                mask = attention_mask[i]  # Shape: (seq_length, seq_length)
                # Ensure mask is on the same device and dtype as embedded
                mask = mask.to(dtype=torch.bool, device=embedded.device)
            else:
                mask = None

            # TransformerEncoder expects input of shape (batch_size, seq_length, d_model) if batch_first=True
            # Here, we process one sample at a time
            output = self.transformer_encoder(
                embedded[i].unsqueeze(0), mask=mask
            )  # Shape: (1, seq_length, d_model)
            transformer_outputs.append(output)

        # Concatenate outputs: (batch_size, seq_length, d_model)
        transformer_output = torch.cat(transformer_outputs, dim=0)

        # Step 4: Apply Word Pooling
        # pooled_word_vectors shape: (total_words, hidden_dim)
        pooled_word_vectors = self.word_pooling(transformer_output, word_boundaries)

        return pooled_word_vectors

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Positional Encoding (RoPE) to the input embeddings.

        Args:
            x (torch.Tensor): Embedded input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: RoPE-enhanced embeddings of shape (batch_size, seq_length, d_model).
        """
        batch_size, seq_length, d_model = x.size()
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum {self.max_seq_length}"
            )

        # Create position indices
        position = torch.arange(
            seq_length, dtype=torch.float, device=x.device
        ).unsqueeze(
            1
        )  # (seq_length, 1)

        # Compute the rotary frequencies
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_model, 2, device=x.device).float() / d_model)
        )  # (d_model/2,)

        # Compute the sinusoidal embeddings
        sinusoid_inp = position * inv_freq  # (seq_length, d_model/2)
        sin = torch.sin(sinusoid_inp)  # (seq_length, d_model/2)
        cos = torch.cos(sinusoid_inp)  # (seq_length, d_model/2)

        # Expand to match batch size
        sin = sin.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (batch_size, seq_length, d_model/2)
        cos = cos.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (batch_size, seq_length, d_model/2)

        # Split the embedding into even and odd parts
        x1 = x[:, :, 0::2]  # (batch_size, seq_length, d_model/2)
        x2 = x[:, :, 1::2]  # (batch_size, seq_length, d_model/2)

        # Apply RoPE
        x_rotated = x1 * cos - x2 * sin  # (batch_size, seq_length, d_model/2)
        x2_new = x1 * sin + x2 * cos  # (batch_size, seq_length, d_model/2)

        # Interleave the rotated components
        x = torch.stack((x_rotated, x2_new), dim=-1).reshape(
            batch_size, seq_length, d_model
        )

        return x


# Example Usage
if __name__ == "__main__":
    # Initialize the tokenizer and pooling module
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    pooling_type = "average"  # Change to 'max' for max pooling

    # Initialize the Transformer Encoder with WordPooling
    model = TransformerEncoderModule(
        vocab_size=256,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=512,
        activation="relu",
        batch_first=True,
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

    batch = []
    for text in [sample_text1, sample_text2]:
        # Tokenize the sample text
        tokens, mask, boundaries = tokenizer.tokenize(text)
        print(f"Token IDs:\n{tokens}\nToken IDs shape: {tokens.shape}")
        print(f"\nAttention Mask:\n{mask}\nMask Shape: {mask.shape}")
        print(f"\nWord Boundaries:\n {boundaries}")

        # Convert tokens and mask to tensors
        # Assuming tokenizer.tokenize returns tokens as a list of lists for batch processing
        # Here, we create a batch size of 1 for demonstration
        tokens_tensor = tokens.to(device)  # Shape: (batch_size, seq_length)
        attention_mask_tensor = mask.to(device)  # Shape: (seq_length, seq_length)
        word_boundaries_list = boundaries

        # Forward pass through the model
        pooled_word_vectors = model(
            tokens_tensor,
            attention_mask=attention_mask_tensor,
            word_boundaries=word_boundaries_list,
        )

        print("\nPooled Word Vectors Shape:", pooled_word_vectors.shape)
        print("\nPooled Word Vectors:\n", pooled_word_vectors)
        batch.append(pooled_word_vectors)

    padder = LearnedPadding(d_model=512).to(device)
    padded_seq, mem_key_mask = padder(batch)
    print(padded_seq.shape)
    # Optionally, decode the tokens to verify correctness
    decoded_text = tokenizer.decode(tokens)
    print("\nDecoded Text:\n", decoded_text)
