import torch
from torch import nn
import more_itertools
from typing import List, Tuple

from tokenizer import ByteLevelTokenizer


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
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        word_boundaries: list[tuple[int]],
    ) -> torch.Tensor:
        """
        Performs pooling over word vectors based on the attention mask.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_len, seq_len) with boolean values.

        Returns:
            torch.Tensor: Tensor of shape (total_words, hidden_dim) containing pooled word vectors.
        """
        pooled_words = []

        for batch_idx, boundaries in enumerate(word_boundaries):
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
                0, hidden_states.size(-1)
            )  # Return empty tensor if no words


if __name__ == "__main__":
    # Initialize the tokenizer and pooling module
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    pooling_module = WordPooling(
        pooling_type="average"
    )  # Change to 'max' for max pooling

    # Sample text
    sample_text = "Hello world! This is a sample text to encode. Tokenize the input text into byte token IDs and create the corresponding attention mask."

    # Tokenize the sample text
    tokens, mask, boundaries = tokenizer.tokenize(sample_text)
    print("Token IDs:\n", tokens)
    print("\nAttention Mask:\n", mask)

    # Simulate hidden states (e.g., from Encoder 1)
    # For demonstration, we'll use random vectors
    hidden_dim = 16
    hidden_states = torch.randn(tokens.size(0), tokens.size(1), hidden_dim)

    # Perform pooling
    pooled_word_vectors = pooling_module(
        hidden_states.to("cuda"), mask.to("cuda"), boundaries
    )
    print("\nPooled Word Vectors Shape:", pooled_word_vectors.shape)
    print("\nPooled Word Vectors:\n", pooled_word_vectors)

    # Decode the tokens to verify correctness
    decoded_text = tokenizer.decode(tokens)
    print("\nDecoded Text:\n", decoded_text)
