import torch
from typing import List, Tuple
import more_itertools
import itertools

# Define special tokens using C1 control characters (bytes 128-159)
PAD_BYTE = 0x80  # <PAD>
MASK_BYTE = 0x81  # <MASK>
EOS_BYTE = 0x82  # <EOS>
BOS_BYTE = 0x83  # <BOS>
SEP_BYTE = 0x84  # <SEP>

SPECIAL_BYTES = {
    PAD_BYTE: "<PAD>",
    MASK_BYTE: "<MASK>",
    EOS_BYTE: "<EOS>",
    BOS_BYTE: "<BOS>",
    SEP_BYTE: "<SEP>",
}

# Reverse mapping for decoding
BYTE_TO_SPECIAL_TOKEN = {v: k for k, v in SPECIAL_BYTES.items()}
SPECIAL_TOKEN_SET = set(SPECIAL_BYTES.values())


class ByteLevelTokenizer:
    def __init__(self, max_sequence_length: int = 64):
        self.max_sequence_length = max_sequence_length
        self.special_bytes = SPECIAL_BYTES
        self.byte_to_token = {byte: byte for byte in range(256)}  # Direct mapping
        # Override special bytes with their corresponding special tokens
        for byte, token in self.special_bytes.items():
            self.byte_to_token[byte] = token
        # Create reverse mapping for decoding
        self.token_to_byte = {token: byte for byte, token in self.special_bytes.items()}

    def split_text(self, text: str) -> List[str]:
        """
        Split text into words, retaining the whitespace at the end of each word.
        Example:
            "Hello world! " -> ["Hello ", "world! "]
        """
        return list(
            map("".join, more_itertools.split_after(text, lambda s: s.isspace()))
        )

    def encode_words(
        self, words: List[str], target_min_len: int = 8
    ) -> List[List[int]]:
        """
        Encode each word into its corresponding byte values (0-255).
        """
        byte_sequences = []
        for word in words:
            byte_seq = list(bytearray(word.encode("utf-8")))
            byte_sequences.append(byte_seq)

        byte_sequences = list(
            more_itertools.constrained_batches(
                byte_sequences, max_size=target_min_len, get_len=len, strict=False
            )
        )
        return [list(more_itertools.flatten(x)) for x in byte_sequences][0:512]

    def pack_sequences(
        self, byte_sequences: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
        """
        Pack multiple word byte sequences into fixed-length byte sequences (max_sequence_length).
        Returns a list of packed byte sequences and corresponding word boundaries.
        """
        packed_sequences = []
        word_boundaries_list = []

        # Use constrained_batches to pack words into sequences without exceeding max_sequence_length
        batches = list(
            more_itertools.constrained_batches(
                byte_sequences,
                max_size=self.max_sequence_length,
                get_len=len,
                strict=False,
            )
        )

        for batch in batches:
            packed = []
            word_boundaries = []
            current_position = 0
            for word in batch:
                if current_position + len(word) > self.max_sequence_length:
                    break  # Skip words that don't fit
                start = current_position
                end = start + len(word)
                packed.extend(word)
                word_boundaries.append((start, end))
                current_position = end
            # If packed sequence is shorter than max_sequence_length, pad it
            padding_length = self.max_sequence_length - len(packed)
            if padding_length > 0:
                packed += [PAD_BYTE] * padding_length
            packed_sequences.append(packed)
            word_boundaries_list.append(word_boundaries)

        return packed_sequences, word_boundaries_list

    def create_attention_mask(
        self, word_boundaries_list: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Create a block diagonal attention mask for each packed sequence.
        Each block corresponds to a word within the sequence.
        """
        batch_size = len(word_boundaries_list)
        sequence_length = self.max_sequence_length
        mask = torch.zeros(
            (batch_size, sequence_length, sequence_length), dtype=torch.bool
        )

        for i, word_boundaries in enumerate(word_boundaries_list):
            for start, end in word_boundaries:
                # Ensure boundaries are within sequence_length
                start = max(0, min(start, sequence_length))
                end = max(0, min(end, sequence_length))
                if start >= end:
                    continue  # Skip invalid boundaries
                mask[i, start:end, start:end] = True

        return mask  # Shape: (batch_size, sequence_length, sequence_length)

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the input text into byte token IDs and create the corresponding attention mask.
        """
        # Step 1: Split text into words, retaining whitespace
        words = self.split_text(text)
        # Step 2: Encode each word into byte sequences
        byte_sequences = self.encode_words(words)
        # Step 3: Pack byte sequences into fixed-length sequences
        packed_sequences, word_boundaries_list = self.pack_sequences(byte_sequences)
        # Step 4: Convert packed sequences to tensor
        token_tensor = torch.tensor(packed_sequences, dtype=torch.uint8)

        # Step 5: Create attention masks
        attention_mask = self.create_attention_mask(word_boundaries_list)
        return token_tensor, attention_mask, word_boundaries_list

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to the original text.
        Excludes special tokens (PAD, MASK, etc.).
        """
        decoded_bytes = []
        for token_id in token_ids.flatten().tolist():
            if token_id in self.special_bytes:
                continue  # Skip special tokens
            decoded_bytes.append(token_id)
        bytes_seq = bytes(decoded_bytes)
        return bytes_seq.decode("latin1")

    def get_special_tokens(self) -> dict:
        """
        Retrieve the mapping of special tokens.
        """
        return self.special_bytes

    def create_targets(self, texts: list[str]) -> torch.Tensor:
        target_ids = []
        for t in texts:
            words = self.split_text(t)
            ids = list(more_itertools.flatten(self.encode_words(words)))
            target_ids.append(ids)
        nested_tensor = torch.nested.nested_tensor(
            [torch.tensor([BOS_BYTE] + t, dtype=torch.uint8) for t in target_ids]
        )
        padded_targets = nested_tensor.to_padded_tensor(padding=PAD_BYTE)
        key_padding_mask = padded_targets == PAD_BYTE
        return padded_targets, key_padding_mask


# Example usage
if __name__ == "__main__":
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    sample_text = "Hello world! This is a sample text to encode. Tokenize the input text into byte token IDs and create the corresponding attention mask."
    tokens, mask, boundaries = tokenizer.tokenize(sample_text)
    print("Token IDs:\n", tokens)
    print("Attention Mask:\n", mask)
    # import matplotlib.pyplot as plt
    # plt.imshow(mask[0])
    # plt.show()
    decoded_text = tokenizer.decode(tokens)
    print("Decoded Text:\n", decoded_text)
