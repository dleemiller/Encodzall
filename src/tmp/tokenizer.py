import torch
from typing import List, Tuple
import more_itertools
import itertools
import random
from string_noise import noise

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


def random_contiguous_sublist(lst, length):
    """
    Selects a random contiguous sublist of a fixed length from a given list.

    Args:
        lst (list): The input list to sample from.
        length (int): The fixed length of the sublist.

    Returns:
        list: A random contiguous sublist of the specified length.
    """
    if len(lst) < length:
        raise ValueError("The list is shorter than the desired sublist length.")

    # Calculate the valid starting index range
    max_start_index = len(lst) - length
    start_index = random.randint(0, max_start_index)

    # Slice the list
    return lst[start_index : start_index + length]


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
        self,
        words: List[str],
        target_min_len: int = 8,
        word_break=16,
    ) -> List[List[int]]:
        """
        Encode each word into its corresponding byte values (0-255).
        """
        byte_sequences = list(map(lambda x: list(bytearray(x.encode("utf-8"))), words))

        # break long sequences
        byte_sequences = list(
            more_itertools.flatten(
                [more_itertools.chunked_even(x, word_break) for x in byte_sequences]
            )
        )

        # collect short word sequences
        byte_sequences = list(
            more_itertools.constrained_batches(
                byte_sequences, max_size=target_min_len, get_len=len, strict=False
            )
        )
        byte_sequences = [list(more_itertools.flatten(x)) for x in byte_sequences]
        return byte_sequences

    def pack_sequences(
        self,
        byte_sequences: List[List[int]],
        mask_prob: float = 0.0,
        add_eos: bool = False,
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
                strict=True,
            )
        )

        for batch in batches:
            packed = [[MASK_BYTE] if random.random() < mask_prob else x for x in batch]
            word_lengths = list(itertools.accumulate([0] + list(map(len, packed))))
            word_boundaries = [
                (word_lengths[i], word_lengths[i + 1]) for i in range(len(packed))
            ]
            packed = list(more_itertools.flatten(packed))

            if add_eos:
                if self.max_sequence_length <= len(packed):
                    packed.pop(-1)
                packed.append(EOS_BYTE)
                word_boundaries.append((len(packed) - 1, len(packed)))

            # If packed sequence is shorter than max_sequence_length, pad it
            padding_length = self.max_sequence_length - len(packed)
            if padding_length > 0:
                packed.extend([PAD_BYTE] * padding_length)

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

    def tokenize(
        self,
        text: str,
        add_eos: bool = True,
        truncate_len: int = 256,
        char_len: int = 2048,
        random_window: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the input text into byte token IDs and create the corresponding attention mask.
        """
        words = self.split_text(text)
        words = list(more_itertools.constrained_batches(words, char_len))[0]
        if len(words) > truncate_len:
            if random_window:
                words = random_contiguous_sublist(words, truncate_len)
            else:
                words = words[0:truncate_len]

        noised_words = list(
            map(lambda x: noise.ocr(x, probability=random.uniform(0.2, 0.6)), words)
        )

        # Step 2: Encode each word into byte sequences
        byte_sequences = self.encode_words(noised_words)

        # Step 3: Pack byte sequences into fixed-length sequences
        packed_sequences, word_boundaries_list = self.pack_sequences(
            byte_sequences, mask_prob=0.1, add_eos=True
        )

        # Step 4: Convert packed sequences to tensor
        token_tensor = torch.tensor(packed_sequences, dtype=torch.uint8)

        # Step 5: Create attention masks
        attention_mask = self.create_attention_mask(word_boundaries_list)

        byte_sequences = self.encode_words(words)
        target_ids = self.create_targets(byte_sequences)
        return token_tensor, attention_mask, word_boundaries_list, target_ids

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

    def create_targets(self, byte_sequences: list[int]) -> torch.Tensor:
        target_ids = [BOS_BYTE] + list(more_itertools.flatten(byte_sequences))
        return target_ids

    def pad_targets(self, target_ids: list[list[int]]):
        nested_tensor = torch.nested.nested_tensor(
            [torch.tensor(t, dtype=torch.uint8) for t in target_ids]
        )
        padded_targets = nested_tensor.to_padded_tensor(padding=PAD_BYTE)
        key_padding_mask = padded_targets == PAD_BYTE
        return padded_targets, key_padding_mask


# Example usage
if __name__ == "__main__":
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    # sample_text = "Hello world! This is a sample text to encode. Tokenize the input text into byte token IDs and create the corresponding attention mask."
    sample_text = "Hello world! I am a cat!"
    tokens, mask, boundaries, targets = tokenizer.tokenize(sample_text)
    print("Token IDs:\n", tokens)
    print("Attention Mask:\n", mask)
    print("Target IDs:\n", targets)
    # import matplotlib.pyplot as plt
    # plt.imshow(mask[0])
    # plt.show()
    decoded_text = tokenizer.decode(tokens)
    print("Decoded Text:\n", decoded_text)
