import torch
import random
import re
import itertools
import more_itertools
from string_noise import noise

from .config.noise_config import NoiseConfig


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

# noise.moe("load")


class ByteLevelTokenizer:
    """
    A character-level tokenizer that packs text sequences into fixed-length blocks,
    applies optional noise (string_noise), and creates a block diagonal attention mask
    (per-word mask). Special tokens are used for <BOS>, <EOS>, <PAD>, <MASK>, <SEP>.

    Attributes:
        max_sequence_length (int): The maximum length (in bytes) of a packed sequence.
                                   Sequences are padded if they are shorter.
        special_bytes (dict): Mapping of special byte values to their string tokens.
        byte_to_token (dict): A direct mapping from each byte (0-255) to itself (identity).
                              Overridden by special tokens.
        token_to_byte (dict): Reverse mapping for decoding from special tokens to byte values.
    """

    def __init__(self, max_sequence_length: int = 64, noise_config: NoiseConfig = None):
        self.max_sequence_length = max_sequence_length
        self.noise_config = noise_config if noise_config else NoiseConfig()

        # Setup special tokens
        self.special_bytes: dict[int, str] = SPECIAL_BYTES
        self.byte_to_token: dict[int, str] = {byte: chr(byte) for byte in range(256)}
        # Override mapping for special bytes
        for byte, token in self.special_bytes.items():
            self.byte_to_token[byte] = token

        # Reverse mapping (string -> byte) for special tokens
        self.token_to_byte = {token: byte for byte, token in self.special_bytes.items()}

    def split_text(self, text: str) -> list[str]:
        """
        Split text into tokens where each token preserves trailing whitespace.

        Args:
            text (str): Input string.

        Returns:
            list[str]: list of tokens with preserved trailing whitespace.
        """
        return re.findall(r"[\s]*\S+[\s\-]*", text)

    def apply_noise_to_words(
        self, words: list[str], noise_prob: float = 0.0
    ) -> list[str]:
        """
        Optionally apply noise to each word using string_noise OCR simulation.
        Probability for each word is randomly chosen between [min_p, max_p].
        """
        # use several kinds of character level noising
        # noise.moe - mispelling (from Mispelling Oblivious Embeddings)  # needs fixed!
        # noise.ocr - simulated ocr character corruption
        # noise.keyboard - nearby keystrokes
        # noise.mask - masking with distinct masks consonants, non-whitespace (punc), vowels and either (general)
        # noise.mask - 0x06-0x0B
        #noise_func = random.choice([noise.ocr, noise.moe, noise.keyboard])
        #noise_func = random.choice([noise.mask, noise.moe, noise.ocr, noise.keyboard])
        noise_func = noise.mask
        return [noise_func(word, probability=noise_prob) for word in words]

    def encode_words(
        self, words: list[str], target_min_len: int = 12, word_break: int = 12
    ) -> list[list[int]]:
        """
        Encode each word into its corresponding byte values (0-255).
        Then breaks long words (in bytes) into smaller chunks of length `word_break`.
        Finally, groups them into batches of size >= target_min_len if possible.

        Args:
            words (list[str]): list of string tokens/words.
            target_min_len (int): Attempt to ensure that each batch (set of chunks)
                                  has at least `target_min_len` bytes.
            word_break (int): Breaks each word's byte-array into smaller chunks of this length.

        Returns:
            list[list[int]]: A list of integer lists, where each sub-list is a chunk of bytes.
        """
        # Convert each word to bytes
        byte_sequences = [list(word.encode("utf-8")) for word in words]

        # Break long sequences into chunks of size `word_break`
        chunked_sequences = []
        for seq in byte_sequences:
            chunked_sequences.extend(more_itertools.chunked_even(seq, word_break))

        # Collect short word sequences into groups of at least `target_min_len` bytes
        constrained = list(
            more_itertools.constrained_batches(
                chunked_sequences, max_size=target_min_len, get_len=len, strict=False
            )
        )
        # Flatten each group
        final_sequences = [list(more_itertools.flatten(group)) for group in constrained]
        return final_sequences

    def pack_sequences(
        self,
        byte_sequences: list[list[int]],
        mask_prob: float = 0.0,
        add_eos: bool = False,
    ) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
        """
        Pack multiple byte sequences into fixed-length blocks (self.max_sequence_length).
        Creates word-boundary tuples for constructing block diagonal attention.

        Args:
            byte_sequences (list[list[int]]): list of byte chunks (words or word-chunks).
            mask_prob (float): Probability that a chunk gets replaced by a single [MASK] token.
            add_eos (bool): If True, append [EOS] at the end of each packed sequence (if space).

        Returns:
            tuple[
                list[list[int]],        # packed sequences (list of integer IDs per block)
                list[list[tuple[int, int]]]  # word boundaries per block
            ]
        """
        packed_sequences = []
        word_boundaries_list = []

        # Constrain the chunks so that each batch doesn't exceed self.max_sequence_length
        # NOTE: setting strict=True will omit any leftover if it doesn't fit exactly,
        # which might be desirable or not. We'll keep it True for uniform blocks.
        if add_eos:
            byte_sequences.append([EOS_BYTE])
        chunk_batches = list(
            more_itertools.constrained_batches(
                byte_sequences,
                max_size=self.max_sequence_length,
                get_len=len,
                strict=True,
            )
        )

        for chunk_batch in chunk_batches:
            # Possibly replace entire chunk with a single [MASK] token
            augmented_batch = []
            for chunk in chunk_batch:
                if random.random() < mask_prob:
                    augmented_batch.append([MASK_BYTE])
                else:
                    augmented_batch.append(chunk)

            # Word boundaries: e.g., if chunk1 is length 5, chunk2 is length 3 => boundaries: [(0,5),(5,8)]
            word_lengths = list(
                itertools.accumulate([0] + [len(c) for c in augmented_batch])
            )
            word_boundaries = [
                (word_lengths[i], word_lengths[i + 1])
                for i in range(len(augmented_batch))
            ]

            # Flatten the chunks
            flattened = list(more_itertools.flatten(augmented_batch))

            # Pad if needed
            if len(flattened) < self.max_sequence_length:
                padding_needed = self.max_sequence_length - len(flattened)
                flattened.extend([PAD_BYTE] * padding_needed)

            packed_sequences.append(flattened)
            word_boundaries_list.append(word_boundaries)

        return packed_sequences, word_boundaries_list

    def create_attention_mask(
        self,
        word_boundaries_list: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """
        Create a block-diagonal attention mask for each packed sequence.
        Each block corresponds to the positions of a single chunk/word within the sequence.

        Args:
            word_boundaries_list (list[list[tuple[int, int]]]):
                For each sequence in the batch, a list of (start, end) for each chunk.

        Returns:
            attention_mask (torch.Tensor):
                A boolean tensor of shape (batch_size, max_sequence_length, max_sequence_length).
        """
        batch_size = len(word_boundaries_list)
        sequence_length = self.max_sequence_length
        attention_mask = torch.zeros(
            (batch_size, sequence_length, sequence_length), dtype=torch.bool
        )

        for i, boundaries in enumerate(word_boundaries_list):
            for start, end in boundaries:
                if 0 <= start < end <= sequence_length:
                    attention_mask[i, start:end, start:end] = True

        return attention_mask

    def create_targets(self, byte_sequences: list[list[int]]) -> list[list[int]]:
        """
        Create target labels by prepending a [BOS] token, and ignoring any augmentation.
        For training, we want the original (non-augmented) sequences.

        Args:
            byte_sequences (list[list[int]]): Each sub-list is a chunk of bytes from the original text.

        Returns:
            list[list[int]]: For each sub-list, a new list with [BOS] + original sequence.
        """
        return list(more_itertools.flatten(byte_sequences))

    def pad_targets(
        self, target_ids: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list-of-lists of target sequences into a padded tensor,
        returning both the padded tensor and a key_padding_mask.

        Args:
            target_ids (list[list[int]]): Each sub-list is a sequence of integer IDs.

        Returns:
            padded_targets (torch.Tensor): shape (batch_size, max_length)
            key_padding_mask (torch.Tensor): boolean mask for padding.
        """
        nested_tensor = torch.nested.nested_tensor(
            [torch.tensor(t, dtype=torch.uint8) for t in target_ids]
        )
        padded_targets = nested_tensor.to_padded_tensor(padding=PAD_BYTE)

        # Create a tensor filled with EOS_BYTE of shape (batch_size, 1)
        eos_column = torch.full(
            (padded_targets.size(0), 1),
            EOS_BYTE,
            dtype=torch.uint8,
        )

        # Concatenate the EOS column to the original tensor along dimension 1 (sequence length)
        padded_targets = torch.cat((eos_column, padded_targets), dim=1)

        key_padding_mask = padded_targets == PAD_BYTE
        return padded_targets, key_padding_mask

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decodes token IDs (bytes) to the original text (latin1).
        Skips any recognized special token IDs.

        Args:
            token_ids (torch.Tensor): shape (...), a tensor of byte values.

        Returns:
            str: Decoded text as a string.
        """
        decoded_bytes = []
        for tid in token_ids.flatten().tolist():
            if tid in self.special_bytes:
                # skip special tokens
                continue
            decoded_bytes.append(tid)
        return bytes(decoded_bytes).decode("latin1")

    def tokenize(
        self,
        text: str,
        add_eos: bool = True,
        truncate_len: int = 320,
        char_len: int = 2048,
        mask_prob: bool = 0.0,
        noise_prob: float | None = None,
        return_byte_seq: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, list[list[tuple[int, int]]], list[list[int]]
    ]:
        """
        High-level method that:
            1) Splits the text,
            2) Optionally truncates using a random window,
            3) Applies optional noise,
            4) Encodes,
            5) Packs sequences,
            6) Creates attention mask,
            7) Prepares final targets (non-augmented).

        Args:
            text (str): Input string.
            add_eos (bool): Whether to add [EOS] in each packed sequence.
            truncate_len (int): If # of tokens > truncate_len, keep that many (possibly random slice).
            char_len (int): Hard cap for the text tokens if the text is extremely large.
            random_window (bool): If True, a random contiguous sublist of tokens is chosen.
            apply_noise (bool): If True, apply noise to the tokens (for training).

        Returns:
            token_tensor (torch.Tensor): shape (batch_size, max_sequence_length)
            attention_mask (torch.Tensor): shape (batch_size, max_sequence_length, max_sequence_length)
            word_boundaries_list (list[list[tuple[int, int]]]): boundaries used to create the mask
            targets (list[list[int]]): Non-augmented target sequences, each with leading [BOS].
        """
        # 1) Split text into tokens/words
        words = self.split_text(text)

        # limit characters
        encoded_lengths = [len(word.encode("utf-8")) for word in words]
        word_len = list(itertools.accumulate(encoded_lengths))

        max_idx = max([i for i, x in enumerate(word_len) if x < char_len])
        words = words[0 : max_idx + 1]

        # 3) Possibly truncate further by random window or just head cut
        if len(words) > truncate_len:
            words = words[:truncate_len]

        # 4) Optionally apply noise
        if noise_prob == -1: # override noise setting
            noised_words = words.copy()
        elif noise_prob or self.noise_config.noise_prob > 0:
            noise_prob = noise_prob or self.noise_config.noise_prob
            assert 0 <= noise_prob <= 1
            noised_words = self.apply_noise_to_words(words, noise_prob=noise_prob)
        else:
            noised_words = words.copy()

        # 5) Encode words (noised for the actual input)
        byte_sequences_noised = self.encode_words(noised_words)
        if return_byte_seq:
            return byte_sequences_noised

        # 6) Pack these sequences
        packed_sequences, word_boundaries_list = self.pack_sequences(
            byte_sequences_noised,
            mask_prob=mask_prob or self.noise_config.mask_prob,
            add_eos=add_eos,
        )

        # Convert to Torch
        token_tensor = torch.tensor(packed_sequences, dtype=torch.uint8)

        # 7) Build attention mask
        attention_mask = self.create_attention_mask(word_boundaries_list)

        # 8) Create targets (non-noised) & then return them raw (or can be padded later)
        #    We re-encode the original words to get the full non-noised bytes:
        targets = self.encode_words(words)
        # targets = self.create_targets(byte_sequences_original)

        # seq_target_ids = torch.nested.nested_tensor(
        #     list(more_itertools.flatten(x)) for x in byte_sequences_original
        # )
        # seq_target_ids
        # word_target_ids = torch.cat(
        #     list(torch.nested.nested_tensor(x) for x in byte_sequences_original)
        # )

        return (token_tensor, attention_mask, word_boundaries_list, targets)

    def get_special_tokens(self) -> dict[int, str]:
        """
        Return the special tokens dictionary.
        """
        return self.special_bytes.copy()


# Example usage
if __name__ == "__main__":
    tokenizer = ByteLevelTokenizer(max_sequence_length=64)
    # sample_text = "Hello world! This is a sample text to encode. Tokenize the input text into byte token IDs and create the corresponding attention mask."
    sample_text = "I Hello world! I am a cat!"
    tokens, mask, boundaries, seq_targets, word_targets = tokenizer.tokenize(
        sample_text, mask_prob=0.0, noise_prob=0.3
    )
    print("Token IDs:\n", tokens)
    print("Attention Mask:\n", mask)
    print("Target IDs:\n", seq_targets, "\nWord Target IDs:\n", word_targets)
    # import matplotlib.pyplot as plt
    # plt.imshow(mask[0])
    # plt.show()
    decoded_text = tokenizer.decode(tokens)
    print("Decoded Text:\n", decoded_text)
