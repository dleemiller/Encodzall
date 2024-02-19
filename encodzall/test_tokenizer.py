import unittest
import tokenizer


class TestCTokenizer(unittest.TestCase):

    def test_basic_ascii(self):
        """Test tokenization of a basic ASCII string."""
        tokens, attention_mask, word_starts = tokenizer.tokenize(
            "hello", max_length=10, pad_id=0, end_id=255
        )
        self.assertEqual(tokens, [104, 101, 108, 108, 111, 255, 0, 0, 0, 0])
        self.assertEqual(
            attention_mask,
            [True, True, True, True, True, True, False, False, False, False],
        )
        self.assertEqual(
            word_starts,
            [True, False, False, False, False, False, False, False, False, False],
        )

    def test_unicode_characters(self):
        """Test tokenization of a string with Unicode characters."""
        tokens, attention_mask, word_starts = tokenizer.tokenize(
            "héllo 🌎", max_length=12, pad_id=0, end_id=255
        )
        expected_tokens = [104, 195, 169, 108, 108, 111, 32, 240, 159, 140, 142, 255]
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(attention_mask, [True] * 11 + [True])
        self.assertEqual(
            word_starts,
            [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        )

    def test_no_padding_or_truncation(self):
        """Test tokenization without padding or truncation."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("test", end_id=255)
        self.assertEqual(tokens, [116, 101, 115, 116, 255])
        self.assertEqual(attention_mask, [True, True, True, True, True])
        self.assertEqual(word_starts, [True, False, False, False, False])

    def test_truncation(self):
        """Test truncation when the input string is longer than max_length."""
        tokens, attention_mask, word_starts = tokenizer.tokenize(
            "hello world", max_length=8, pad_id=0, end_id=255
        )
        self.assertEqual(tokens, [104, 101, 108, 108, 111, 32, 119, 255])
        self.assertEqual(attention_mask, [True] * 8)
        self.assertEqual(
            word_starts, [True, False, False, False, False, False, True, False]
        )

    def test_padding(self):
        """Test padding when the input string is shorter than max_length."""
        tokens, attention_mask, word_starts = tokenizer.tokenize(
            "pad", max_length=6, pad_id=0, end_id=255
        )
        self.assertEqual(tokens, [112, 97, 100, 255, 0, 0])
        self.assertEqual(attention_mask, [True, True, True, True, False, False])
        self.assertEqual(word_starts, [True, False, False, False, False, False])

    def test_basic_segmentation(self):
        """Test basic functionality with a simple input."""
        tokens = [
            104,
            101,
            108,
            108,
            111,
            32,
            119,
            111,
            114,
            108,
            100,
            255,
        ]  # 'hello world'
        attention_mask = [True] * 11 + [False]  # False for padding/end_id
        word_starts = [
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        max_length = 6
        max_words = 3
        pad_id = 0
        end_id = 255

        words, word_attention = tokenizer.segment_words(
            tokens, attention_mask, word_starts, max_length, max_words, pad_id, end_id
        )

        expected_words = [
            [104, 101, 108, 108, 111, 32],
            [119, 111, 114, 108, 100, 255],
            [0, 0, 0, 0, 0, 0],
        ]
        expected_attention = [True, True, False]  # Last list is all padding
        self.assertEqual(words, expected_words)
        self.assertEqual(word_attention, expected_attention)

    def test_unicode_handling(self):
        """Test handling of unicode characters correctly segmented into words."""
        # Assuming the token list is already properly encoded to handle Unicode
        tokens = [
            104,
            195,
            169,
            108,
            108,
            111,
            32,
            240,
            159,
            140,
            142,
            255,
        ]  # 'héllo 🌎'
        attention_mask = [True] * 11 + [False]
        word_starts = [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
        max_length = 7
        max_words = 2
        pad_id = 0
        end_id = 255

        words, word_attention = tokenizer.segment_words(
            tokens, attention_mask, word_starts, max_length, max_words, pad_id, end_id
        )

        expected_words = [
            [104, 195, 169, 108, 108, 111, 32],
            [240, 159, 140, 142, 255, 255, 0],
        ]
        expected_attention = [True, True]

        self.assertEqual(words, expected_words)
        self.assertEqual(word_attention, expected_attention)

    def test_padding_and_truncation(self):
        """Test correct padding and truncation of words and outer list."""
        tokens = [104, 101, 32, 119, 111, 114, 108, 100, 255]  # 'he world'
        attention_mask = [True] * 8 + [False]
        word_starts = [True, False, False, True, False, False, False, False, False]
        max_length = 4  # Truncate/pad each word to 4 tokens
        max_words = 4  # Pad outer list to 4 words
        pad_id = 0
        end_id = 255

        words, word_attention = tokenizer.segment_words(
            tokens, attention_mask, word_starts, max_length, max_words, pad_id, end_id
        )

        expected_words = [
            [104, 101, 32, 255],
            [119, 111, 114, 108],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]  # Note padding and truncation
        expected_attention = [True, True, False, False]

        self.assertEqual(words, expected_words)
        self.assertEqual(word_attention, expected_attention)


if __name__ == "__main__":
    unittest.main()
