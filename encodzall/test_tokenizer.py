import unittest
import tokenizer

class TestCTokenizer(unittest.TestCase):

    def test_basic_ascii(self):
        """Test tokenization of a basic ASCII string."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("hello", max_length=10, pad_id=0, end_id=255)
        self.assertEqual(tokens, [104, 101, 108, 108, 111, 255, 0, 0, 0, 0])
        self.assertEqual(attention_mask, [True, True, True, True, True, True, False, False, False, False])
        self.assertEqual(word_starts, [True, False, False, False, False, False, False, False, False, False])

    def test_unicode_characters(self):
        """Test tokenization of a string with Unicode characters."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("héllo 🌎", max_length=12, pad_id=0, end_id=255)
        expected_tokens = [104, 195, 169, 108, 108, 111, 32, 240, 159, 140, 142, 255]
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(attention_mask, [True] * 11 + [True])
        self.assertEqual(word_starts, [True, False, False, False, False, False, False, True, False, False, False, False])

    def test_no_padding_or_truncation(self):
        """Test tokenization without padding or truncation."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("test", end_id=255)
        self.assertEqual(tokens, [116, 101, 115, 116, 255])
        self.assertEqual(attention_mask, [True, True, True, True, True])
        self.assertEqual(word_starts, [True, False, False, False, False])

    def test_truncation(self):
        """Test truncation when the input string is longer than max_length."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("hello world", max_length=8, pad_id=0, end_id=255)
        self.assertEqual(tokens, [104, 101, 108, 108, 111, 32, 119, 255])
        self.assertEqual(attention_mask, [True] * 8)
        self.assertEqual(word_starts, [True, False, False, False, False, False, True, False])

    def test_padding(self):
        """Test padding when the input string is shorter than max_length."""
        tokens, attention_mask, word_starts = tokenizer.tokenize("pad", max_length=6, pad_id=0, end_id=255)
        self.assertEqual(tokens, [112, 97, 100, 255, 0, 0])
        self.assertEqual(attention_mask, [True, True, True, True, False, False])
        self.assertEqual(word_starts, [True, False, False, False, False, False])

if __name__ == "__main__":
    unittest.main()

