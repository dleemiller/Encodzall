import unittest
import torch
from encodzall.utils.gather_vectors import gather_word_starts


class TestGatherWordStarts(unittest.TestCase):
    def setUp(self):
        self.pad_id = 0

    def test_uniform_word_starts(self):
        """Test with uniform distribution of word starts."""
        data = torch.arange(24).view(2, 3, 4).float()  # Example data tensor
        word_starts = torch.tensor(
            [[True, False, True], [True, True, False]], dtype=torch.bool
        )
        gathered, mask = gather_word_starts(data, word_starts, self.pad_id)

        expected_gathered = torch.tensor(
            [[[0, 1, 2, 3], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19]]]
        ).float()
        expected_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

        torch.testing.assert_close(gathered, expected_gathered)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_varying_word_starts(self):
        """Test with varying number of word starts across batches."""
        data = torch.arange(40).view(2, 5, 4).float()  # Larger data tensor
        word_starts = torch.tensor(
            [[True, False, True, False, True], [True, True, False, False, False]],
            dtype=torch.bool,
        )
        gathered, mask = gather_word_starts(data, word_starts, self.pad_id, max_words=3)

        expected_gathered = torch.tensor(
            [
                [[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]],
                [[20, 21, 22, 23], [24, 25, 26, 27], [0, 0, 0, 0]],
            ]
        ).float()
        expected_mask = torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        )

        torch.testing.assert_close(gathered, expected_gathered)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_no_word_starts(self):
        """Test with no word starts."""
        data = torch.arange(12).view(1, 3, 4).float()  # Single batch data tensor
        word_starts = torch.tensor([[False, False, False]], dtype=torch.bool)
        gathered, mask = gather_word_starts(data, word_starts, self.pad_id)

        expected_gathered = torch.zeros(
            1, 0, 4
        ).float()  # Expect a single row of padding
        expected_mask = torch.tensor([[]], dtype=torch.bool)

        torch.testing.assert_close(gathered, expected_gathered)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_max_words_exceeds_true_counts(self):
        """Test when max_words is larger than the number of True word starts."""
        data = torch.arange(24).view(2, 3, 4).float()
        word_starts = torch.tensor(
            [[True, False, True], [True, True, False]], dtype=torch.bool
        )
        gathered, mask = gather_word_starts(data, word_starts, self.pad_id, max_words=5)

        # Expect padding to fill up to max_words
        expected_gathered = torch.tensor(
            [
                [
                    [0, 1, 2, 3],
                    [8, 9, 10, 11],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        ).float()
        expected_mask = torch.tensor(
            [[True, True, False, False, False], [True, True, False, False, False]],
            dtype=torch.bool,
        )

        torch.testing.assert_close(gathered, expected_gathered)
        self.assertTrue(torch.equal(mask, expected_mask))


if __name__ == "__main__":
    unittest.main()
