import torch

from encodzall.tokenizer import tokenize, DEFAULT_PAD_ID, DEFAULT_END_ID


class Tokenizer:
    _instance = None
    _config = None

    @property
    def max_length(self):
        return self._config.max_seq_length

    @property
    def max_words(self):
        return self._config.max_words

    @property
    def max_word_length(self):
        return self._config.max_word_length

    @property
    def pad_id(self):
        return self._config.pad_id

    @property
    def end_id(self):
        return self._config.end_id

    def tokenize(self, text: str):
        input_ids, attention_mask, word_start = tokenize(
            text, max_length=self.max_length, pad_id=self.pad_id, end_id=self.end_id
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.uint8),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "word_start": torch.tensor(word_start, dtype=torch.bool),
        }

    def targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_start: torch.Tensor,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        words, word_attention = tokenizer.make_targets(
            input_ids.tolist(),
            attention_mask.tolist(),
            word_start.tolist(),
            self.max_word_length,
            self.max_words,
            self.pad_id,
            self.end_id,
        )
        return {
            "target_ids": torch.tensor(words, dtype=torch.uint8),
            "target_mask": torch.tensor(word_attention, dtype=torch.bool),
        }

    @classmethod
    def init(cls, config=None):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            assert config is not None
            cls._config = config.tokenizer
        return cls._instance
