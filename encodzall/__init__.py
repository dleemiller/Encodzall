from dataclasses import dataclass
import torch

from encodzall.tokenizer import tokenize, DEFAULT_PAD_ID, DEFAULT_END_ID


@dataclass
class DefaultTokenizerConfig:
    max_length: int = 2048
    max_words: int = 512
    pad_id: int = DEFAULT_PAD_ID
    end_id: int = DEFAULT_END_ID


class Tokenizer:
    _instance = None
    _config = None

    @property
    def max_length(self):
        return self._config.max_length

    @property
    def max_words(self):
        return self._config.max_words

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

    @classmethod
    def init(cls, config=None):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            assert config is not None
            cls._config = config
        return cls._instance
