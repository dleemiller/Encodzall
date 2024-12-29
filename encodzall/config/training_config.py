# training_config.py
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    output_dir: str
    max_sequence_length: int
    dataset_split: str = "train[0:100000]"
    dataset_name: str = "wikimedia/wikipedia"
    dataset_language: str = "20231101.en"
    seed: int = 42
