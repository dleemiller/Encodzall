from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    output_dir: str
    max_sequence_length: int
    dataset_split: str
    dataset_name: str
    dataset_subset: str
    seed: int = 42
    target_loss: float = 1.0
    pid_Kp: float = 1.0
    pid_Ki: float = 0.1
    pid_Kd: float = 0.05
    prob_initial: float = 0.0
    prob_min: float = 0.0
    prob_max: float = 1.0
    mask_ratio: float = 0.2
    noise_ratio: float = 0.8
    checkpoint_interval: int = 10000
