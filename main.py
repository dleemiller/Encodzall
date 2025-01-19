import os
import argparse
from datetime import datetime
from datasets import load_dataset, DownloadConfig
import torch
from torch.utils.tensorboard import SummaryWriter

from encodzall import encodzall_xs, Encodzall, ByteLevelTokenizer
from encodzall.config.training_config import TrainingConfig
from encodzall.config.noise_config import NoiseConfig
from encodzall.training.utils import set_seed
from encodzall.training.trainer2 import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Encodzall model with PID-controlled noise and contrastive loss"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="Load checkpoint weights only",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save TensorBoard logs",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/datasets/",
        help="Directory to cache datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()

def get_training_config(output_dir: str) -> TrainingConfig:
    """Initialize training configuration."""
    return TrainingConfig(
        num_epochs=1,  # Adjust as needed
        batch_size=96,
        learning_rate=3e-4,
        warmup_steps=2500,
        output_dir="./checkpoints",
        max_sequence_length=64,
        dataset_split="train",
        dataset_name="skymizer/fineweb-edu-dedup-45B",
        dataset_subset="default",
        # dataset_subset="20231101.en",
        # dataset_name="wikimedia/wikipedia",
        # dataset_split="train[0:1000]",
        seed=42,
        target_loss=0.2,  # Desired loss
        pid_Kp=1.0,  # Proportional gain
        pid_Ki=0.1,  # Integral gain
        pid_Kd=0.05,  # Derivative gain
        prob_initial=0.0,  # Initial total noise + mask probability
        prob_min=0.0,  # Minimum total probability
        prob_max=1.0,  # Maximum total probability
        mask_ratio=0.2,  # Fixed ratio for masking
        noise_ratio=0.8,  # Fixed ratio for noise
        checkpoint_interval=2000,
    )
    # # contrastive
    # return TrainingConfig(
    #     num_epochs=1,  # Adjust as needed
    #     batch_size=224,
    #     learning_rate=1e-4,
    #     warmup_steps=0,
    #     output_dir="./checkpoints",
    #     max_sequence_length=64,
    #     dataset_split="train",
    #     dataset_name="skymizer/fineweb-edu-dedup-45B",
    #     dataset_subset="default",
    #     # dataset_subset="20231101.en",
    #     # dataset_name="wikimedia/wikipedia",
    #     # dataset_split="train[0:1000]",
    #     seed=42,
    #     target_loss=0.2,  # Desired loss
    #     pid_Kp=0.02,  # Proportional gain
    #     pid_Ki=0.002,  # Integral gain
    #     pid_Kd=0.001,  # Derivative gain
    #     prob_initial=0.2,  # Initial total noise + mask probability
    #     prob_min=0.0,  # Minimum total probability
    #     prob_max=1.0,  # Maximum total probability
    #     mask_ratio=0.0,  # Fixed ratio for word masking
    #     noise_ratio=1.0,  # Fixed ratio for character noise
    #     checkpoint_interval=2000,
    # )


def setup_dataset(config: TrainingConfig, cache_dir: str):
    """Load and prepare the dataset."""
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_subset,
        split=config.dataset_split,
        cache_dir=cache_dir,
        download_config=DownloadConfig(resume_download=True),
    )
    return dataset

def setup_tokenizer(config: TrainingConfig) -> ByteLevelTokenizer:
    """Initialize the tokenizer with noise configuration."""
    noise_config = NoiseConfig(
        prob=config.prob_initial,
        mask_ratio=config.mask_ratio,
        noise_ratio=config.noise_ratio,
    )
    return ByteLevelTokenizer(
        max_sequence_length=config.max_sequence_length,
        noise_config=noise_config,
    )

def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Initialize configuration
    training_config = get_training_config(args.output_dir)
    
    # Set seed for reproducibility
    set_seed(training_config.seed)

    # Setup dataset and tokenizer
    dataset = setup_dataset(training_config, args.cache_dir)
    tokenizer = setup_tokenizer(training_config)

    # Initialize model
    model = Encodzall(config=encodzall_xs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup TensorBoard logging
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=unique_log_dir)

    try:
        # Start training
        train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=training_config,
            device=device,
            checkpoint_path=args.checkpoint,
            weights_only=args.weights_only,
            writer=writer,
            #stage=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Always close the TensorBoard writer
        writer.close()

if __name__ == "__main__":
    main()
