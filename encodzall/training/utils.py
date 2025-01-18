import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any
import torch


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_token_weights(filepath: str, device: torch.device) -> torch.Tensor:
    """Load token weights from a JSON file."""
    with open(filepath, "r") as fh:
        weights = json.load(fh)
        idx, weight = zip(*sorted(weights.items()))
        weight = torch.tensor(weight, dtype=torch.float32).to(device)
    return weight


def save_failure_data(
    failure_dir: str,
    step: int,
    batch: Dict[str, Any],
    inputs: Dict[str, Any],
    exception: Exception,
) -> None:
    """Save inputs and batch text to a file for debugging."""
    os.makedirs(failure_dir, exist_ok=True)
    failure_data = {
        "step": step,
        "batch_text": batch.get("text", []),
        "inputs": {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        },
        "exception": str(exception),
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    failure_file = os.path.join(failure_dir, f"failure_step_{step}_{timestamp}.pkl")
    with open(failure_file, "wb") as f:
        pickle.dump(failure_data, f)
    print(f"Saved failure data to {failure_file}")
