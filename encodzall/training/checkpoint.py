import os
import torch
from simple_pid import PID
from typing import Optional, Tuple, Any
from encodzall.config.training_config import TrainingConfig


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    pid: PID,
    step: int,
    config: TrainingConfig,
    prob: float,
    dataset_offset: int,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "pid_tunings": pid.tunings,
        "pid_setpoint": pid.setpoint,
        "pid_auto_mode": pid.auto_mode,
        "prob": prob,
        "dataset_offset": dataset_offset,
        "config": config.__dict__,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    config: Optional[TrainingConfig] = None,
    load_weights_only: bool = False,
) -> Tuple[int, TrainingConfig, PID, float, int]:
    """
    Load training checkpoint.

    If `load_weights_only` is True, then only model weights are loaded; all other
    training states (optimizer, scheduler, scaler, PID, steps) are ignored or reset.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # ---------------------------------------------------------------------
    # If just loading the weights, skip loading optimizer, scheduler, etc.
    # ---------------------------------------------------------------------
    if load_weights_only:
        return

    # ---------------------------------------------------------------------
    # Otherwise, load the full training state
    # ---------------------------------------------------------------------
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Rebuild PID from checkpoint
    pid = PID(
        Kp=checkpoint["pid_tunings"][0],
        Ki=checkpoint["pid_tunings"][1],
        Kd=checkpoint["pid_tunings"][2],
        setpoint=checkpoint["pid_setpoint"],
    )
    pid.auto_mode = checkpoint.get("pid_auto_mode", True)

    prob = checkpoint.get("prob", config.prob_initial if config else 0.0)
    dataset_offset = checkpoint.get("dataset_offset", 0)

    if not config:
        saved_config = checkpoint.get("config")
        if saved_config:
            config = TrainingConfig(**saved_config)
        else:
            raise ValueError("Config missing from checkpoint and not provided.")

    return checkpoint["step"] + 1, config, pid, prob, dataset_offset + 1
