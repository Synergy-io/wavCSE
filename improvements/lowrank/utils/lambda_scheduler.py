"""Epoch-based scheduling for the RAMUSA regularization coefficient."""

import math


def get_ramusa_lambda(
    schedule_type: str,
    global_epoch: int,
    lambda_max: float,
    warmup_epochs: int = 0,
    delay_epochs: int = 0,
) -> float:
    """Return the RAMUSA coefficient for a zero-based global epoch."""
    if not math.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be finite and non-negative")
    if global_epoch < 0:
        raise ValueError("global_epoch must be non-negative")
    if warmup_epochs < 0 or delay_epochs < 0:
        raise ValueError("warmup_epochs and delay_epochs must be non-negative")

    epoch = global_epoch + 1
    if schedule_type == "fixed":
        return float(lambda_max)
    if schedule_type == "linear_warmup":
        if warmup_epochs == 0:
            return float(lambda_max)
        return float(lambda_max * min(1.0, epoch / float(warmup_epochs)))
    if schedule_type == "delayed_linear_warmup":
        if epoch <= delay_epochs:
            return 0.0
        if warmup_epochs == 0:
            return float(lambda_max)
        progress = min(1.0, (epoch - delay_epochs) / float(warmup_epochs))
        return float(lambda_max * progress)

    raise ValueError(f"Unknown RAMUSA lambda schedule: {schedule_type}")