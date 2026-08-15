"""Reproducibility and experiment tracking helpers."""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for repeatable local experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        # warn_only keeps operations usable on hardware that lacks a strict
        # deterministic implementation while still surfacing a warning.
        torch.use_deterministic_algorithms(True, warn_only=True)


def config_snapshot(config_module):
    """Return JSON-safe configuration values that are useful for reproduction."""
    snapshot = {}
    for name in dir(config_module):
        if name.startswith("_") or not name.islower():
            continue
        value = getattr(config_module, name)
        if isinstance(value, Path):
            snapshot[name] = str(value)
        elif isinstance(value, torch.device):
            snapshot[name] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            snapshot[name] = value
    return snapshot


class ExperimentTracker:
    """Persist configuration and epoch metrics as JSONL and TensorBoard events."""

    def __init__(self, experiment_dir, config_values, enable_tensorboard=True):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        (self.experiment_dir / "config.json").write_text(
            json.dumps(config_values, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
            except ModuleNotFoundError:
                logging.warning(
                    "TensorBoard is unavailable. Install requirements.txt to enable event logs."
                )

    def log_epoch(self, epoch, metrics):
        record = {
            "epoch": epoch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, epoch)
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
