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
    """Persist per-experiment and cross-experiment training histories."""

    def __init__(self, experiment_dir, config_values, enable_tensorboard=True):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.experiment_dir.parent
        self.experiment_name = self.experiment_dir.name
        self.architecture = config_values.get("model_architecture", "unknown")
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.history_jsonl_path = self.train_dir / "history.jsonl"
        self.history_log_path = self.train_dir / "history.log"
        (self.experiment_dir / "config.json").write_text(
            json.dumps(config_values, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.writer = None
        self.history_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
                history_dir = self.train_dir / "tensorboard_history"
                history_records = self._read_history_records()
                history_has_events = any(history_dir.glob("events.out.tfevents.*"))
                self.history_writer = SummaryWriter(log_dir=str(history_dir))
                if not history_has_events:
                    for step, record in enumerate(history_records, start=1):
                        self._write_history_scalars(self.history_writer, record, step)
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
        history_records = self._refresh_history_files()
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, epoch)
            self.writer.flush()
        if self.history_writer is not None:
            self._write_history_scalars(self.history_writer, history_records[-1], len(history_records))
            self.history_writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.history_writer is not None:
            self.history_writer.close()

    def _read_history_records(self):
        """Read every experiment metrics file so existing runs remain visible."""
        records = []
        for folder in sorted(self.train_dir.glob("exp*")):
            metrics_path = folder / "metrics.jsonl"
            if not folder.is_dir() or not metrics_path.is_file():
                continue
            architecture = self._experiment_architecture(folder)
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record["experiment"] = folder.name
                    record["architecture"] = architecture
                    records.append(record)
        return sorted(
            records,
            key=lambda record: (
                record.get("timestamp_utc", ""),
                record["experiment"],
                record.get("epoch", 0),
            ),
        )

    @staticmethod
    def _experiment_architecture(folder):
        config_path = folder / "config.json"
        if not config_path.is_file():
            return "unknown"
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("model_architecture", "unknown")

    def _refresh_history_files(self):
        records = self._read_history_records()
        self.history_jsonl_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        human_lines = []
        for record in records:
            metrics = ", ".join(
                f"{name}={value:.6f}"
                for name, value in sorted(record.items())
                if isinstance(value, float)
            )
            human_lines.append(
                f"{record.get('timestamp_utc', '')} "
                f"experiment={record['experiment']} "
                f"architecture={record['architecture']} "
                f"epoch={record.get('epoch', '')} {metrics}".rstrip()
            )
        self.history_log_path.write_text("\n".join(human_lines) + "\n", encoding="utf-8")
        return records

    @staticmethod
    def _write_history_scalars(writer, record, step):
        architecture = record.get("architecture", "unknown")
        for name, value in record.items():
            if isinstance(value, (int, float)) and name != "epoch":
                writer.add_scalar(f"history/{architecture}/{name}", value, step)
