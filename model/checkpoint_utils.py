"""Checkpoint helpers that keep old translation weights loadable."""

from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    architecture,
    metrics=None,
    optimizer=None,
    epoch=None,
    config_values=None,
):
    """Save model, metrics, and optional training state for resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    payload = {
        "architecture": architecture,
        "model_state_dict": state_model.state_dict(),
        "metrics": metrics or {},
        "epoch": epoch,
        "config": config_values or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(
        payload,
        path,
    )


def load_checkpoint(path, model, device, expected_architecture):
    """Load new metadata checkpoints and the project's legacy state_dict files."""
    try:
        # Prefer safe tensor-only deserialization on current PyTorch versions.
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        # Keep compatibility with older PyTorch releases that predate weights_only.
        checkpoint = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        architecture = checkpoint.get("architecture")
        if architecture is not None and architecture != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture is '{architecture}', but config requests "
                f"'{expected_architecture}'."
            )
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        # The original repository saved a raw state_dict. It is supported for
        # encoder-decoder translation checkpoints only.
        if expected_architecture != "encoder_decoder":
            raise ValueError(
                "A legacy checkpoint has no architecture metadata and cannot be "
                "safely loaded as a decoder-only model. Train decoder_only first."
            )
        state_dict = checkpoint
        metadata = {"architecture": "encoder_decoder", "legacy": True}
    model.load_state_dict(state_dict)
    return metadata


def restore_training_checkpoint(path, model, device, expected_architecture, optimizer=None):
    """Restore a metadata checkpoint and optional optimizer/scheduler state."""
    metadata = load_checkpoint(path, model, device, expected_architecture)
    if optimizer is not None:
        optimizer_state = metadata.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("Checkpoint has no optimizer state and cannot resume training.")
        optimizer.load_state_dict(optimizer_state)
    return metadata
