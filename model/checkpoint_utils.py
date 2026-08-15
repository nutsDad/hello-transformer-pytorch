"""Checkpoint helpers that keep old translation weights loadable."""

from pathlib import Path

import torch


def save_checkpoint(path, model, architecture, metrics=None):
    """Save model state with architecture metadata for new checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        },
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
