"""Central configuration for local Windows and Alibaba Cloud runs.

Change only ``runtime_profile`` and ``model_architecture`` for the usual
environment/task switches. Environment variables are optional conveniences
for cloud jobs and do not replace the file-based configuration.
"""

import os
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent

# Main switches. Valid values are documented in README.md.
runtime_profile = os.getenv("TRANSFORMER_PROFILE", "windows_cpu")
model_architecture = os.getenv("TRANSFORMER_MODEL_ARCHITECTURE", "encoder_decoder")
run_action = os.getenv("TRANSFORMER_ACTION", "train")

RUNTIME_PROFILES = {
    # Safe default for any Windows computer, including machines without an NVIDIA GPU.
    "windows_cpu": {
        "device": "cpu",
        "device_index": 0,
        "use_data_parallel": False,
        "num_workers": 0,
        "pin_memory": False,
    },
    # Generic CUDA GPU server profile. It falls back to CPU only when using "auto".
    "gpu_server": {
        "device": "cuda",
        "device_index": 0,
        "use_data_parallel": False,
        "num_workers": 2,
        "pin_memory": True,
    },
    # Useful for a shared codebase: CUDA when available, otherwise CPU.
    "auto": {
        "device": "auto",
        "device_index": 0,
        "use_data_parallel": False,
        "num_workers": 0,
        "pin_memory": True,
    },
}

# Backward-compatible alias for earlier configuration files.
RUNTIME_PROFILES["aliyun_gpu"] = RUNTIME_PROFILES["gpu_server"]

if runtime_profile not in RUNTIME_PROFILES:
    raise ValueError(f"Unknown runtime_profile: {runtime_profile}")
if model_architecture not in {"encoder_decoder", "encoder_only"}:
    raise ValueError(f"Unknown model_architecture: {model_architecture}")

runtime = RUNTIME_PROFILES[runtime_profile]
requested_device = runtime["device"]
device_index = runtime["device_index"]

if requested_device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "runtime_profile='gpu_server' requires a CUDA-enabled PyTorch installation "
            "and a visible NVIDIA GPU. Use 'windows_cpu' or 'auto' on this machine."
        )
    device = torch.device(f"cuda:{device_index}")
elif requested_device == "auto" and torch.cuda.is_available():
    device = torch.device(f"cuda:{device_index}")
else:
    device = torch.device("cpu")

use_data_parallel = (
    runtime["use_data_parallel"]
    and device.type == "cuda"
    and torch.cuda.device_count() > 1
)
device_ids = list(range(torch.cuda.device_count())) if use_data_parallel else []
num_workers = runtime["num_workers"]
pin_memory = runtime["pin_memory"] and device.type == "cuda"

# Model size. Use a much smaller configuration for a CPU smoke test.
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
dropout = 0.1

# Kept for compatibility with the original project comments.
d_k = 64
d_v = 64

# Tokenizer/vocabulary settings.
src_vocab_size = 32000
tgt_vocab_size = 32000
padding_idx = 0
bos_idx = 2
eos_idx = 3

# Training and inference settings.
batch_size = 32
epoch_num = 3
learning_rate = 3e-4
max_len = 60
max_source_len = 128
beam_size = 3

# Translation data and legacy checkpoint locations.
data_dir = PROJECT_DIR / "data"
train_data_path = data_dir / "json" / "train.json"
dev_data_path = data_dir / "json" / "dev.json"
test_data_path = data_dir / "json" / "test.json"
model_path = PROJECT_DIR / "weights" / "transformer_model.pth"
test_model_path = PROJECT_DIR / "run" / "train" / "exp" / "weights" / "best_bleu_26.30.pth"
inference_model_path = test_model_path

# Encoder-only settings. Each JSON item is {"text": "...", "label": 0}.
encoder_only_tokenizer = "english"  # "english" or "chinese"
encoder_only_vocab_size = src_vocab_size
encoder_only_num_labels = 2
encoder_only_label_names = ["negative", "positive"]
encoder_only_pooling = "mean"  # "mean" or "cls"
encoder_only_train_data_path = data_dir / "json" / "encoder_only_train.json"
encoder_only_dev_data_path = data_dir / "json" / "encoder_only_dev.json"
encoder_only_test_data_path = data_dir / "json" / "encoder_only_test.json"

# Output paths are independent from the current working directory.
run_dir = PROJECT_DIR / "run"
