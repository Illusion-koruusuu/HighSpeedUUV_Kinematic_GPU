"""PyTorch device selection and GPU performance settings."""
import torch


def resolve_device(requested):
    """Resolve a user-facing device name and validate CUDA availability."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access an NVIDIA GPU. "
            "Check nvidia-smi and the CUDA_VISIBLE_DEVICES setting."
        )
    return device


def configure_torch(device, threads=None):
    """Enable fast A100-class TF32 paths without changing model precision."""
    if threads is not None:
        torch.set_num_threads(max(1, int(threads)))
    if device.type != "cuda":
        return
    index = 0 if device.index is None else device.index
    torch.cuda.set_device(index)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
