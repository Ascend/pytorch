"""Runtime compatibility helpers for upstream PyTorch tests on NPU CI."""

try:
    import torch
except Exception:
    torch = None


def _patch_cuda_device_capability() -> None:
    if torch is None or not hasattr(torch, "cuda"):
        return

    original_get_device_capability = getattr(torch.cuda, "get_device_capability", None)
    if original_get_device_capability is None:
        return

    def safe_get_device_capability(*args, **kwargs):
        try:
            capability = original_get_device_capability(*args, **kwargs)
        except Exception:
            capability = None

        if isinstance(capability, tuple) and len(capability) == 2:
            return capability

        npu_module = getattr(torch, "npu", None)
        if npu_module is not None:
            try:
                if npu_module.is_available():
                    return (10, 0)
            except Exception:
                pass
        return (0, 0)

    torch.cuda.get_device_capability = safe_get_device_capability


_patch_cuda_device_capability()