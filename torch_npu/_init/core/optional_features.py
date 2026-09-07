import os
import warnings

from torch_npu.utils.utils import _is_interactive_command_line


def _enable_sanitizer_if_needed():
    """
    Enable NPU Sanitizer.
    """
    if "TORCH_NPU_SANITIZER" in os.environ:
        import torch_npu.npu._sanitizer as csan

        csan.enable_npu_sanitizer()


def _configure_interactive_mode():
    if _is_interactive_command_line():
        os.environ["TASK_QUEUE_ENABLE"] = "0"
        warnings.warn(
            "On the interactive interface, the value of TASK_QUEUE_ENABLE is set to 0 by default. "
            "Do not set it to 1 to avoid unexpected errors."
        )


def _enable_transfer_to_npu_if_needed():
    """
    Enable transfer_to_npu via environment variable
    """
    transfer_to_npu_env = os.getenv("TORCH_TRANSFER_TO_NPU", "0")
    if transfer_to_npu_env == "1":
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    elif transfer_to_npu_env != "0":
        raise ValueError(
            f"Invalid value for TORCH_TRANSFER_TO_NPU: {transfer_to_npu_env}. "
            "Only '0' or '1' is supported."
        )


def _preload_inductor_for_hccl_fr_if_needed():
    """Preload the PyTorch Inductor symbols used by HCCL flight recorder."""
    trace_buffer_size = os.getenv("TORCH_HCCL_TRACE_BUFFER_SIZE", "0")
    try:
        if int(trace_buffer_size) <= 0:
            return
    except ValueError:
        warnings.warn(
            f"Invalid TORCH_HCCL_TRACE_BUFFER_SIZE={trace_buffer_size!r}; "
            "skip torch._inductor preload."
        )
        return

    try:
        import importlib

        importlib.import_module("torch._inductor")
        codecache = importlib.import_module("torch._inductor.codecache")
        py_code_cache = getattr(codecache, "PyCodeCache", None)
        if getattr(py_code_cache, "stack_frames_for_code", None) is None:
            warnings.warn(
                "HCCL flight recorder is enabled, but "
                "torch._inductor.codecache.PyCodeCache.stack_frames_for_code is unavailable."
            )
    except Exception as error:
        warnings.warn(
            "Failed to preload torch._inductor for HCCL flight recorder: "
            f"{error!r}. HCCL dump will continue with best-effort symbolization."
        )


def _enable_optional_features():
    _enable_sanitizer_if_needed()
    _configure_interactive_mode()
    _enable_transfer_to_npu_if_needed()
    _preload_inductor_for_hccl_fr_if_needed()
