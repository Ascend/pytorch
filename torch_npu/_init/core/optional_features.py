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
            "Do not set it to 1 to prevent some unknown errors"
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


def _enable_optional_features():
    _enable_sanitizer_if_needed()
    _configure_interactive_mode()
    _enable_transfer_to_npu_if_needed()
