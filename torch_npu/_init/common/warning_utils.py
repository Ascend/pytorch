import os
import warnings

_deprecated_warning_emitted = False


def _should_print_warning():
    global _deprecated_warning_emitted

    disabled_warning = os.environ.get("TORCH_NPU_WARNING_DISABLE")
    if disabled_warning is None:
        disabled_warning = os.environ.get("TORCH_NPU_DISABLED_WARNING")
    if disabled_warning == "1":
        return False

    rank = os.environ.get("RANK", None)
    if rank is not None and rank != "0":
        return False

    if not _deprecated_warning_emitted and "TORCH_NPU_DISABLED_WARNING" in os.environ:
        _deprecated_warning_emitted = True
        warnings.warn(
            "TORCH_NPU_DISABLED_WARNING is deprecated and will be removed in a "
            "future version. Use TORCH_NPU_WARNING_DISABLE instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return True
