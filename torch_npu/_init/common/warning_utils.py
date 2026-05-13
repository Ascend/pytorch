import os


def _should_print_warning():
    disabled_warning = os.environ.get("TORCH_NPU_DISABLED_WARNING", "0")
    if disabled_warning == "1":
        return False

    rank = os.environ.get("RANK", None)
    return rank is None or rank == "0"
