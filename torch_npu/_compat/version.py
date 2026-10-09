import torch

__all__ = ["CURRENT_VERSION", "MIN_SUPPORTED_VERSION"]


def _parse(version_str: str) -> tuple:
    parts = version_str.split("+")[0].split(".")
    return (int(parts[0]), int(parts[1]))


CURRENT_VERSION: tuple = _parse(torch.__version__)

# Bump this when dropping old version support: the registry rejects any side whose
# threshold is <= this value, and tools/compat_check/version_compat_check.py finds
# the remaining COMPAT blocks.
MIN_SUPPORTED_VERSION: tuple = (2, 13)
