import torch


def _parse(version_str: str) -> tuple:
    parts = version_str.split("+")[0].split(".")
    return (int(parts[0]), int(parts[1]))


CURRENT_VERSION: tuple = _parse(torch.__version__)

# Bump this when dropping old version support; run tools/check_compat.py to find stale COMPAT blocks.
MIN_SUPPORTED_VERSION: tuple = (2, 10)
