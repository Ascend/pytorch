"""Loggers for fxrt's Python code.

torch_npu registers the ``fxrt`` log alias, so ``TORCH_NPU_LOGS`` and
``torch._logging.set_logs`` control every ``fxrt.*`` logger together with the
C++ side of fxrt. Records carry the same ``[FXRT]`` prefix as the C++ logs.
"""

import logging

_PREFIX = "[FXRT] "


class _PrefixAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"{_PREFIX}{msg}", kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return the logger for an ``fxrt.*`` module, prefixing records with ``[FXRT]``."""
    return _PrefixAdapter(logging.getLogger(name), {})
