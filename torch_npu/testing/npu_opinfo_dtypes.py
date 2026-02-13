"""
NPU OpInfo extra dtype configuration.
"""

from __future__ import annotations

__all__ = ["NPU_OPINFO_DTYPES"]

import typing as _typing


NPU_OPINFO_DTYPES: _typing.Dict[str, _typing.Dict[str, _typing.Dict[str, _typing.Any]]] = {
    "__rmod__": {
        "forward": {
            "extra": [
                "int32",
                "int64",
            ]
        }
    }
}
