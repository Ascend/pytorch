"""Public exports for torch_npu._inductor.mfusion.

Keep this package import lightweight for public_bindings tests. Heavy
dependencies from graph_fusion (inductor internals / optional mfusion stack)
are loaded lazily via __getattr__.
"""

from typing import Any


__all__ = ["MFusionPatch", "mfusion_graph_fusion"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import graph_fusion as _graph_fusion

        return getattr(_graph_fusion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
