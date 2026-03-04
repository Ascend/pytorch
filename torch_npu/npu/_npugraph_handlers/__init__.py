"""NPU Graph Operator Handler Framework (Registry + Template Method).

This package exposes the public API for the NPU Graph operator handler
mechanism and ensures that all **built-in** handlers are registered at
import time.

Public API (re-exported from :mod:`npugraph_handler`)
-----------------------------------------------------
- :class:`NpuGraphOpHandler`          -- base class for operator handlers.
- :func:`register_npu_graph_handler`  -- class decorator to register a handler.

"""

# Re-export public API from the core module
from .npugraph_handler import (
    NpuGraphOpHandler,
    register_npu_graph_handler,
)

# Auto-register built-in handlers, Import built-in handler classes to trigger their ``@register_npu_graph_handler`` decorators.
from .ifa_handler import (  # noqa: F401
    _IFAv1DefaultHandler,
    _IFAv2DefaultHandler,
)
from .simple_handler import _SimpleGraphHandler  # noqa: F401
