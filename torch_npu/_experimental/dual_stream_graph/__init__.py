"""Dual-stream graph replay over two isolated native NPU graph-tree domains.

Importing this package installs nothing.  Runtime wiring happens only while a
:class:`GraphExecutionDomain` is active (see ``domain.py`` and ``hooks.py``).

The graph tree core (``torch_npu/npu/_graph_tree.py``,
``torch_npu/utils/_graph_tree.py``) is never edited: each lane runs on its own
native ``TreeManagerContainer`` -> ``NPUGraphTreeManager`` -> private memory
pool, and the four integration points (container lookup, tree reset, Inductor
``cudagraphify``, Inductor reset) are wrapped externally and idempotently.

Public entry points
-------------------
- :class:`DualBatchRunner`  -- run one or two argument groups on isolated trees
- :class:`GraphExecutionDomain` -- one lane's execution domain (advanced use)
- :func:`compile_region`    -- compile a raw function with an isolated backend
- :func:`plan_split` / :func:`plan_chunks` -- the batch splitting policy
"""

from torch_npu._experimental.dual_stream_graph.domain import (
    GraphExecutionDomain,
    current_domain,
    reset_graph_execution_domains,
)
from torch_npu._experimental.dual_stream_graph.backend import (
    IsolatedGraphBackend,
    compile_region,
)
from torch_npu._experimental.dual_stream_graph.runner import DualBatchRunner
from torch_npu._experimental.dual_stream_graph.plan import (
    plan_chunks,
    plan_split,
)

__all__ = [
    "DualBatchRunner",
    "GraphExecutionDomain",
    "IsolatedGraphBackend",
    "compile_region",
    "current_domain",
    "plan_chunks",
    "plan_split",
    "reset_graph_execution_domains",
]
