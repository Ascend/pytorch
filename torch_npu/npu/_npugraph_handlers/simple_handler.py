"""Simple Graph Handlers (Paged Attention / MLA)."""

__all__ = []

from .npugraph_handler import NpuGraphOpHandler, register_npu_graph_handler


@register_npu_graph_handler([
    "_npu_paged_attention.default",
    "npu_multi_head_latent_attention.out",
])
class _SimpleGraphHandler(NpuGraphOpHandler):
    """Handler for PA (Paged Attention) and MLA operators.

    Update behavior is declarative via :attr:`UPDATE_SPECS`. The base class's
    spec-driven ``update_args`` walks this map and assigns the matching
    update_input key's value to the recorded arg slot.
    """

    UPDATE_SPECS = {
        "_npu_paged_attention.default": [("arg", 7, "context_lens")],
        "npu_multi_head_latent_attention.out": [("arg", 5, "context_lens")],
    }
