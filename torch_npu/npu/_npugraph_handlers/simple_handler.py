"""Simple Graph Handlers (Paged Attention / MLA)."""

__all__ = []

from .npugraph_handler import NpuGraphOpHandler, register_npu_graph_handler


@register_npu_graph_handler([
    "_npu_paged_attention.default",
    "npu_multi_head_latent_attention.out",
])
class _SimpleGraphHandler(NpuGraphOpHandler):
    """Handler for PA (Paged Attention) and MLA operators.

    Attributes:
        _OP_ARG_SPECS (dict[str, tuple[int, str]]): Specifies
            ``op_name -> (arg_index, update_key)`` for each supported
            operator.
    """

    _OP_ARG_SPECS = {
        "_npu_paged_attention.default": (7, "context_lens"),
        "npu_multi_head_latent_attention.out": (5, "context_lens"),
    }

    @classmethod
    def update_args(cls, record, update_input):
        spec = cls._OP_ARG_SPECS.get(record.op_cache_entry.__name__)
        if spec:
            arg_index, key = spec
            if key in update_input and len(record.args) >= (arg_index + 1):
                record.args[arg_index] = update_input[key]
