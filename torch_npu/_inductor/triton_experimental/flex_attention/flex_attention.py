"""Experimental FlexAttention policy and sparse-mask metadata kernels."""

from typing import Any, Dict, Optional

import sympy

import torch
from torch._inductor.virtualized import V

from torch_npu._inductor.kernel.flex_attention import (
    _filter_autotune_ir_nodes,
    _force_fixed_layout,
    _get_num_vector_core,
    create_compact_q_offsets_fake,
    create_sparse_mask_num_blocks_fake,
    create_zero_int_tensor_fake,
)
from torch_npu._inductor.kernel.flex_attention import (
    patch_flex_attention as _shared_patch_flex_attention,
)
from .flex_attention_strategy import FlexAttentionBackendStrategy
from .flex_attention_adapter import _compile_copy, register_experimental_flex_attention
from .flex_attention_template import (
    flex_attention_bwd_dkdv_mask_out,
    flex_attention_bwd_dkdv_reduce,
    flex_attention_bwd_dkdv_tasklist,
    flex_attention_bwd_dkdv_tasklist_no_split,
    flex_attention_bwd_dq_mask_out,
    flex_attention_bwd_mask_compact,
    flex_attention_bwd_mask_pos,
    flex_attention_compact_offsets,
    flex_attention_compact_mapping,
    flex_attention_fwd_mask_compact,
    flex_attention_fwd_mask_out,
)

from torch._inductor.ir import FixedLayout
from torch._inductor.lowering import empty_strided, lowerings
from torch._inductor.select_algorithm import autotune_select_algorithm

_LINEARIZE_FUSION_NODE_BUDGET = 24
_LINEARIZE_FUSION_CAPTURE_BUDGET = 8


def _graph_node_count(graph) -> int:
    fx_graph = getattr(graph, "graph", graph)
    nodes = getattr(fx_graph, "nodes", None)
    if nodes is None:
        return 0
    return sum(1 for node in nodes if getattr(node, "op", None) != "output")


def _prepare_linearize_fusion_options(
    kernel_options: Optional[Dict[str, Any]],
    score_mod_graph=None,
    mask_mod_graph=None,
) -> dict[str, Any]:
    options = dict(kernel_options or {})
    score_nodes = _graph_node_count(score_mod_graph)
    mask_nodes = _graph_node_count(mask_mod_graph)
    total_nodes = score_nodes + mask_nodes
    captured = len(options.get("FUSED_CAPTURED_BUFFERS", ()))
    options.setdefault("TRITON_EXPERIMENTAL_LINEARIZE", True)
    options.setdefault("TRITON_EXPERIMENTAL_FUSION", True)
    options["FUSE_SMALL_SUBGRAPHS"] = (
        total_nodes <= _LINEARIZE_FUSION_NODE_BUDGET
        and captured <= _LINEARIZE_FUSION_CAPTURE_BUDGET
    )
    options["FUSION_NODE_COUNT"] = total_nodes
    return options

_EXPERIMENTAL_STRATEGY = FlexAttentionBackendStrategy(
    name="triton_experimental",
    option_transform=_prepare_linearize_fusion_options,
    extra_decode_configs=((64, 2, 1),),
    forward_mask_out_template=flex_attention_fwd_mask_out,
    forward_mask_compact_template=flex_attention_fwd_mask_compact,
    backward_dq_mask_out_template=flex_attention_bwd_dq_mask_out,
    backward_dkdv_mask_out_template=flex_attention_bwd_dkdv_mask_out,
    backward_mask_compact_template=flex_attention_bwd_mask_compact,
    backward_mask_pos_template=flex_attention_bwd_mask_pos,
    backward_tasklist_template=flex_attention_bwd_dkdv_tasklist,
    backward_tasklist_no_split_template=flex_attention_bwd_dkdv_tasklist_no_split,
    backward_reduce_template=flex_attention_bwd_dkdv_reduce,
)
aten = torch.ops.aten


def _build_runtime_compact_sparse_mask_offsets(
    *,
    kv_num_blocks,
    kv_indices,
    device,
    context: str,
):
    q_offsets = empty_strided(kv_num_blocks.get_size(), None, dtype=torch.int32, device=device)
    row_count = sympy.prod(kv_num_blocks.get_size())
    upper_capacity = row_count * kv_indices.get_size()[3]
    V.graph.sizevars.check(sympy.Le(upper_capacity, sympy.Integer(torch.iinfo(torch.int32).max)))
    total_blocks = empty_strided([1], stride=[1], dtype=torch.int32, device=device)
    total_blocks = _force_fixed_layout(lowerings[aten.fill_](total_blocks, 0), [1])
    layout = FixedLayout(device, torch.int32, [1], stride=[1])
    inputs = [q_offsets, total_blocks, kv_num_blocks]
    choices = []
    flex_attention_compact_offsets.maybe_append_choice(
        choices=choices, input_nodes=inputs, layout=layout, mutated_inputs=inputs[:2],
        call_sizes=[row_count], num_stages=1, num_warps=4, NUM_VECTOR_CORE=_get_num_vector_core(),
    )
    if not choices:
        raise RuntimeError(f"{context} could not create compact sparse mask metadata kernel")
    autotune_select_algorithm(
        f"{context}_compact_sparse_mask_metadata", choices,
        _filter_autotune_ir_nodes(inputs, choices), layout,
        input_gen_fns={
            0: create_compact_q_offsets_fake,
            1: create_zero_int_tensor_fake,
            2: create_sparse_mask_num_blocks_fake,
        },
    )
    return q_offsets, total_blocks, row_count


from torch_npu._inductor.kernel import flex_attention as _default_flex_attention

_build_runtime_compact_sparse_mask_mapping = _compile_copy(
    _default_flex_attention._build_runtime_compact_sparse_mask_mapping,
    {"flex_attention_compact_mapping": flex_attention_compact_mapping},
)


def patch_flex_attention() -> None:
    _shared_patch_flex_attention()


def _register_npu_inductor_flex_attention():
    register_experimental_flex_attention(
        _EXPERIMENTAL_STRATEGY,
        _build_runtime_compact_sparse_mask_offsets,
        _build_runtime_compact_sparse_mask_mapping,
    )

_validate_device = _default_flex_attention._validate_device
