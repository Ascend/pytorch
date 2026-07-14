""" Triton Implementation of the flex_attention Kernel"""

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial, wraps
from types import FunctionType
from typing import Any, Dict, Optional, Sequence, Union

import sympy

import torch
from torch._inductor.virtualized import V, ops
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from torch._inductor import config
from torch_npu._inductor import config as npu_config
from torch_npu._inductor.config import log

from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    ExternKernel,
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Scatter,
    StorageBox,
    Subgraph,
    TensorBox,
)
from torch._inductor.lowering import (
    _full,
    check_and_broadcast_indices,
    empty_strided,
    expand,
    fallback_handler,
    index_output_size_and_inner_fn,
    lowerings,
    register_lowering,
    to_dtype,
)
from torch._inductor.select_algorithm import autotune_select_algorithm, realize_inputs, TritonTemplate
from torch._inductor.kernel.flex_decoding import create_flex_decoding_kernel
from torch.nn.attention import flex_attention as flex_attention_module
from torch._inductor.kernel.flex_attention import (
    maybe_realize,
    construct_strides,
    create_placeholder,
    set_head_dim_values,
    create_indices_fake,
    SymbolicGridFn,
    flex_attention_grid,
    validate_joint_graph,
    process_joint_outputs,
    build_subgraph_buffer,
    create_num_blocks_fake_generator
)


def _tag_flex_attention_report_choices(new_choices, mode, cfg):
    """Attach tiling metadata used by NPU fallback choice ordering."""
    if "BLOCK_M" in cfg and "BLOCK_N" in cfg:
        report_config = {
            "BLOCK_M": cfg["BLOCK_M"],
            "BLOCK_N": cfg["BLOCK_N"],
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
        }
    else:
        report_config = {
            "BLOCK_M": cfg["BLOCK_M1"],
            "BLOCK_N": cfg["BLOCK_N1"],
            "BLOCK_M2": cfg["BLOCK_M2"],
            "BLOCK_N2": cfg["BLOCK_N2"],
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
        }
    for choice in new_choices:
        setattr(choice, "_flex_attention_report_mode", mode)
        setattr(choice, "_flex_attention_report_config", report_config.copy())


def _tag_choice_configs(new_choices, attr_name: str, cfg: dict[str, Any]) -> None:
    """Attach tiling metadata used by NPU fallback choice ordering."""
    for choice in new_choices:
        setattr(choice, attr_name, cfg.copy())


def _tag_choice_attr(new_choices, attr_name: str, value: Any) -> None:
    for choice in new_choices:
        setattr(choice, attr_name, value)


_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION = "COMPACT_SPARSE_MASK_TOTAL_BLOCKS"
_COMPACT_SPARSE_MASK_ATTR = "_npu_compact_sparse_mask_metadata"


def _make_closure_cell(value):
    def capture():
        return value

    return capture.__closure__[0]


def _maybe_cast_mask_mod_constant_to_fp32(value):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dtype == torch.bool:
        return value
    if getattr(value.dtype, "is_floating_point", False):
        return value
    if getattr(value.dtype, "is_complex", False):
        return value
    return value.to(dtype=torch.float32)


def _maybe_cast_mask_mod_tensor_constants_to_fp32(mask_mod):
    if isinstance(mask_mod, partial):
        converted_func = _maybe_cast_mask_mod_tensor_constants_to_fp32(mask_mod.func)
        converted_args = tuple(
            _maybe_cast_mask_mod_constant_to_fp32(arg) for arg in mask_mod.args
        )
        converted_keywords = None
        keyword_changed = False
        if mask_mod.keywords:
            converted_keywords = {
                name: _maybe_cast_mask_mod_constant_to_fp32(value)
                for name, value in mask_mod.keywords.items()
            }
            keyword_changed = any(
                converted_keywords[name] is not value
                for name, value in mask_mod.keywords.items()
            )
        changed = (
            converted_func is not mask_mod.func
            or any(new is not old for new, old in zip(converted_args, mask_mod.args))
            or keyword_changed
        )
        if not changed:
            return mask_mod
        converted = partial(
            converted_func,
            *converted_args,
            **(converted_keywords or {}),
        )
        converted.__dict__.update(getattr(mask_mod, "__dict__", {}))
        return converted

    if not isinstance(mask_mod, FunctionType) or mask_mod.__closure__ is None:
        return mask_mod

    converted_cells = []
    changed = False
    for cell in mask_mod.__closure__:
        try:
            value = cell.cell_contents
        except ValueError:
            converted_cells.append(cell)
            continue
        converted_value = _maybe_cast_mask_mod_constant_to_fp32(value)
        if converted_value is value:
            converted_cells.append(cell)
            continue
        converted_cells.append(_make_closure_cell(converted_value))
        changed = True

    if not changed:
        return mask_mod

    converted = FunctionType(
        mask_mod.__code__,
        mask_mod.__globals__,
        mask_mod.__name__,
        mask_mod.__defaults__,
        tuple(converted_cells),
    )
    converted.__kwdefaults__ = getattr(mask_mod, "__kwdefaults__", None)
    converted.__annotations__ = dict(getattr(mask_mod, "__annotations__", {}))
    converted.__dict__.update(getattr(mask_mod, "__dict__", {}))
    converted.__module__ = getattr(mask_mod, "__module__", None)
    converted.__qualname__ = getattr(mask_mod, "__qualname__", mask_mod.__name__)
    return converted


def _precompute_compact_sparse_mask_metadata(block_mask: Any) -> dict[str, Any]:
    kv_num_blks = getattr(block_mask, "kv_num_blocks", None)
    if kv_num_blks is None:
        raise ValueError("block_mask is missing kv_num_blocks")
    if not isinstance(kv_num_blks, torch.Tensor):
        raise TypeError(f"expected tensor kv_num_blocks, got {type(kv_num_blks).__name__}")

    q_counts_by_segment = kv_num_blks.reshape(-1, kv_num_blks.shape[-1]).to(torch.int32)
    q_counts = q_counts_by_segment.reshape(-1)
    q_counts_cpu = q_counts.detach().to("cpu", dtype=torch.int64)
    if q_counts_cpu.numel() == 0:
        raise ValueError("kv_num_blocks is empty")
    total_normal_blocks = int(q_counts_cpu.sum().item())
    if total_normal_blocks <= 0:
        raise ValueError("compact sparse mask requires at least one normal block")
    max_normal_blocks = int(q_counts_cpu.max().item())
    if max_normal_blocks <= 0:
        raise ValueError("compact sparse mask max normal blocks must be positive")

    segment_totals = torch.sum(q_counts_by_segment, dim=1, dtype=torch.int32)
    segment_bases = torch.empty_like(segment_totals)
    segment_bases[0] = 0
    if segment_totals.numel() > 1:
        segment_bases[1:] = torch.cumsum(segment_totals[:-1], dim=0)
    q_offsets_by_segment = torch.empty(
        (q_counts_by_segment.shape[0], q_counts_by_segment.shape[1] + 1),
        dtype=torch.int32,
        device=kv_num_blks.device,
    )
    q_offsets_by_segment[:, 0] = segment_bases
    q_offsets_by_segment[:, 1:] = (
        segment_bases[:, None] + torch.cumsum(q_counts_by_segment, dim=1)
    )
    q_offsets = q_offsets_by_segment.reshape(-1).contiguous()

    row_ids = torch.arange(q_counts.numel(), dtype=torch.int32, device=kv_num_blks.device)
    local_blk_ids = torch.arange(max_normal_blocks, dtype=torch.int32, device=kv_num_blks.device)
    valid_flat_mask = local_blk_ids.unsqueeze(0) < q_counts.unsqueeze(1)
    flat_to_row = row_ids.unsqueeze(1).expand(-1, max_normal_blocks)[valid_flat_mask].contiguous()
    flat_to_blk = local_blk_ids.unsqueeze(0).expand(q_counts.numel(), -1)[valid_flat_mask].contiguous()

    return {
        "q_offsets": q_offsets,
        "flat_to_row": flat_to_row,
        "flat_to_blk": flat_to_blk,
        "total_normal_blocks": total_normal_blocks,
        "sparse_mask_hq": int(kv_num_blks.shape[1]),
    }


def _wrap_mask_mod_with_compact_sparse_mask_metadata(mask_mod, metadata: dict[str, Any]):
    q_offsets = metadata["q_offsets"]
    flat_to_row = metadata["flat_to_row"]
    flat_to_blk = metadata["flat_to_blk"]

    @wraps(mask_mod)
    def wrapped_mask_mod(b, h, q_idx, kv_idx):
        result = mask_mod(b, h, q_idx, kv_idx)
        if isinstance(q_idx, torch.Tensor):
            zero_index = torch.zeros_like(q_idx, dtype=torch.int64)
            zero_value = torch.zeros_like(q_idx, dtype=torch.float32)
            sentinel = (
                q_offsets[zero_index].to(torch.float32)
                + flat_to_row[zero_index].to(torch.float32)
                + flat_to_blk[zero_index].to(torch.float32)
            ) < zero_value
        else:
            zero_value = torch.tensor(0.0, dtype=torch.float32, device=q_offsets.device)
            sentinel = (
                q_offsets[0].to(torch.float32)
                + flat_to_row[0].to(torch.float32)
                + flat_to_blk[0].to(torch.float32)
            ) < zero_value
        if isinstance(result, torch.Tensor):
            return torch.logical_or(result, sentinel)
        return bool(result) or bool(sentinel.item())

    setattr(wrapped_mask_mod, _COMPACT_SPARSE_MASK_ATTR, metadata)
    return wrapped_mask_mod


def create_zero_int_tensor_fake(x) -> torch.Tensor:
    size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
    return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())


def create_compact_q_offsets_fake(x) -> torch.Tensor:
    size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
    return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())


def _create_sparse_mask_num_blocks_fake_generator(max_normal_blocks: int):
    num_blocks_for_autotuning = 1 if int(max_normal_blocks) > 0 else 0

    def create_sparse_mask_num_blocks_fake(x) -> torch.Tensor:
        size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
        return torch.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_sparse_mask_num_blocks_fake


def _create_sparse_mask_indices_fake_generator():
    def create_sparse_mask_indices_fake(x) -> torch.Tensor:
        size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
        return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())

    return create_sparse_mask_indices_fake


from torch_npu._inductor.kernel.flex_attention_metadata import (
    apply_kernel_options_from_eager_block_mask,
    apply_kernel_options_from_block_sparse_mask,
    infer_eager_block_mask_kernel_options,
)
from torch_npu._inductor.kernel.flex_attention_config_generator import (
    build_sparse_mask_candidate_configs,
    get_bwd_dkdv_compile_options,
    get_bwd_dq_compile_options,
    generate_bwd_split_mask_out_candidate_configs,
    generate_fwd_candidate_configs,
    is_bwd_config_compatible,
    prefer_max_tiling_without_benchmark,
    sparse_mask_attention_cvpipeline_config_variants,
    validate_benchmark_config,
)
from torch_npu._inductor.select_algorithm import NPUTritonTemplate
from torch_npu._inductor import config as npu_config
aten = torch.ops.aten
Expr = sympy.Expr


def _maybe_copy_to_dtype(x: TensorBox, dtype: torch.dtype) -> TensorBox:
    if x.get_dtype() == dtype:
        return x
    return to_dtype(x, dtype, copy=True)


def _force_fixed_layout(x: TensorBox, strides: Sequence[Any]) -> TensorBox:
    data = x.data
    if isinstance(data, StorageBox) and isinstance(data.data, ComputedBuffer):
        data.data.layout = FixedLayout(
            x.get_device(),
            x.get_dtype(),
            x.get_size(),
            stride=[sympy.sympify(s) for s in strides],
        )
    return x


def _get_graph_output_node(graph) -> Any:
    for node in reversed(graph.nodes):
        if node.op != "output":
            continue
        output_arg = node.args[0]
        if isinstance(output_arg, (tuple, list)):
            return output_arg[0] if output_arg else None
        return output_arg
    return None


def _is_score_mod_identity_graph(fw_graph) -> bool:
    graph = fw_graph.graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if not placeholders:
        return False
    return _get_graph_output_node(graph) is placeholders[0]


def _is_grad_score_mod_identity_graph(joint_graph) -> bool:
    graph = joint_graph.graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(placeholders) < 6:
        return False
    grad_score_ph = placeholders[5]
    output_arg = _get_graph_output_node(graph)
    if output_arg is grad_score_ph:
        return True

    for node in reversed(graph.nodes):
        if node.op != "output":
            continue
        raw_output = node.args[0]
        if isinstance(raw_output, (tuple, list)) and raw_output:
            return raw_output[0] is grad_score_ph
        return False
    return False


def patch_flex_attention() -> None:
    """Patch the Python flex_attention entry so eager block-mask metadata is injected transparently."""
    current_flex_attention = flex_attention_module.flex_attention
    current_create_block_mask = flex_attention_module.create_block_mask
    if (
        getattr(current_flex_attention, "_npu_metadata_patch_applied", False)
        and getattr(current_create_block_mask, "_npu_metadata_patch_applied", False)
    ):
        return

    def flex_attention_with_metadata(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Any = None,
        block_mask: Any = None,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        kernel_options: Optional[Dict[str, Any]] = None,
    ):
        """Inject eager block-mask metadata before delegating to the original flex_attention entry."""
        updated_kernel_options = apply_kernel_options_from_eager_block_mask(
            kernel_options,
            block_mask,
            context="py-api",
            allow_tensor_analysis=not torch.compiler.is_dynamo_compiling(),
        )
        cached_options = getattr(block_mask, "_npu_flex_attention_kernel_options", None)
        if isinstance(cached_options, dict):
            updated_kernel_options = dict(updated_kernel_options)
            if _COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION in cached_options:
                updated_kernel_options[_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION] = (
                    cached_options[_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION]
                )
        return current_flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
            kernel_options=updated_kernel_options,
        )

    if not getattr(current_flex_attention, "_npu_metadata_patch_applied", False):
        flex_attention_with_metadata = wraps(current_flex_attention)(flex_attention_with_metadata)
        flex_attention_with_metadata._npu_metadata_patch_applied = True
        flex_attention_module.flex_attention = flex_attention_with_metadata

    if not getattr(current_create_block_mask, "_npu_metadata_patch_applied", False):
        @wraps(current_create_block_mask)
        def create_block_mask_with_metadata(*args, **kwargs):
            if args:
                converted_mask_mod = _maybe_cast_mask_mod_tensor_constants_to_fp32(args[0])
                if converted_mask_mod is not args[0]:
                    args = (converted_mask_mod, *args[1:])
            elif "mask_mod" in kwargs:
                converted_mask_mod = _maybe_cast_mask_mod_tensor_constants_to_fp32(
                    kwargs["mask_mod"]
                )
                if converted_mask_mod is not kwargs["mask_mod"]:
                    kwargs = dict(kwargs)
                    kwargs["mask_mod"] = converted_mask_mod

            block_mask = current_create_block_mask(*args, **kwargs)
            try:
                kernel_options = infer_eager_block_mask_kernel_options(block_mask)
                merged_kernel_options = dict(kernel_options) if kernel_options else {}
                try:
                    compact_metadata = _precompute_compact_sparse_mask_metadata(block_mask)
                except Exception as compact_exc:
                    raise RuntimeError(
                        "NPU flex attention requires compact sparse mask metadata"
                    ) from compact_exc

                block_mask.mask_mod = _wrap_mask_mod_with_compact_sparse_mask_metadata(
                    block_mask.mask_mod,
                    compact_metadata,
                )
                setattr(
                    block_mask,
                    _COMPACT_SPARSE_MASK_ATTR,
                    compact_metadata,
                )
                merged_kernel_options[_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION] = (
                    compact_metadata["total_normal_blocks"]
                )
                merged_kernel_options["SPARSE_MASK_HQ"] = compact_metadata[
                    "sparse_mask_hq"
                ]
                merged_kernel_options["SPARSE_MASK_HEAD_SHARED"] = False
                if kernel_options:
                    setattr(
                        block_mask,
                        "_npu_flex_attention_kernel_options",
                        merged_kernel_options,
                    )
                    log.info(
                        "[flex_attention][create_block_mask] cached kernel options: %s",
                        merged_kernel_options,
                    )
                else:
                    setattr(
                        block_mask,
                        "_npu_flex_attention_kernel_options",
                        merged_kernel_options,
                    )
            except Exception as exc:
                log.debug(
                    "Failed to cache kernel options on BlockMask: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            return block_mask

        create_block_mask_with_metadata._npu_metadata_patch_applied = True
        flex_attention_module.create_block_mask = create_block_mask_with_metadata


def _get_flex_attention_additional_lowerings():
    """
    Get additional lowerings for flex_attention subgraph.

    These lowerings are used to allow index and bitwise operations to be lowered
    as pointwise ops instead of fallback in the mask_mod subgraph.
    """
    from torch._inductor.lowering import make_pointwise, index_impl
    from torch._inductor.subgraph_lowering import PointwiseSubgraphLowering

    additional_lowerings = {}

    def index_pointwise(x, indices):
        return index_impl(x, indices, check=True)

    additional_lowerings[aten.index] = index_pointwise
    additional_lowerings[aten.index.Tensor] = index_pointwise

    bitwise_and_fn = make_pointwise(ops.bitwise_and)

    def bitwise_and_tensor(a, b):
        return bitwise_and_fn(a, b)

    bitwise_or_fn = make_pointwise(ops.bitwise_or)

    def bitwise_or_tensor(a, b):
        return bitwise_or_fn(a, b)

    bitwise_not_fn = make_pointwise(ops.bitwise_not)

    def bitwise_not_default(a):
        return bitwise_not_fn(a)

    additional_lowerings[aten.bitwise_and.Tensor] = bitwise_and_tensor
    additional_lowerings[aten.bitwise_or.Tensor] = bitwise_or_tensor
    additional_lowerings[aten.bitwise_not.default] = bitwise_not_default

    return additional_lowerings


def _build_subgraph_buffer_with_additional_lowerings(args, subgraph):
    """
    Build subgraph buffer with additional lowerings for flex_attention.

    This function creates a PointwiseSubgraphLowering with additional_lowerings
    to handle index and bitwise operations as pointwise ops.
    """
    from torch._inductor.subgraph_lowering import PointwiseSubgraphLowering

    additional_lowerings = _get_flex_attention_additional_lowerings()
    pw_subgraph = PointwiseSubgraphLowering(
        subgraph.graph_module,
        root_graph_lowering=V.graph,
        additional_lowerings=additional_lowerings,
    )
    with V.set_graph_handler(pw_subgraph):
        pw_subgraph.run(*args)

    def convert_output_node_to_buffer(output_buffer):
        from torch._inductor.ir import ComputedBuffer, FlexibleLayout, StorageBox
        if output_buffer is None:
            return None
        if isinstance(output_buffer, ComputedBuffer):
            return output_buffer
        assert isinstance(output_buffer, TensorBox), (
            "The output node for flex attention's subgraph must be a TensorBox, but got: ",
            type(output_buffer),
        )
        assert isinstance(output_buffer.data, StorageBox), (
            "The output node for the flex attention subgraph must be a StorageBox, but got: ",
            type(output_buffer),
        )
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=output_buffer.data.get_device(),
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def _use_flex_decoding(query, kernel_options):
    force_flex = kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)
    short_query_length = V.graph.sizevars.evaluate_expr(
        sympy.Lt(query.get_size()[-2], 128)
    )
    non_zero_length = V.graph.sizevars.evaluate_expr(sympy.Gt(query.get_size()[-2], 0))
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))
    return not force_flex and short_query_length and static_batch and static_num_heads


def _validate_device(query, key, value):
    return


# Inner Triton functions shared by flex_attention & split-k decoding kernels.
compute_next_offset_func = r"""
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset
"""

get_bounded_indices_func = r"""
@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""


load_checked_block = r"""
@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
"""

load_checked_2d = r"""
@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
"""

compute_sparse_mask_kernel_compact = r"""
{{def_kernel("SPARSE_MASK", "Q_OFFSETS", "FLAT_TO_ROW", "FLAT_TO_BLK", "KV_NUM_BLKS", "KV_IDX")}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}

    TOTAL_ENTRIES : tl.constexpr = TOTAL_FLAT_ENTRIES * NUM_Q_SUB_BLOCKS * NUM_KV_SUB_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        kv_sub = entry_idx % NUM_KV_SUB_BLOCKS
        tmp = entry_idx // NUM_KV_SUB_BLOCKS
        q_sub = tmp % NUM_Q_SUB_BLOCKS
        flat_blk = tmp // NUM_Q_SUB_BLOCKS

        flat_row = tl.load(FLAT_TO_ROW + flat_blk)
        blk_pos = tl.load(FLAT_TO_BLK + flat_blk)
        sq_idx = flat_row % NUM_SPARSE_Q_BLOCKS
        tmp_row = flat_row // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp_row % SPARSE_HQ
        sparse_z = tmp_row // SPARSE_HQ

        q_offset_idx = sparse_z * SPARSE_HQ * (NUM_SPARSE_Q_BLOCKS + 1) + sparse_h * (NUM_SPARSE_Q_BLOCKS + 1) + sq_idx
        expected_flat_blk = tl.load(Q_OFFSETS + q_offset_idx) + blk_pos

        idx_offset = (
            sparse_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + sq_idx * stride_kv_idx_m
            + blk_pos * stride_kv_idx_blk
        )
        kv_block = tl.load(KV_IDX + idx_offset)

        q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE

        offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
        offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)

        m = offs_m[:, None]
        n = offs_n[None, :]
        off_z = sparse_z
        off_h = sparse_h

        {{ modification(
            subgraph_number=0,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        store_mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
        mask_mod_output = mask_mod_output & store_mask
        mask_base = SPARSE_MASK + expected_flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local[:, None] * SPARSE_MASK_STRIDE_M + offs_n_local[None, :]
        tl.store(mask_base + mask_offsets, mask_mod_output.to(tl.int8), mask=store_mask)
"""


compute_bwd_sparse_mask_kernel_compact = r"""
{{def_kernel("Q_OFFSETS", "FLAT_TO_ROW", "FLAT_TO_BLK", "KV_NUM_BLKS", "KV_IDX")}}
    SPARSE_MASK = arg_SPARSE_MASK
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}

    TOTAL_ENTRIES : tl.constexpr = TOTAL_FLAT_ENTRIES * NUM_Q_SUB_BLOCKS * NUM_KV_SUB_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        kv_sub = entry_idx % NUM_KV_SUB_BLOCKS
        tmp = entry_idx // NUM_KV_SUB_BLOCKS
        q_sub = tmp % NUM_Q_SUB_BLOCKS
        flat_blk = tmp // NUM_Q_SUB_BLOCKS

        flat_row = tl.load(FLAT_TO_ROW + flat_blk)
        blk_pos = tl.load(FLAT_TO_BLK + flat_blk)
        sq_idx = flat_row % NUM_SPARSE_Q_BLOCKS
        tmp_row = flat_row // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp_row % SPARSE_HQ
        sparse_z = tmp_row // SPARSE_HQ

        q_offset_idx = sparse_z * SPARSE_HQ * (NUM_SPARSE_Q_BLOCKS + 1) + sparse_h * (NUM_SPARSE_Q_BLOCKS + 1) + sq_idx
        expected_flat_blk = tl.load(Q_OFFSETS + q_offset_idx) + blk_pos

        idx_offset = (
            sparse_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + sq_idx * stride_kv_idx_m
            + blk_pos * stride_kv_idx_blk
        )
        kv_block = tl.load(KV_IDX + idx_offset)

        q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE

        offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
        offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)

        m = offs_m[:, None]
        n = offs_n[None, :]
        off_z = sparse_z
        off_h = sparse_h

        {{ modification(
            subgraph_number=0,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        store_mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
        mask_mod_output = mask_mod_output & store_mask
        mask_base = SPARSE_MASK + expected_flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local[:, None] * SPARSE_MASK_STRIDE_M + offs_n_local[None, :]
        tl.store(mask_base + mask_offsets, mask_mod_output, mask=store_mask)
"""

compute_sparse_mask_block_pos_kernel = r"""
{{def_kernel("KV_NUM_BLKS", "KV_IDX")}}
    SPARSE_MASK_BLOCK_POS = arg_SPARSE_MASK_BLOCK_POS
    stride_kv_num_blks_z = {{stride("KV_NUM_BLKS", 0)}}
    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_num_blks_m = {{stride("KV_NUM_BLKS", 2)}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
    stride_block_pos_z = SPARSE_MASK_BLOCK_POS_STRIDE_Z
    stride_block_pos_h = SPARSE_MASK_BLOCK_POS_STRIDE_H
    stride_block_pos_q = SPARSE_MASK_BLOCK_POS_STRIDE_Q

    TOTAL_ENTRIES : tl.constexpr = SPARSE_Z * SPARSE_HQ * NUM_SPARSE_Q_BLOCKS * MAX_NORMAL_BLOCKS
    TOTAL_POSITIONS : tl.constexpr = SPARSE_Z * SPARSE_HQ * NUM_SPARSE_Q_BLOCKS * NUM_SPARSE_KV_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for pos_idx in range(pid, TOTAL_POSITIONS, num_programs):
        kv_block = pos_idx % NUM_SPARSE_KV_BLOCKS
        tmp = pos_idx // NUM_SPARSE_KV_BLOCKS
        sq_idx = tmp % NUM_SPARSE_Q_BLOCKS
        tmp = tmp // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp % SPARSE_HQ
        sparse_z = tmp // SPARSE_HQ

        block_pos_offset = (
            sparse_z * stride_block_pos_z
            + sparse_h * stride_block_pos_h
            + sq_idx * stride_block_pos_q
            + kv_block
        )
        tl.store(SPARSE_MASK_BLOCK_POS + block_pos_offset, -1)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        blk_pos = entry_idx % MAX_NORMAL_BLOCKS
        tmp = entry_idx // MAX_NORMAL_BLOCKS
        sq_idx = tmp % NUM_SPARSE_Q_BLOCKS
        tmp = tmp // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp % SPARSE_HQ
        sparse_z = tmp // SPARSE_HQ

        nb_offset = (
            sparse_z * stride_kv_num_blks_z
            + sparse_h * stride_kv_num_blks_h
            + sq_idx * stride_kv_num_blks_m
        )
        num_blks = tl.load(KV_NUM_BLKS + nb_offset)

        if blk_pos < num_blks:
            idx_offset = (
                sparse_z * stride_kv_idx_z
                + sparse_h * stride_kv_idx_h
                + sq_idx * stride_kv_idx_m
                + blk_pos * stride_kv_idx_blk
            )
            kv_block = tl.load(KV_IDX + idx_offset)
            block_pos_offset = (
                sparse_z * stride_block_pos_z
                + sparse_h * stride_block_pos_h
                + sq_idx * stride_block_pos_q
                + kv_block
            )
            tl.store(SPARSE_MASK_BLOCK_POS + block_pos_offset, blk_pos)
"""


compute_forward_block_mn_sparse_mask = r"""
@triton.jit
def forward_block_mn_sparse_mask(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION,
    q_start,
    blk_idx_in_list,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}
    # -- load k --
    k = tl.load(K_block_ptr)
    # -- compute qk ---
    qk = tl.dot(q, k)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if not IS_FULL_BLOCKS:
        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        SPARSE_HQ: tl.constexpr = {{size("KV_NUM_BLKS", 1)}}
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        q_sparse_start = q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        sparse_h = off_h % SPARSE_HQ
        sparse_mask_h = off_h % SPARSE_MASK_HQ

        stride_kv_idx_z = {{stride("KV_IDX", 0)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}
        stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
        kv_block = tl.load(
            arg_KV_IDX
            + off_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + q_sparse_idx * stride_kv_idx_m
            + blk_idx_in_list * stride_kv_idx_blk
        )

        offs_m_local = offs_m - q_sparse_start
        offs_n_local = offs_n - kv_block * SPARSE_KV_BLOCK_SIZE
        q_offsets_idx = (
            off_z * SPARSE_MASK_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
            + sparse_mask_h * (NUM_SPARSE_Q_BLOCKS + 1)
            + q_sparse_idx
        )
        flat_blk = tl.load(arg_Q_OFFSETS + q_offsets_idx) + blk_idx_in_list
        mask_base = arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
        mask_bounds = (offs_m < Q_LEN) & (offs_n < KV_LEN)
        mask_mod_output = tl.load(mask_base + mask_offsets, mask=mask_bounds, other=0) != 0

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = mask_mod_output & (offs_n < KV_LEN)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    elif CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    # -- compute scaling constant ---
    m_ij = tl.maximum(
        m_i,
        tl.max(post_mod_scores, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp(m_i - m_ij_masked)
    p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc)
    # NPU compile hint for performance optimization
    if ENABLE_COMPILE_HINT:
        tl.extra.cann.extension.compile_hint(acc, "hivm.tile_mix_cube_num", 2)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

"""

compute_forward_inner_sparse_mask_direct_index = r"""
@triton.jit
def forward_inner_sparse_mask_direct_index(
    {{gen_argdefs()}},
    q, K, V, Q_LEN, KV_LEN,
    stride_kk, stride_kn, stride_vn, stride_vk,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    off_z, off_h, offs_m,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    q_start,
    IS_FULL_BLOCKS,
):
    {{gen_defines() | indent_except_first(1)}}

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    if PRESCALE_QK:
        q = (q * SM_SCALE).to(MATMUL_PRECISION)

    for start_n in range(block_n_start, block_n_end):
        blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
        kv_block = tl.load(kv_indices + blk_idx_in_list)
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0),
        )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn_sparse_mask(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n[None, :],
                MATMUL_PRECISION,
                q_start,
                blk_idx_in_list,
                IS_FULL_BLOCKS,
            )
        else:
            acc, l_i, m_i = forward_block_mn_sparse_mask(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n[None, :],
                MATMUL_PRECISION,
                q_start,
                blk_idx_in_list,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )

    return acc, l_i, m_i

"""


compute_flex_attention_sparse_mask_normal_in_loop_no_load_balance = r"""
{{def_kernel("Q", "K", "V", "SPARSE_MASK", "Q_OFFSETS", "KV_NUM_BLKS", "KV_IDX", "INTERMEDIATE_ACC", "INTERMEDIATE_L", "INTERMEDIATE_M")}}
    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    for tile_id in range(tl.program_id(0), NUM_SPARSE_Q_BLOCKS * ZQ * HQ, tl.num_programs(0)):
        q_start = tile_id // (ZQ * HQ)
        off_zh = tile_id % (ZQ * HQ)
        off_zq = off_zh // HQ
        off_hq = off_zh % HQ
        off_zkv = off_zq % ZKV
        off_hkv = off_hq // GQA_SHARED_HEADS

        Q_tile = Q + off_zq * stride_qz + off_hq * stride_qh
        K_tile = K + off_zkv * stride_kz + off_hkv * stride_kh
        V_tile = V + off_zkv * stride_vz + off_hkv * stride_vh

        SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
        SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_idx_hq = off_hq % SPARSE_HQ

        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

        stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE

        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_sparse_idx
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + q_sparse_idx * stride_kv_idx_m

        Q_block_ptr = tl.make_block_ptr(
            base=Q_tile,
            shape=(Q_LEN, QK_HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(q_start * BLOCK_M, 0),
            block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
            order=(1, 0),
        )
        q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(
            kv_num_blocks * SPARSE_KV_MULTIPLE,
            tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )

        acc, l_i, m_i = forward_inner_sparse_mask_direct_index(
            {{gen_argdefs()}},
            q, K_tile, V_tile, Q_LEN, KV_LEN,
            stride_kk, stride_kn, stride_vn, stride_vk,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            q_start,
            IS_FULL_BLOCKS=False,
        )

        off_hz = off_zq * HQ + off_hq
        offs_m_local = offs_m - q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)
        acc_base = (
            arg_INTERMEDIATE_ACC
            + off_hz * NUM_SPARSE_Q_BLOCKS * SPARSE_Q_BLOCK_SIZE * V_HEAD_DIM_ROUNDED
            + q_sparse_idx * SPARSE_Q_BLOCK_SIZE * V_HEAD_DIM_ROUNDED
        )
        tl.store(
            acc_base + offs_m_local[:, None] * V_HEAD_DIM_ROUNDED + idx_d[None, :],
            acc,
            mask=(offs_m[:, None] < Q_LEN) & (idx_d[None, :] < V_HEAD_DIM),
        )
        state_base = (
            off_hz * NUM_SPARSE_Q_BLOCKS * SPARSE_Q_BLOCK_SIZE
            + q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        )
        tl.store(arg_INTERMEDIATE_L + state_base + offs_m_local, l_i, mask=offs_m < Q_LEN)
        tl.store(arg_INTERMEDIATE_M + state_base + offs_m_local, m_i, mask=offs_m < Q_LEN)
"""

compute_flex_attention_sparse_mask_full_128_in_loop_no_load_balance = r"""
{{def_kernel("Q", "K", "V", "LSE", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "INTERMEDIATE_ACC", "INTERMEDIATE_L", "INTERMEDIATE_M")}}
    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    for tile_id in range(tl.program_id(0), NUM_SPARSE_Q_BLOCKS * ZQ * HQ, tl.num_programs(0)):
        q_start = tile_id // (ZQ * HQ)
        off_zh = tile_id % (ZQ * HQ)
        off_zq = off_zh // HQ
        off_hq = off_zh % HQ
        off_zkv = off_zq % ZKV
        off_hkv = off_hq // GQA_SHARED_HEADS

        Q_tile = Q + off_zq * stride_qz + off_hq * stride_qh
        K_tile = K + off_zkv * stride_kz + off_hkv * stride_kh
        V_tile = V + off_zkv * stride_vz + off_hkv * stride_vh

        SPARSE_Z = {{size("FULL_KV_NUM_BLKS", 0)}}
        SPARSE_HQ = {{size("FULL_KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_idx_hq = off_hq % SPARSE_HQ

        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
        FULL128_SUBTILES: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE

        stride_full_kv_num_blks_h = {{stride("FULL_KV_NUM_BLKS", 1)}}
        stride_full_kv_idx_h = {{stride("FULL_KV_IDX", 1)}}
        stride_full_kv_idx_m = {{stride("FULL_KV_IDX", 2)}}

        off_hz = off_zq * HQ + off_hq
        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_local = offs_m - q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)
        acc_base = (
            arg_INTERMEDIATE_ACC
            + off_hz * NUM_SPARSE_Q_BLOCKS * SPARSE_Q_BLOCK_SIZE * V_HEAD_DIM_ROUNDED
            + q_sparse_idx * SPARSE_Q_BLOCK_SIZE * V_HEAD_DIM_ROUNDED
        )
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            acc = tl.load(
                acc_base + offs_m_local[:, None] * V_HEAD_DIM_ROUNDED + idx_d[None, :]
            )
        else:
            acc = tl.load(
                acc_base + offs_m_local[:, None] * V_HEAD_DIM_ROUNDED + idx_d[None, :],
                mask=(offs_m[:, None] < Q_LEN) & (idx_d[None, :] < V_HEAD_DIM),
                other=0.0,
            )
        state_base = (
            off_hz * NUM_SPARSE_Q_BLOCKS * SPARSE_Q_BLOCK_SIZE
            + q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        )
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            l_i = tl.load(arg_INTERMEDIATE_L + state_base + offs_m_local)
            m_i = tl.load(arg_INTERMEDIATE_M + state_base + offs_m_local)
        else:
            l_i = tl.load(arg_INTERMEDIATE_L + state_base + offs_m_local, mask=offs_m < Q_LEN, other=0.0)
            m_i = tl.load(arg_INTERMEDIATE_M + state_base + offs_m_local, mask=offs_m < Q_LEN, other=float("-inf"))

        full_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
        full_kv_num_blks_offset = full_hz_offset * stride_full_kv_num_blks_h + q_sparse_idx
        full_kv_idx_offset = full_hz_offset * stride_full_kv_idx_h + q_sparse_idx * stride_full_kv_idx_m
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + full_kv_num_blks_offset)

        if kv_num_blocks > 0:
            kv_indices = FULL_KV_IDX + full_kv_idx_offset

            Q_block_ptr = tl.make_block_ptr(
                base=Q_tile,
                shape=(Q_LEN, QK_HEAD_DIM),
                strides=(stride_qm, stride_qk),
                offsets=(q_start * BLOCK_M, 0),
                block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
                order=(1, 0),
            )
            q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)

            for start_n in range(0, kv_num_blocks):
                kv_block_start = tl.load(kv_indices + start_n) * SPARSE_KV_BLOCK_SIZE
                for sub_idx in range(0, FULL128_SUBTILES):
                    kv_start = kv_block_start + sub_idx * BLOCK_N
                    K_block_ptr = tl.make_block_ptr(
                        base=K_tile,
                        shape=(KV_LEN, QK_HEAD_DIM),
                        strides=(stride_kn, stride_kk),
                        offsets=(kv_start, 0),
                        block_shape=(BLOCK_N, QK_HEAD_DIM_ROUNDED),
                        order=(1, 0),
                    )
                    V_block_ptr = tl.make_block_ptr(
                        base=V_tile,
                        shape=(KV_LEN, V_HEAD_DIM),
                        strides=(stride_vn, stride_vk),
                        offsets=(kv_start, 0),
                        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
                        order=(1, 0),
                    )
                    offs_n = kv_start + tl.arange(0, BLOCK_N)

                    if IS_DIVISIBLE:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                        )
                    else:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                            CHECK_BLOCK_BOUNDARY=True,
                        )

        l_i = tl.where(l_i == 0.0, 1, l_i)
        acc = acc / l_i[:, None]
        idx_zq = off_zq
        idx_hq = off_hq
        idx_m = offs_m[:, None]
        idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]
        mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

        {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask", indent_width=8)}}

        if OUTPUT_LOGSUMEXP:
            l_ptrs = LSE + off_hz * Q_LEN + offs_m
            lse = m_i + tl.math.log(l_i)
            if IS_DIVISIBLE:
                tl.store(l_ptrs, lse)
            else:
                tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
"""

compute_flex_attention_sparse_mask_in_loop_no_load_balance = r"""
{{def_kernel("Q", "K", "V", "SPARSE_MASK", "Q_OFFSETS", "KV_NUM_BLKS", "KV_IDX", "LSE", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)
    tl.static_assert(SPARSE_Q_BLOCK_SIZE == BLOCK_M)

    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    for tile_id in range(tl.program_id(0), NUM_SPARSE_Q_BLOCKS * ZQ * HQ, tl.num_programs(0)):
        q_start = tile_id % NUM_SPARSE_Q_BLOCKS
        off_zh = tile_id // NUM_SPARSE_Q_BLOCKS
        off_zq = off_zh // HQ
        off_hq = off_zh % HQ
        off_zkv = off_zq % ZKV
        off_hkv = off_hq // GQA_SHARED_HEADS

        Q_tile = Q + off_zq * stride_qz + off_hq * stride_qh
        K_tile = K + off_zkv * stride_kz + off_hkv * stride_kh
        V_tile = V + off_zkv * stride_vz + off_hkv * stride_vh

        SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
        SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_idx_hq = off_hq % SPARSE_HQ

        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
        FULL128_SUBTILES: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

        stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE

        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_sparse_idx
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + q_sparse_idx * stride_kv_idx_m

        Q_block_ptr = tl.make_block_ptr(
            base=Q_tile,
            shape=(Q_LEN, QK_HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(q_start * BLOCK_M, 0),
            block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
            order=(1, 0),
        )
        q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(
            kv_num_blocks * SPARSE_KV_MULTIPLE,
            tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )

        acc, l_i, m_i = forward_inner_sparse_mask_direct_index(
            {{gen_argdefs()}},
            q, K_tile, V_tile, Q_LEN, KV_LEN,
            stride_kk, stride_kn, stride_vn, stride_vk,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            q_start,
            IS_FULL_BLOCKS=False,
        )

        FULL_SPARSE_Z = {{size("FULL_KV_NUM_BLKS", 0)}}
        FULL_SPARSE_HQ = {{size("FULL_KV_NUM_BLKS", 1)}}
        full_sparse_idx_z = off_zq % FULL_SPARSE_Z
        full_sparse_idx_hq = off_hq % FULL_SPARSE_HQ

        stride_full_kv_num_blks_h = {{stride("FULL_KV_NUM_BLKS", 1)}}
        stride_full_kv_idx_h = {{stride("FULL_KV_IDX", 1)}}
        stride_full_kv_idx_m = {{stride("FULL_KV_IDX", 2)}}

        full_hz_offset = full_sparse_idx_z * FULL_SPARSE_HQ + full_sparse_idx_hq
        full_kv_num_blks_offset = full_hz_offset * stride_full_kv_num_blks_h + q_sparse_idx
        full_kv_idx_offset = full_hz_offset * stride_full_kv_idx_h + q_sparse_idx * stride_full_kv_idx_m
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + full_kv_num_blks_offset)

        if kv_num_blocks > 0:
            kv_indices = FULL_KV_IDX + full_kv_idx_offset

            for start_n in range(0, kv_num_blocks):
                kv_block_start = tl.load(kv_indices + start_n) * SPARSE_KV_BLOCK_SIZE
                for sub_idx in range(0, FULL128_SUBTILES):
                    kv_start = kv_block_start + sub_idx * BLOCK_N
                    K_block_ptr = tl.make_block_ptr(
                        base=K_tile,
                        shape=(KV_LEN, QK_HEAD_DIM),
                        strides=(stride_kn, stride_kk),
                        offsets=(kv_start, 0),
                        block_shape=(BLOCK_N, QK_HEAD_DIM_ROUNDED),
                        order=(1, 0),
                    )
                    V_block_ptr = tl.make_block_ptr(
                        base=V_tile,
                        shape=(KV_LEN, V_HEAD_DIM),
                        strides=(stride_vn, stride_vk),
                        offsets=(kv_start, 0),
                        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
                        order=(1, 0),
                    )
                    offs_n = kv_start + tl.arange(0, BLOCK_N)

                    if IS_DIVISIBLE:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                        )
                    else:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                            CHECK_BLOCK_BOUNDARY=True,
                        )

        l_i = tl.where(l_i == 0.0, 1, l_i)
        acc = acc / l_i[:, None]
        idx_zq = off_zq
        idx_hq = off_hq
        idx_m = offs_m[:, None]
        idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]
        mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

        {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask", indent_width=8)}}

        if OUTPUT_LOGSUMEXP:
            off_hz = off_zq * HQ + off_hq
            l_ptrs = LSE + off_hz * Q_LEN + offs_m
            lse = m_i + tl.math.log(l_i)
            if IS_DIVISIBLE:
                tl.store(l_ptrs, lse)
            else:
                tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
"""

compute_forward_block_mn_full = r"""
@triton.jit
def forward_block_mn_full(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    acc, l_i, m_i,
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION,
    CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1)}}
    k = load_checked_block(K_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    qk = tl.dot(q, tl.trans(k))
    if not PRESCALE_QK:
        qk *= SM_SCALE

    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    m_ij = tl.maximum(
        m_i,
        tl.max(post_mod_scores, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp(m_i - m_ij_masked)
    p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc)
    if ENABLE_COMPILE_HINT:
        tl.extra.cann.extension.compile_hint(acc, "hivm.tile_mix_cube_num", 2)
    m_i = m_ij
    return acc, l_i, m_i

"""


@SymbolicGridFn
def flex_attention_in_loop_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    num_m_blocks = cdiv(num_queries, meta["BLOCK_M"])
    total_tiles = num_m_blocks * batch_size * q_heads
    return (min(total_tiles, meta["NUM_CUBE_CORE"]), 1, 1)


@SymbolicGridFn
def sparse_mask_grid(*args, **kwargs):
    """Compute grid for sparse mask materialization kernel."""
    meta = kwargs.get("meta")
    if meta is None:
        meta = args[-1]
    if "TOTAL_FLAT_ENTRIES" in meta:
        total_entries = (
            meta["TOTAL_FLAT_ENTRIES"]
            * meta["NUM_Q_SUB_BLOCKS"]
            * meta["NUM_KV_SUB_BLOCKS"]
        )
    else:
        total_entries = (
            meta["SPARSE_Z"]
            * meta["SPARSE_HQ"]
            * meta["NUM_SPARSE_Q_BLOCKS"]
            * meta["NUM_Q_SUB_BLOCKS"]
            * meta["MAX_NORMAL_BLOCKS"]
            * meta["NUM_KV_SUB_BLOCKS"]
        )
    num_vector_cores = 48
    return (min(total_entries, num_vector_cores), 1, 1)

del TritonTemplate.all_templates["flex_attention"]
del TritonTemplate.all_templates["flex_attention_backward"]

flex_attention_sparse_mask_normal_template_in_loop_no_load_balance = NPUTritonTemplate(
    name="flex_attention_sparse_mask_normal_in_loop_no_load_balance",
    grid=flex_attention_in_loop_grid,
    source=compute_flex_attention_sparse_mask_normal_in_loop_no_load_balance
    + compute_forward_inner_sparse_mask_direct_index
    + compute_forward_block_mn_sparse_mask
    + load_checked_block
    + get_bounded_indices_func,
)

flex_attention_sparse_mask_full_128_template_in_loop_no_load_balance = NPUTritonTemplate(
    name="flex_attention_sparse_mask_full_128_in_loop_no_load_balance",
    grid=flex_attention_in_loop_grid,
    source=compute_flex_attention_sparse_mask_full_128_in_loop_no_load_balance
    + compute_forward_block_mn_full
    + load_checked_block
    + get_bounded_indices_func,
)

flex_attention_sparse_mask_template_in_loop_no_load_balance = NPUTritonTemplate(
    name="flex_attention_sparse_mask_in_loop_no_load_balance",
    grid=flex_attention_in_loop_grid,
    source=compute_flex_attention_sparse_mask_in_loop_no_load_balance
    + compute_forward_inner_sparse_mask_direct_index
    + compute_forward_block_mn_sparse_mask
    + compute_forward_block_mn_full
    + load_checked_block
    + get_bounded_indices_func,
)

sparse_mask_kernel_compact_template = NPUTritonTemplate(
    name="sparse_mask_kernel_compact",
    grid=sparse_mask_grid,
    source=compute_sparse_mask_kernel_compact,
)

bwd_sparse_mask_kernel_compact_template = NPUTritonTemplate(
    name="bwd_sparse_mask_kernel_compact",
    grid=sparse_mask_grid,
    source=compute_bwd_sparse_mask_kernel_compact,
    manual_output_buffer="arg_SPARSE_MASK",
)

sparse_mask_block_pos_template = NPUTritonTemplate(
    name="sparse_mask_block_pos",
    grid=sparse_mask_grid,
    source=compute_sparse_mask_block_pos_kernel,
    manual_output_buffer="arg_SPARSE_MASK_BLOCK_POS",
)


def _get_num_cube_core() -> int:
    return max(int(getattr(npu_config, "num_cube_core", 1)), 1)


def _collect_subgraph_read_names(subgraph_buffer):
    read_names = set()

    def collect(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif hasattr(value, "get_read_names"):
            read_names.update(value.get_read_names())

    collect(subgraph_buffer)
    return read_names


def _filter_used_subgraph_buffers(subgraph_buffer, other_buffers):
    read_names = _collect_subgraph_read_names(subgraph_buffer)
    if not read_names:
        return list(other_buffers)

    used_buffers = []
    unused_buffer_names = []
    for buffer in other_buffers:
        get_name = getattr(buffer, "get_name", None)
        buffer_name = None
        if get_name is not None:
            try:
                buffer_name = get_name()
            except (AssertionError, NotImplementedError):
                buffer_name = None

        if buffer_name is None or buffer_name in read_names:
            used_buffers.append(buffer)
        else:
            unused_buffer_names.append(buffer_name)

    if unused_buffer_names:
        log.info(
            "Filtered %d unused mask_mod buffers from mask kernel autotune inputs: %s",
            len(unused_buffer_names),
            unused_buffer_names,
        )

    return used_buffers


def _build_persistent_bwd_launch_meta(
    batch_size_hint: int,
    kv_heads_hint: int,
    num_key_value_hint: int,
    block_n1: int,
) -> Dict[str, Union[int, bool]]:
    num_kv_blocks = (num_key_value_hint + block_n1 - 1) // block_n1
    num_tasks = num_kv_blocks * batch_size_hint * kv_heads_hint
    num_cube_core = max(int(npu_config.num_cube_core), 1)
    launch_programs = max(min(num_tasks, num_cube_core), 1)
    tasks_per_program = (num_tasks + launch_programs - 1) // launch_programs

    log.info(
        "[persistent-bwd] Computing launch meta with BLOCK_N1=%d: "
        "NUM_KV_BLOCKS=%d (%d/%d), NUM_TASKS=%d (%d*%d*%d), "
        "LAUNCH_PROGRAMS=%d, TASKS_PER_PROGRAM=%d",
        block_n1,
        num_kv_blocks,
        num_key_value_hint,
        block_n1,
        num_tasks,
        num_kv_blocks,
        batch_size_hint,
        kv_heads_hint,
        launch_programs,
        tasks_per_program,
    )

    return {
        "PERSISTENT_MODE": True,
        "NUM_TASKS": num_tasks,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "LAUNCH_PROGRAMS": launch_programs,
        "TASKS_PER_PROGRAM": tasks_per_program,
    }


def _build_qmajor_dq_launch_meta(
    batch_size_hint: int,
    q_heads_hint: int,
    num_queries_hint: int,
    block_m: int,
) -> dict[str, int]:
    num_q_blocks = (num_queries_hint + block_m - 1) // block_m
    num_tasks = num_q_blocks * batch_size_hint * q_heads_hint
    num_cube_core = max(int(npu_config.num_cube_core), 1)
    launch_programs = max(min(num_tasks, num_cube_core), 1)

    log.info(
        "[qmajor-dq-bwd] Computing launch meta with BLOCK_M2=%d: "
        "DQ_NUM_Q_BLOCKS=%d (%d/%d), DQ_NUM_TASKS=%d (%d*%d*%d), "
        "DQ_LAUNCH_PROGRAMS=%d",
        block_m,
        num_q_blocks,
        num_queries_hint,
        block_m,
        num_tasks,
        num_q_blocks,
        batch_size_hint,
        q_heads_hint,
        launch_programs,
    )

    return {
        "DQ_NUM_Q_BLOCKS": num_q_blocks,
        "DQ_NUM_TASKS": num_tasks,
        "DQ_LAUNCH_PROGRAMS": launch_programs,
    }


@SymbolicGridFn
def flex_attention_backward_dq_grid(
    batch_size, q_heads, num_queries, qk_head_dim, kv_heads, num_key_value, meta, *, cdiv
):
    return (meta["DQ_LAUNCH_PROGRAMS"], 1, 1)


@SymbolicGridFn
def flex_attention_backward_dkdv_grid(
    batch_size, q_heads, num_queries, qk_head_dim, kv_heads, num_key_value, meta, *, cdiv
):
    return (meta["LAUNCH_PROGRAMS"], 1, 1)


flex_attention_backward_qmajor_dq_source = r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ", "SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}
    stride_dqz, stride_dqh, stride_dqm, stride_dqd = {{stride("DQ")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}
    MATMUL_PRECISION = Q.dtype.element_ty

    tl.static_assert(BLOCK_M2 == SPARSE_Q_BLOCK_SIZE)
    tl.static_assert(BLOCK_N2 == SPARSE_KV_BLOCK_SIZE)

    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    stride_kv_num_blks_z = {{stride("KV_NUM_BLKS", 0)}}
    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_num_blks_m = {{stride("KV_NUM_BLKS", 2)}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
    stride_full_kv_num_blks_z = {{stride("FULL_KV_NUM_BLKS", 0)}}
    stride_full_kv_num_blks_h = {{stride("FULL_KV_NUM_BLKS", 1)}}
    stride_full_kv_num_blks_m = {{stride("FULL_KV_NUM_BLKS", 2)}}
    stride_full_kv_idx_z = {{stride("FULL_KV_IDX", 0)}}
    stride_full_kv_idx_h = {{stride("FULL_KV_IDX", 1)}}
    stride_full_kv_idx_m = {{stride("FULL_KV_IDX", 2)}}
    stride_full_kv_idx_blk = {{stride("FULL_KV_IDX", 3)}}

    for task_id in range(pid, DQ_NUM_TASKS, num_core):
        q_block = task_id % DQ_NUM_Q_BLOCKS
        off_zq = (task_id // DQ_NUM_Q_BLOCKS) // HQ
        off_hq = (task_id // DQ_NUM_Q_BLOCKS) % HQ
        off_hkv = off_hq // GQA_SHARED_HEADS
        off_zkv = off_zq % ZKV
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_h = off_hq % SPARSE_HQ
        sparse_mask_h = off_hq % SPARSE_MASK_HQ

        q_start = q_block * BLOCK_M2
        offs_m = q_start + tl.arange(0, BLOCK_M2)

        q_base = Q + stride_qz * off_zq + stride_qh * off_hq
        k_base = K + stride_kz * off_zkv + stride_kh * off_hkv
        v_base = V + stride_vz * off_zkv + stride_vh * off_hkv
        do_base = DO + stride_doz * off_zq + stride_doh * off_hq
        dq_base = DQ + stride_dqz * off_zq + stride_dqh * off_hq
        off_chz = ((off_zq * HQ + off_hq) * Q_LEN).to(tl.int64)
        lse_base = LSE + off_chz
        delta_base = DELTA + off_chz

        q = load_checked_2d(Q + stride_qz * off_zq + stride_qh * off_hq, offs_m, offs_k, stride_qm, stride_qd, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)
        do = load_checked_2d(DO + stride_doz * off_zq + stride_doh * off_hq, offs_m, offs_v, stride_dom, stride_dod, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)
        lse = tl.load(lse_base + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        Di = tl.load(delta_base + offs_m, mask=offs_m < Q_LEN, other=0.0)
        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        kv_num_offset = (
            sparse_idx_z * stride_kv_num_blks_z
            + sparse_h * stride_kv_num_blks_h
            + q_block * stride_kv_num_blks_m
        )
        kv_idx_offset = (
            sparse_idx_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + q_block * stride_kv_idx_m
        )
        q_offsets_idx = (
            sparse_idx_z * SPARSE_MASK_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
            + sparse_mask_h * (NUM_SPARSE_Q_BLOCKS + 1)
            + q_block
        )
        q_offset_base = tl.load(arg_Q_OFFSETS + q_offsets_idx)

        kv_num_blocks = tl.load(arg_KV_NUM_BLKS + kv_num_offset)
        for blk_pos in range(0, kv_num_blocks):
            kv_sparse_idx = tl.load(arg_KV_IDX + kv_idx_offset + blk_pos * stride_kv_idx_blk)
            offs_n = kv_sparse_idx * SPARSE_KV_BLOCK_SIZE + tl.arange(0, BLOCK_N2)
            k = load_checked_2d(k_base, offs_n, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
            v = load_checked_2d(v_base, offs_n, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
            qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            if not PRESCALE_QK:
                qk *= SM_SCALE

            m = get_bounded_indices(offs_m[:, None], Q_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
            n = get_bounded_indices(offs_n[None, :], KV_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
            flat_blk = q_offset_base + blk_pos
            offs_m_local = offs_m[:, None] - q_block * SPARSE_Q_BLOCK_SIZE
            offs_n_local = offs_n[None, :] - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
            mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
            mask_bounds = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
            mask_mod_output = tl.load(
                arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK + mask_offsets,
                mask=mask_bounds,
                other=0,
            )
{% if BWD_SCORE_MOD_IS_IDENTITY %}
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
            pre_mod_scores = qk
{% endif %}
            qk = tl.where(mask_mod_output & (offs_n[None, :] < KV_LEN), qk, float("-inf"))
            p = tl.math.exp(qk - lse[:, None])
{% else %}
            pre_mod_scores = qk
            {{ modification(
                subgraph_number=0,
                output_name="post_mod_scores",
                score="qk",
                b="off_zq",
                h="off_hq",
                m="m",
                n="n",
                out="qk",
            ) | indent_except_first(3) }}
            post_mod_scores = tl.where(mask_mod_output & (offs_n[None, :] < KV_LEN), post_mod_scores, float("-inf"))
            p = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - Di[:, None])
{% if BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
            ds = tl.where(mask_mod_output, ds, 0.0)
{% else %}
            {{ modification(
                subgraph_number=1,
                output_name="grad_scores",
                score="pre_mod_scores",
                b="off_zq",
                h="off_hq",
                m="m",
                n="n",
                grad_score_mod="ds",
            ) | indent_except_first(3) }}
            ds = tl.where(mask_mod_output, grad_scores, 0.0)
{% endif %}
            dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            full_kv_num_offset = (
                sparse_idx_z * stride_full_kv_num_blks_z
                + sparse_h * stride_full_kv_num_blks_h
                + q_block * stride_full_kv_num_blks_m
            )
            full_kv_idx_offset = (
                sparse_idx_z * stride_full_kv_idx_z
                + sparse_h * stride_full_kv_idx_h
                + q_block * stride_full_kv_idx_m
            )
            full_kv_num_blocks = tl.load(arg_FULL_KV_NUM_BLKS + full_kv_num_offset)
            for blk_pos in range(0, full_kv_num_blocks):
                kv_sparse_idx = tl.load(arg_FULL_KV_IDX + full_kv_idx_offset + blk_pos * stride_full_kv_idx_blk)
                offs_n = kv_sparse_idx * SPARSE_KV_BLOCK_SIZE + tl.arange(0, BLOCK_N2)
                k = load_checked_2d(k_base, offs_n, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
                v = load_checked_2d(v_base, offs_n, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                if not PRESCALE_QK:
                    qk *= SM_SCALE

                m = get_bounded_indices(offs_m[:, None], Q_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
                n = get_bounded_indices(offs_n[None, :], KV_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
{% if BWD_SCORE_MOD_IS_IDENTITY %}
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
                pre_mod_scores = qk
{% endif %}
                qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))
                p = tl.math.exp(qk - lse[:, None])
{% else %}
                pre_mod_scores = qk
                {{ modification(
                    subgraph_number=0,
                    output_name="post_mod_scores",
                    score="qk",
                    b="off_zq",
                    h="off_hq",
                    m="m",
                    n="n",
                    out="qk",
                ) | indent_except_first(4) }}
                post_mod_scores = tl.where(offs_n[None, :] < KV_LEN, post_mod_scores, float("-inf"))
                p = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - Di[:, None])
{% if BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
{% else %}
                {{ modification(
                    subgraph_number=1,
                    output_name="grad_scores",
                    score="pre_mod_scores",
                    b="off_zq",
                    h="off_hq",
                    m="m",
                    n="n",
                    grad_score_mod="ds",
                ) | indent_except_first(4) }}
                grad_scores = tl.where(offs_n[None, :] < KV_LEN, grad_scores, 0.0)
                dq += tl.dot(grad_scores.to(MATMUL_PRECISION), k, input_precision="ieee")
{% endif %}

        dq *= SM_SCALE
        index_m = offs_m[:, None]
        index_k = offs_k[None, :]
        if SAFE_HEAD_DIM:
            dq_mask = index_m < Q_LEN
        else:
            dq_mask = (index_m < Q_LEN) & (index_k < QK_HEAD_DIM)
        tl.store(dq_base + index_m * stride_dqm + index_k * stride_dqd, dq, mask=dq_mask)

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:
        return tl.load(ptr)
"""



flex_attention_backward_dkdv_only_source = r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK", "SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d). Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}

    stride_dvz, stride_dvh, stride_dvm, stride_dvd = {{stride("DV")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_TASKS = NUM_KV_BLOCKS * ZKV * HKV

    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

    stride_q_num_blks_h = {{stride("Q_NUM_BLKS", 1)}}
    stride_q_idx_h = {{stride("Q_IDX", 1)}}
    stride_q_idx_n = {{stride("Q_IDX", 2)}}

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_zq = (task_id // NUM_KV_BLOCKS) // HKV
        off_hkv = (task_id // NUM_KV_BLOCKS) % HKV
        off_zkv = off_zq % ZKV
        sparse_idx_z = off_zq % SPARSE_Z

        start_n1 = kv_start_block * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)
        pid_mask = kv_start_block // SPARSE_KV_MULTIPLE

        k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
        v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
        dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

        K1 = K + k_adj
        V1 = V + v_adj
        DV1 = DV + dv_adj

        k = load_checked_2d(K1, offs_n1, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
        if PRESCALE_QK:
            k = (k * SM_SCALE).to(MATMUL_PRECISION)
        v = load_checked_2d(V1, offs_n1, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1
            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n

            q_indices = Q_IDX + sparse_q_idx_offset
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
            hi = tl.minimum(
                sparse_q_num_blocks * SPARSE_Q_MULTIPLE,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
            )
            for start_m in range(0, hi):
                blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                bwd_dkdv_block_mn(
                    {{gen_argdefs()}},
                    Q1, DO1, DK, DELTA1, LSE1, DV1,
                    k, v, Q_LEN, KV_LEN,
                    off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, q_block, pid_mask, offs_k, offs_v,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                    MATMUL_PRECISION,
                    False, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                )

            if HAS_FULL_BLOCKS:
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                hi = tl.minimum(
                    sparse_q_num_blocks * SPARSE_Q_MULTIPLE,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
                )
                for start_m in range(0, hi):
                    blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                    offs_m1 = q_start + tl.arange(0, BLOCK_M1)
{% if not PRESCALE_QK %}
                    bwd_dkdv_full_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV1,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                        MATMUL_PRECISION,
                        CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
{% else %}
                    bwd_dkdv_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV1,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, q_block, pid_mask, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                        MATMUL_PRECISION,
                        True, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
{% endif %}

@triton.jit
def bwd_dkdv_block_mn(
    {{gen_argdefs()}},
    Q, DO, DK, DELTA, LSE, DV,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n1, offs_m1, start_m1, q_sparse_idx, kv_sparse_idx, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qd),
        offsets=(start_m1, 0),
        block_shape=(BLOCK_M1, QK_HEAD_DIM_ROUNDED),
        order=(1, 0),
    )
    qT = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(qT, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n1[None, :], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

{% if BWD_SCORE_MOD_IS_IDENTITY %}
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    pre_mod_scores = qkT
{% endif %}
    if CHECK_BLOCK_BOUNDARY:
        qkT = tl.where(offs_n1[None, :] < KV_LEN, qkT, float("-inf"))
{% else %}
    pre_mod_scores = qkT
    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qkT",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qkT"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n1[None, :] < KV_LEN, post_mod_scores, float("-inf"))
{% endif %}

    if not IS_FULL_BLOCKS:
        SPARSE_Z: tl.constexpr = {{size("KV_NUM_BLKS", 0)}}
        SPARSE_HQ: tl.constexpr = {{size("KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_z % SPARSE_Z
        sparse_mask_h = off_hq % SPARSE_MASK_HQ
        q_sparse_start = q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        block_pos_offset = (
            sparse_idx_z * {{stride("SPARSE_MASK_BLOCK_POS", 0)}}
            + sparse_mask_h * {{stride("SPARSE_MASK_BLOCK_POS", 1)}}
            + q_sparse_idx * {{stride("SPARSE_MASK_BLOCK_POS", 2)}}
            + kv_sparse_idx
        )
        blk_idx_in_list = tl.load(arg_SPARSE_MASK_BLOCK_POS + block_pos_offset)
        has_matching_kv_block = blk_idx_in_list >= 0
        safe_blk_idx = tl.maximum(blk_idx_in_list, 0)

        offs_m_local = offs_m1[:, None] - q_sparse_start
        offs_n_local = offs_n1[None, :] - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
        q_offsets_idx = (
            sparse_idx_z * SPARSE_MASK_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
            + sparse_mask_h * (NUM_SPARSE_Q_BLOCKS + 1)
            + q_sparse_idx
        )
        flat_blk = tl.load(arg_Q_OFFSETS + q_offsets_idx) + safe_blk_idx
        mask_base = arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
        mask_mod_output = tl.load(
            mask_base + mask_offsets,
        )
        mask_mod_output = mask_mod_output & has_matching_kv_block & (offs_m1[:, None] < Q_LEN)
{% if BWD_SCORE_MOD_IS_IDENTITY %}
        qkT = tl.where(mask_mod_output & (offs_n1[None, :] < KV_LEN), qkT, float("-inf"))
{% else %}
        post_mod_scores = tl.where(
            mask_mod_output & (offs_n1[None, :] < KV_LEN),
            post_mod_scores,
            float("-inf"),
        )
{% endif %}

{% if BWD_SCORE_MOD_IS_IDENTITY %}
    pT = tl.math.exp(qkT - lse[:, None]).to(MATMUL_PRECISION)
{% else %}
    pT = tl.math.exp(post_mod_scores - lse[:, None]).to(MATMUL_PRECISION)
{% endif %}
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Q_LEN, V_HEAD_DIM),
        strides=(stride_dom, stride_dod),
        offsets=(start_m1, 0),
        block_shape=(BLOCK_M1, V_HEAD_DIM_ROUNDED),
        order=(1, 0),
    )
    do = load_checked_block(DO_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    dv = tl.dot(tl.trans(pT.to(MATMUL_PRECISION)), do, input_precision="ieee")
    index_n = offs_n1[:, None]
    index_v = offs_v[None, :]
    dv_ptrs = DV + index_n * stride_dvm + index_v * stride_dvd
    tl.atomic_add(
        dv_ptrs,
        dv,
        mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM),
    )
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN, other=0.0)
    dpT = tl.dot(do, tl.trans(v), input_precision="ieee").to(MATMUL_PRECISION)
    dsT = (pT * (dpT - Di[:, None])).to(MATMUL_PRECISION)
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    {{ modification(
        subgraph_number=1,
        output_name="grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if RUN_CAPTURED_GRADS %}
    idx_b = off_z
    idx_h = off_hq
    idx_m = m
    idx_n = n
    scatter_mask = (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
    {{ modification(
        subgraph_number=3,
        output_name=None,
        mask="scatter_mask",
        score="pre_mod_scores",
        b="idx_b",
        h="idx_h",
        m="idx_m",
        n="idx_n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    dsT = grad_scores
{% endif %}
    if not IS_FULL_BLOCKS:
        dsT = tl.where(mask_mod_output, dsT, 0.0)
{% if not BWD_IDENTITY_SCORE_AND_GRAD %}
    dsT = tl.where(offs_m1[:, None] < Q_LEN, dsT, 0.0)
{% endif %}

    index_k = offs_k[None, :]

    dk = tl.dot(tl.trans(dsT).to(MATMUL_PRECISION), qT, input_precision="ieee")
{% if PRESCALE_QK %}
    dk *= SM_SCALE
{% endif %}
    if SAFE_HEAD_DIM:
        dk_mask = index_n < KV_LEN
    else:
        dk_mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)
    dk_ptrs = DK + tl.broadcast_to(
        index_n * stride_kn + index_k * stride_kd + stride_kh * off_hkv + stride_kz * off_z,
        dk.shape,
    )
    tl.atomic_add(dk_ptrs, dk, mask=dk_mask)
    if ENABLE_COMPILE_HINT:
        tl.extra.cann.extension.compile_hint(dk, "hivm.tile_mix_cube_num", 2)

@triton.jit
def bwd_dkdv_full_block_mn(
    {{gen_argdefs()}},
    Q, DO, DK, DELTA, LSE, DV,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n1, offs_m1, start_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
    MATMUL_PRECISION,
    CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qd),
        offsets=(start_m1, 0),
        block_shape=(BLOCK_M1, QK_HEAD_DIM_ROUNDED),
        order=(1, 0),
    )
    qT = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(qT, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n1[None, :], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

{% if BWD_SCORE_MOD_IS_IDENTITY %}
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    pre_mod_scores = qkT
{% endif %}
    if CHECK_BLOCK_BOUNDARY:
        qkT = tl.where(offs_n1[None, :] < KV_LEN, qkT, float("-inf"))
    pT = tl.math.exp(qkT - lse[:, None]).to(MATMUL_PRECISION)
{% else %}
    pre_mod_scores = qkT
    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qkT",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qkT"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n1[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    pT = tl.math.exp(post_mod_scores - lse[:, None]).to(MATMUL_PRECISION)
{% endif %}
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Q_LEN, V_HEAD_DIM),
        strides=(stride_dom, stride_dod),
        offsets=(start_m1, 0),
        block_shape=(BLOCK_M1, V_HEAD_DIM_ROUNDED),
        order=(1, 0),
    )
    do = load_checked_block(DO_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    index_n = offs_n1[:, None]
    dv = tl.dot(tl.trans(pT.to(MATMUL_PRECISION)), do, input_precision="ieee")
    index_v = offs_v[None, :]
    dv_ptrs = DV + index_n * stride_dvm + index_v * stride_dvd
    tl.atomic_add(
        dv_ptrs,
        dv,
        mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM),
    )
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN, other=0.0)
    dpT = tl.dot(do, tl.trans(v), input_precision="ieee").to(MATMUL_PRECISION)
    dsT = (pT * (dpT - Di[:, None])).to(MATMUL_PRECISION)
{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    {{ modification(
        subgraph_number=1,
        output_name="grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if RUN_CAPTURED_GRADS %}
    idx_b = off_z
    idx_h = off_hq
    idx_m = m
    idx_n = n
    scatter_mask = (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
    {{ modification(
        subgraph_number=3,
        output_name=None,
        mask="scatter_mask",
        score="pre_mod_scores",
        b="idx_b",
        h="idx_h",
        m="idx_m",
        n="idx_n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if not BWD_GRAD_SCORE_MOD_IS_IDENTITY %}
    dsT = grad_scores
{% endif %}
{% if not BWD_IDENTITY_SCORE_AND_GRAD %}
    dsT = tl.where(offs_m1[:, None] < Q_LEN, dsT, 0.0)
{% endif %}
    index_k = offs_k[None, :]

    dk = tl.dot(tl.trans(dsT).to(MATMUL_PRECISION), qT, input_precision="ieee")
{% if PRESCALE_QK %}
    dk *= SM_SCALE
{% endif %}
    if SAFE_HEAD_DIM:
        dk_mask = index_n < KV_LEN
    else:
        dk_mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)
    dk_ptrs = DK + tl.broadcast_to(
        index_n * stride_kn + index_k * stride_kd + stride_kh * off_hkv + stride_kz * off_z,
        dk.shape,
    )
    tl.atomic_add(dk_ptrs, dk, mask=dk_mask)
    if ENABLE_COMPILE_HINT:
        tl.extra.cann.extension.compile_hint(dk, "hivm.tile_mix_cube_num", 2)


@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
"""

flex_attention_backward_qmajor_dq_template = NPUTritonTemplate(
    name="flex_attention_backward_qmajor_dq",
    grid=flex_attention_backward_dq_grid,
    source=flex_attention_backward_qmajor_dq_source,
)

flex_attention_backward_dkdv_only_template = NPUTritonTemplate(
    name="flex_attention_backward_dkdv_only",
    grid=flex_attention_backward_dkdv_grid,
    source=flex_attention_backward_dkdv_only_source,
)


def _register_npu_inductor_flex_attention():
    @register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
    def flex_attention(
        query,
        key,
        value,
        subgraph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ):
        # below is npu path
        (
            _,  # q_length
            _,  # kv_length
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            mask_graph,
        ) = block_mask

        placeholder_inps = [
            create_placeholder(name, dtype, query.get_device())
            for name, dtype in [
                ("score", query.get_dtype()),
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        subgraph_buffer = _build_subgraph_buffer_with_additional_lowerings(
            placeholder_inps + list(score_mod_other_buffers), subgraph
        )

        mask_graph_placeholder_inps = [
            create_placeholder(name, dtype, query.get_device())
            for name, dtype in [
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        mask_graph_buffer = _build_subgraph_buffer_with_additional_lowerings(
            mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
        )

        kernel_options = dict(kernel_options)
        # Mark symbols in custom kernel options as static shapes and add guards.
        kernel_options = {
            k: V.graph.sizevars.evaluate_static_shape(v)
            if isinstance(v, sympy.Symbol)
            else v
            for k, v in kernel_options.items()
        }

        if _use_flex_decoding(query, kernel_options):
            return create_flex_decoding_kernel(
                query,
                key,
                value,
                block_mask,
                scale,
                kernel_options,
                subgraph_buffer,
                mask_graph_buffer,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )

        (
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = maybe_realize(
            [
                query,
                key,
                value,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

        score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
        mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

        Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
        Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
        assert V.graph.sizevars.evaluate_expr(
            sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)
        ), f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        B = Bq

        if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
            kernel_options.setdefault("IS_DIVISIBLE", False)
        else:
            kernel_options.setdefault("IS_DIVISIBLE", True)

        # Reuse query strides for output layout despite different last dimension.
        # This works because only the last dim differs and we check it is contiguous.
        q_strides = query.get_stride()
        assert q_strides[-1] == 1, "Query must be contiguous in the last dimension"

        # Construct output layout with strides matching the query.
        out_size = [B, Hq, seq_len_q, v_head_dim]
        fill_order = get_fill_order(query.get_stride(), V.graph.sizevars.shape_env)
        out_strides = construct_strides(out_size, fill_order)

        layout = FixedLayout(
            query.get_device(),
            query.get_dtype(),
            [B, Hq, seq_len_q, v_head_dim],
            stride=[sympy.sympify(s) for s in out_strides],
        )
        # see NOTE:[TritonTemplates with multiple outputs]
        logsumexp_shape = [B, Hq, seq_len_q]
        logsumexp = empty_strided(
            logsumexp_shape,
            None,
            dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
            device=query.get_device(),
        )
        kernel_options.setdefault("SM_SCALE", scale)

        # Determine GQA broadcast factor.
        gqa_shared_heads = Hq // Hkv
        kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

        compact_q_offsets = None
        compact_flat_to_row = None
        compact_flat_to_blk = None
        compact_tail_end = len(mask_mod_other_buffers)
        compact_tail_start = compact_tail_end - 3
        assert compact_tail_start >= 0, (
            "expected captured compact sparse mask metadata in mask_mod_other_buffers"
        )
        (
            compact_q_offsets,
            compact_flat_to_row,
            compact_flat_to_blk,
        ) = mask_mod_other_buffers[compact_tail_start:compact_tail_end]

        # HAS_FULL_BLOCKS is supplied by the eager create_block_mask patch and cached on
        # the BlockMask, so lowering only consumes it.
        has_full_blocks = bool(kernel_options.get("HAS_FULL_BLOCKS", False))
        kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
        if not has_full_blocks:
            raise RuntimeError(
                "NPU flex attention requires split sparse-mask metadata with full blocks"
            )

        set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)


        # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
        SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)
        SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_Q_BLOCK_SIZE)

        metadata_sparse_hq = kv_num_blocks.get_size()[1]
        num_sparse_q_blocks = kv_num_blocks.get_size()[2]
        metadata_max_normal_blocks = kv_indices.get_size()[3]
        metadata_sparse_hq_val = V.graph.sizevars.evaluate_static_shape(metadata_sparse_hq)
        num_sparse_q_blocks_val = V.graph.sizevars.evaluate_static_shape(num_sparse_q_blocks)
        metadata_max_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(metadata_max_normal_blocks)
        sparse_mask_hq_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options.get("SPARSE_MASK_HQ", metadata_sparse_hq_val)
        )
        sparse_mask_max_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options.get("SPARSE_MASK_MAX_NORMAL_BLOCKS", metadata_max_normal_blocks_val)
        )
        kernel_options.setdefault("SPARSE_MASK_HQ", sparse_mask_hq_val)
        kernel_options.setdefault("SPARSE_MASK_MAX_NORMAL_BLOCKS", sparse_mask_max_normal_blocks_val)
        assert compact_q_offsets is not None
        assert compact_flat_to_row is not None
        assert compact_flat_to_blk is not None
        total_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options[_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION]
        )
        sparse_mask_size = [
            total_normal_blocks_val,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        ]
        sparse_mask_strides = [
            SPARSE_Q_BLOCK_SIZE * SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            1,
        ]
        sparse_mask_layout = FixedLayout(
            query.get_device(),
            torch.int8,
            sparse_mask_size,
            stride=[sympy.sympify(s) for s in sparse_mask_strides],
        )
        sparse_mask_buffer = empty_strided(
            sparse_mask_size,
            sparse_mask_strides,
            dtype=torch.int8,
            device=query.get_device(),
        )
        kernel_options.setdefault("SPARSE_MASK_STRIDE_BLK", sparse_mask_strides[0])
        kernel_options.setdefault("SPARSE_MASK_STRIDE_M", sparse_mask_strides[1])

        kernel_options.setdefault("NUM_SPARSE_Q_BLOCKS", num_sparse_q_blocks_val)

        fwd_call_size_hints = V.graph.sizevars.size_hints(
            query.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        fwd_batch_size_hint, fwd_q_heads_hint, fwd_num_queries_hint, _ = fwd_call_size_hints
        fwd_num_cube_core = _get_num_cube_core()

        log.debug("flex_attention lowering: query=%s key=%s value=%s SPARSE_Q=%s SPARSE_KV=%s kernel_options=%s use_config_generator=%s",
                  query.get_size(), key.get_size(), value.get_size(),
                  SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE, kernel_options,
                  npu_config.flex_attention.use_config_generator)

        # Validate benchmark configuration before autotuning
        log.debug("Benchmark Configuration Validation")
        validate_benchmark_config()  # Now only warns, doesn't raise errors

        choices: list[Any] = []
        forward_input_nodes = [
            query,
            key,
            value,
            sparse_mask_buffer,
            compact_q_offsets,
            kv_num_blocks,
            kv_indices,
            logsumexp,
            full_kv_num_blocks,
            full_kv_indices,
        ]

        log.debug(
            "Config Generation Mode: use_config_generator=%s",
            npu_config.flex_attention.use_config_generator,
        )
        dict_configs = generate_fwd_candidate_configs(
            query_shape=query.get_size(),
            key_shape=key.get_size(),
            dtype=query.get_dtype(),
            sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
            sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            num_cube_core=fwd_num_cube_core,
            head_dim=V.graph.sizevars.evaluate_static_shape(query.get_size()[-1]),
        )

        sparse_mask_split_configs = []
        seen_sparse_mask_split_configs = set()
        for cfg in dict_configs:
            if cfg["BLOCK_M"] != SPARSE_Q_BLOCK_SIZE:
                continue
            split_cfg = copy.deepcopy(cfg)
            split_cfg["BLOCK_N"] = SPARSE_KV_BLOCK_SIZE
            split_cfg["num_stages"] = 1
            split_key = (
                split_cfg["BLOCK_M"],
                split_cfg["BLOCK_N"],
                split_cfg["num_warps"],
                split_cfg["num_stages"],
            )
            if split_key in seen_sparse_mask_split_configs:
                continue
            seen_sparse_mask_split_configs.add(split_key)
            sparse_mask_split_configs.append(split_cfg)

        if not sparse_mask_split_configs and dict_configs:
            split_cfg = copy.deepcopy(dict_configs[0])
            split_cfg["BLOCK_M"] = SPARSE_Q_BLOCK_SIZE
            split_cfg["BLOCK_N"] = SPARSE_KV_BLOCK_SIZE
            split_cfg["num_stages"] = 1
            sparse_mask_split_configs.append(split_cfg)
        dict_configs = sparse_mask_split_configs
        if not dict_configs:
            raise RuntimeError(
                "Sparse mask load balancing requires BLOCK_M to equal "
                f"SPARSE_Q_BLOCK_SIZE={SPARSE_Q_BLOCK_SIZE} and BLOCK_N "
                f"to equal SPARSE_KV_BLOCK_SIZE={SPARSE_KV_BLOCK_SIZE}."
            )

        log.debug("dict_configs count: %d configs: %s", len(dict_configs), dict_configs)

        log.info("Generated %d configs for flex_attention", len(dict_configs))

        # Note, we don't need to pass in the captured buffers explicitly
        # because they're implicitly added by the score_mod function
        # We do need to explicitly pass it in for autotuning though.
        original_kernel_options = kernel_options.copy()
        for cfg in dict_configs:
            BLOCK_M = cfg["BLOCK_M"]
            BLOCK_N = cfg["BLOCK_N"]

            log.debug("Processing config: BLOCK_M=%d BLOCK_N=%d SPARSE_KV%%BLOCK_N=%d SPARSE_Q%%BLOCK_M=%d",
                      BLOCK_M, BLOCK_N, SPARSE_KV_BLOCK_SIZE % BLOCK_N, SPARSE_Q_BLOCK_SIZE % BLOCK_M)

            if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0 or SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0:
                if len(dict_configs) == 1:
                    raise ValueError(
                        f"Q and KV block size must be divisible by BLOCK_M and BLOCK_N. We "
                        f"got Q_BLOCK_SIZE={SPARSE_Q_BLOCK_SIZE} and KV_BLOCK_SIZE={SPARSE_KV_BLOCK_SIZE}."
                    )
                log.debug("Skipping config - block size not divisible")
                continue

            cur_kernel_options = original_kernel_options.copy()
            # Performance tuning
            # Triton parameters
            # Remove prefix for forward kernels options and delete backward kernel options.
            for k in list(cur_kernel_options.keys()):
                if k.startswith("fwd_"):
                    v = cur_kernel_options.pop(k)
                    cur_kernel_options[k[4:]] = v
                if k.startswith("bwd_"):
                    cur_kernel_options.pop(k)

            # Apply all config parameters (BLOCK_M, BLOCK_N, num_warps, num_stages, NPU params)
            for k, v in cfg.items():
                cur_kernel_options.setdefault(k, v)

            # Blocksparse options
            cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
            cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
            cur_kernel_options = apply_kernel_options_from_block_sparse_mask(
                cur_kernel_options,
                kv_num_blocks,
                kv_indices,
                context="fwd",
            )
            cur_kernel_options.setdefault("NUM_CUBE_CORE", fwd_num_cube_core)
            fwd_grid_x = (fwd_num_queries_hint + BLOCK_M - 1) // BLOCK_M
            fwd_grid_y = fwd_batch_size_hint * fwd_q_heads_hint
            fwd_grid_z = 1
            fwd_total_programs = fwd_grid_x * fwd_grid_y * fwd_grid_z
            fwd_wave_estimate = (
                fwd_total_programs + fwd_num_cube_core - 1
            ) // fwd_num_cube_core

            log.debug(
                "fwd-choice: cfg=%s grid=(%d,%d,%d) total_programs=%d aicore_count=%d exceeds_aicores=%s wave_estimate=%d kv_num_blocks_type=%s kv_indices_type=%s final_kernel_options=%s",
                cfg, fwd_grid_x, fwd_grid_y, fwd_grid_z, fwd_total_programs, fwd_num_cube_core,
                fwd_total_programs > fwd_num_cube_core, fwd_wave_estimate,
                type(kv_num_blocks).__name__, type(kv_indices).__name__, cur_kernel_options
            )


            try:
                forward_kernel_options = cur_kernel_options.copy()
                forward_kernel_options["num_stages"] = 2
                choice_count = len(choices)
                forward_errors = []
                for forward_variant_options in sparse_mask_attention_cvpipeline_config_variants(
                    forward_kernel_options,
                    block_n=forward_kernel_options["BLOCK_N"],
                ):
                    log.info(
                        "Appending sparse-mask forward choice BLOCK_M=%d BLOCK_N=%d multibuffer=%s",
                        forward_kernel_options["BLOCK_M"],
                        forward_kernel_options["BLOCK_N"],
                        forward_variant_options.get("multibuffer"),
                    )
                    error = flex_attention_sparse_mask_template_in_loop_no_load_balance.maybe_append_choice(
                        choices=choices,
                        input_nodes=forward_input_nodes,
                        layout=layout,
                        subgraphs=[subgraph_buffer],
                        mutated_inputs=[logsumexp],
                        call_sizes=query.get_size(),
                        **forward_variant_options,
                    )
                    if error is not None:
                        forward_errors.append(error)

                if len(choices) == choice_count:
                    error = forward_errors[0] if forward_errors else "sparse-mask forward choice was not appended"
                    log.warning("Config %s compilation returned error: %s", cfg, error)
                    if len(dict_configs) == 1:
                        if isinstance(error, BaseException):
                            raise error
                        raise RuntimeError(str(error))
                    continue

                _tag_flex_attention_report_choices(
                    choices[choice_count:],
                    "forward",
                    cfg,
                )
            except Exception as e:
                # Catch compilation errors and skip this config
                log.warning("Config %s compilation failed: %s: %s", cfg, type(e).__name__, str(e)[:200])
                # Continue to next config instead of raising
                continue

        sparse_mask_choices = []
        sparse_mask_base_kernel_options = {
            "SPARSE_Z": V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[0]),
            "SPARSE_HQ": kernel_options.get(
                "SPARSE_MASK_HQ",
                V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[1]),
            ),
            "NUM_SPARSE_Q_BLOCKS": V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[2]),
            "MAX_NORMAL_BLOCKS": kernel_options.get(
                "SPARSE_MASK_MAX_NORMAL_BLOCKS",
                V.graph.sizevars.evaluate_static_shape(kv_indices.get_size()[3]),
            ),
            "SPARSE_Q_BLOCK_SIZE": SPARSE_Q_BLOCK_SIZE,
            "SPARSE_KV_BLOCK_SIZE": SPARSE_KV_BLOCK_SIZE,
            "Q_LEN": V.graph.sizevars.evaluate_static_shape(seq_len_q),
            "KV_LEN": V.graph.sizevars.evaluate_static_shape(seq_len_kv),
        }
        sparse_mask_base_kernel_options.update(
            {
                "TOTAL_FLAT_ENTRIES": kernel_options[
                    _COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION
                ],
                "SPARSE_MASK_STRIDE_BLK": sparse_mask_strides[0],
                "SPARSE_MASK_STRIDE_M": sparse_mask_strides[1],
            }
        )

        sparse_mask_tiling_configs = build_sparse_mask_candidate_configs(
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )
        log.info(
            "Generated %d sparse mask kernel tiling configs from "
            "SPARSE_Q_BLOCK_SIZE=%d, SPARSE_KV_BLOCK_SIZE=%d "
            "(multi_tiling_enabled=%s): %s",
            len(sparse_mask_tiling_configs),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            npu_config.flex_attention.use_config_generator,
            sparse_mask_tiling_configs,
        )

        for sparse_mask_tiling_config in sparse_mask_tiling_configs:
            sparse_mask_kernel_options = sparse_mask_base_kernel_options.copy()
            sparse_mask_kernel_options.update(sparse_mask_tiling_config)
            num_choices_before = len(sparse_mask_choices)
            sparse_mask_template = sparse_mask_kernel_compact_template
            sparse_mask_input_nodes = [
                sparse_mask_buffer,
                compact_q_offsets,
                compact_flat_to_row,
                compact_flat_to_blk,
                kv_num_blocks,
                kv_indices,
            ]
            try:
                sparse_mask_template.maybe_append_choice(
                    choices=sparse_mask_choices,
                    input_nodes=sparse_mask_input_nodes,
                    layout=sparse_mask_layout,
                    subgraphs=[mask_graph_buffer],
                    mutated_inputs=[sparse_mask_buffer],
                    call_sizes=sparse_mask_buffer.get_size(),
                    **sparse_mask_kernel_options,
                )
                if len(sparse_mask_choices) > num_choices_before:
                    _tag_choice_configs(
                        sparse_mask_choices[num_choices_before:],
                        "_sparse_mask_report_config",
                        sparse_mask_tiling_config,
                    )
                    if prefer_max_tiling_without_benchmark():
                        _tag_choice_attr(
                            sparse_mask_choices[num_choices_before:],
                            "_nobench_select_first_compilable",
                            True,
                        )
                    log.info(
                        "Sparse mask kernel choice created successfully: %s",
                        sparse_mask_tiling_config,
                    )
                else:
                    log.warning(
                        "Sparse mask kernel choice was not appended for config: %s",
                        sparse_mask_tiling_config,
                    )
            except Exception as e:
                log.warning(
                    "Sparse mask kernel choice creation failed for config %s: %s: %s",
                    sparse_mask_tiling_config,
                    type(e).__name__,
                    str(e)[:200],
                )

        if not sparse_mask_choices:
            raise RuntimeError(
                f"All {len(sparse_mask_tiling_configs)} sparse mask kernel tiling "
                "configs failed to create choices. Cannot proceed with sparse_mask mode."
            )
        log.info(
            "Sparse mask kernel choices created: %d/%d",
            len(sparse_mask_choices),
            len(sparse_mask_tiling_configs),
        )

        inputs_for_autotuning = forward_input_nodes + list(score_mod_other_buffers)
        input_gen_fns = {
            4: create_compact_q_offsets_fake,
            5: _create_sparse_mask_num_blocks_fake_generator(
                kernel_options.get(
                    "SPARSE_MASK_MAX_NORMAL_BLOCKS",
                    V.graph.sizevars.evaluate_static_shape(kv_indices.get_size()[3]),
                )
            ),
            6: _create_sparse_mask_indices_fake_generator(),
            8: create_num_blocks_fake_generator(full_kv_indices),
            9: create_indices_fake,
        }
        # Check if we have at least one successful choice
        if not choices:
            raise RuntimeError(
                f"All {len(dict_configs)} configs failed to compile. "
                f"Cannot proceed with flex_attention. "
                f"Please check the compilation errors above."
            )

        log.info(
            "fwd-summary: choices=%d total_configs=%d failed_configs=%d call_size_hints=%s aicore_count=%d",
            len(choices), len(dict_configs), len(dict_configs) - len(choices),
            tuple(fwd_call_size_hints), fwd_num_cube_core
        )

        # Print clear overall statistics for flex_attention config compilation
        log.info(
            "flex-attention-summary: total_configs=%d successful=%d failed=%d",
            len(dict_configs), len(choices), len(dict_configs) - len(choices)
        )

        if len(choices) < len(dict_configs):
            log.warning(
                "%d out of %d configs failed to compile. Proceeding with %d successful configs.",
                len(dict_configs) - len(choices), len(dict_configs), len(choices)
            )

        sparse_mask_autotune_other_buffers = _filter_used_subgraph_buffers(
            mask_graph_buffer,
            mask_mod_other_buffers,
        )
        compact_explicit_buffers = [
            compact_q_offsets,
            compact_flat_to_row,
            compact_flat_to_blk,
        ]
        sparse_mask_autotune_other_buffers = [
            buffer
            for buffer in sparse_mask_autotune_other_buffers
            if all(buffer is not explicit for explicit in compact_explicit_buffers)
        ]
        sparse_mask_inputs_for_autotuning = (
            [
                sparse_mask_buffer,
                compact_q_offsets,
                compact_flat_to_row,
                compact_flat_to_blk,
                kv_num_blocks,
                kv_indices,
            ]
            + sparse_mask_autotune_other_buffers
        )
        sparse_mask_input_gen_fns = {
            1: create_compact_q_offsets_fake,
            2: create_zero_int_tensor_fake,
            3: create_zero_int_tensor_fake,
            4: _create_sparse_mask_num_blocks_fake_generator(
                sparse_mask_base_kernel_options["MAX_NORMAL_BLOCKS"]
            ),
            5: _create_sparse_mask_indices_fake_generator(),
        }
        log.info("Sparse mask kernel autotune starting with %d choices", len(sparse_mask_choices))
        autotune_select_algorithm(
            "sparse_mask_kernel",
            sparse_mask_choices,
            sparse_mask_inputs_for_autotuning,
            sparse_mask_layout,
            input_gen_fns=sparse_mask_input_gen_fns,
        )
        log.info(
            "Sparse mask kernel autotune completed with %d choices",
            len(sparse_mask_choices),
        )

        result = autotune_select_algorithm(
            "flex_attention",
            choices,
            inputs_for_autotuning,
            layout,
            input_gen_fns=input_gen_fns,
        )

        return (result, logsumexp)


    @register_lowering(torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None)
    def flex_attention_backward(*args, **kwargs):
        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ) = args
        (
            _,  # q_length
            _,  # kv_length
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            mask_graph,
        ) = block_mask

        (
            query,
            key,
            value,
            grad_out,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = maybe_realize(
            [
                query,
                key,
                value,
                grad_out,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

        device = query.get_device()
        dtype = query.get_dtype()
        Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
        Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

        assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), (
            f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )

        kernel_options = dict(kernel_options)
        use_bwd_mask_out = npu_config.flex_attention.bwd_mask_out
        if not use_bwd_mask_out:
            raise NotImplementedError(
                "NPU flex attention backward only supports fused compact sparse mask-out."
            )
        # Mark symbols in custom kernel options as static shapes and add guards.
        kernel_options = {
            k: V.graph.sizevars.evaluate_static_shape(v)
            if isinstance(v, sympy.Symbol)
            else v
            for k, v in kernel_options.items()
        }
        # kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
        if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
            kernel_options.setdefault("IS_DIVISIBLE", False)
        else:
            kernel_options.setdefault("IS_DIVISIBLE", True)

        fwd_placeholder_inps = [
            create_placeholder(name, dtype, device)
            for name, dtype in [
                ("score", dtype),
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        fw_subgraph_buffer = _build_subgraph_buffer_with_additional_lowerings(
            fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph
        )
        score_mod_is_identity = _is_score_mod_identity_graph(fw_graph)
        log.debug(
            "flex_attention_backward fw_graph identity_check=%s graph:\n%s",
            score_mod_is_identity,
            fw_graph.graph_module.graph,
        )

        joint_placeholder_inps = fwd_placeholder_inps + [
            create_placeholder("grad_score_mod", dtype, device)
        ]
        # Sometimes we have weird unused nodes here
        joint_graph.graph_module.graph.eliminate_dead_code()

        # It is hard to raise nice errors for some joint graphs during subgraph lowering
        # This lets us do some checks before attempting to lower
        validate_joint_graph(joint_graph.graph_module.graph)
        grad_score_mod_is_identity = _is_grad_score_mod_identity_graph(joint_graph)
        log.debug(
            "flex_attention_backward joint_graph identity_check=%s graph:\n%s",
            grad_score_mod_is_identity,
            joint_graph.graph_module.graph,
        )

        all_joint_outputs = _build_subgraph_buffer_with_additional_lowerings(
            joint_placeholder_inps + list(score_mod_other_buffers),
            joint_graph,
        )

        joint_outputs = process_joint_outputs(
            all_joint_outputs, len(joint_placeholder_inps)
        )

        mask_graph_placeholder_inps = [
            create_placeholder(name, dtype, query.get_device())
            for name, dtype in [
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        mask_graph_buffer = _build_subgraph_buffer_with_additional_lowerings(
            mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
        )

        # Construct layout with stride order matching K
        key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
        key_strides = key.get_stride()

        layout_broadcasted_k = FixedLayout(
            key.get_device(),
            key.get_dtype(),
            key_size,
            stride=[sympy.sympify(s) for s in key_strides],
        )
        layout_broadcasted_k_accum = FixedLayout(
            key.get_device(),
            torch.float32,
            key_size,
            stride=[sympy.sympify(s) for s in key_strides],
        )

        # Create delta which will is needed for the bwd's kernel
        grad_lse = grad_logsumexp
        mul_delta = lowerings[aten.mul](out, grad_out)
        delta = lowerings[aten.sum](mul_delta, axis=-1)
        delta = lowerings[aten.sub](delta, grad_lse)
        delta = ExternKernel.require_contiguous(delta)

        grad_lse, delta = maybe_realize([grad_lse, delta])

        # # see NOTE:[TritonTemplates with multiple outputs]
        query_size = [Bq, Hq, seq_len_q, qk_head_dim]
        grad_query_strides = query.get_stride()
        grad_query = empty_strided(
            query_size,
            stride=[sympy.sympify(s) for s in grad_query_strides],
            dtype=query.get_dtype(),
            device=query.get_device(),
        )

        # Construct output layout with stride order matching value
        value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
        value_strides = value.get_stride()

        broadcasted_grad_value = empty_strided(
            value_size,
            stride=[sympy.sympify(s) for s in value_strides],
            dtype=torch.float32,
            device=value.get_device(),
        )
        broadcasted_grad_key_accum = empty_strided(
            key_size,
            stride=[sympy.sympify(s) for s in key_strides],
            dtype=torch.float32,
            device=key.get_device(),
        )
        broadcasted_grad_value = _force_fixed_layout(
            lowerings[aten.fill_](broadcasted_grad_value, 0),
            value_strides,
        )
        broadcasted_grad_key_accum = _force_fixed_layout(
            lowerings[aten.fill_](broadcasted_grad_key_accum, 0),
            key_strides,
        )

        kernel_options.setdefault("SM_SCALE", scale)
        # Determine GQA factor
        gqa_shared_heads = Hq // Hkv
        kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

        # HAS_FULL_BLOCKS is supplied by the eager create_block_mask patch and cached on
        # the BlockMask, so lowering only consumes it.
        has_full_blocks = bool(kernel_options.get("HAS_FULL_BLOCKS", False))
        kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
        if not has_full_blocks:
            raise RuntimeError(
                "NPU flex attention backward requires split sparse-mask metadata with full blocks"
            )

        set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

        SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_Q_BLOCK_SIZE)
        SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)

        compact_tail_end = len(mask_mod_other_buffers)
        compact_tail_start = compact_tail_end - 3
        assert compact_tail_start >= 0, (
            "expected captured compact sparse mask metadata in mask_mod_other_buffers"
        )
        (
            compact_q_offsets,
            compact_flat_to_row,
            compact_flat_to_blk,
        ) = mask_mod_other_buffers[compact_tail_start:compact_tail_end]

        sparse_z = kv_num_blocks.get_size()[0]
        metadata_sparse_hq = kv_num_blocks.get_size()[1]
        num_sparse_q_blocks = kv_num_blocks.get_size()[2]
        metadata_max_normal_blocks = kv_indices.get_size()[3]
        metadata_sparse_hq_val = V.graph.sizevars.evaluate_static_shape(metadata_sparse_hq)
        num_sparse_q_blocks_val = V.graph.sizevars.evaluate_static_shape(num_sparse_q_blocks)
        metadata_max_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(metadata_max_normal_blocks)
        sparse_z_val = V.graph.sizevars.evaluate_static_shape(sparse_z)
        sparse_mask_hq_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options.get("SPARSE_MASK_HQ", metadata_sparse_hq_val)
        )
        sparse_mask_max_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options.get("SPARSE_MASK_MAX_NORMAL_BLOCKS", metadata_max_normal_blocks_val)
        )
        kernel_options.setdefault("SPARSE_MASK_HQ", sparse_mask_hq_val)
        kernel_options.setdefault("SPARSE_MASK_MAX_NORMAL_BLOCKS", sparse_mask_max_normal_blocks_val)
        kv_len_val = V.graph.sizevars.evaluate_static_shape(seq_len_kv)
        total_normal_blocks_val = V.graph.sizevars.evaluate_static_shape(
            kernel_options[_COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION]
        )
        bwd_sparse_mask_size = [
            total_normal_blocks_val,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        ]
        bwd_sparse_mask_strides = [
            SPARSE_Q_BLOCK_SIZE * SPARSE_KV_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            1,
        ]
        bwd_sparse_mask_layout = FixedLayout(
            query.get_device(),
            torch.bool,
            bwd_sparse_mask_size,
            stride=[sympy.sympify(s) for s in bwd_sparse_mask_strides],
        )
        kernel_options.setdefault("SPARSE_MASK_STRIDE_BLK", bwd_sparse_mask_strides[0])
        kernel_options.setdefault("SPARSE_MASK_STRIDE_M", bwd_sparse_mask_strides[1])

        num_sparse_kv_blocks_val = (
            kv_len_val + SPARSE_KV_BLOCK_SIZE - 1
        ) // SPARSE_KV_BLOCK_SIZE
        bwd_sparse_mask_block_pos_size = [
            sparse_z_val,
            sparse_mask_hq_val,
            num_sparse_q_blocks_val,
            num_sparse_kv_blocks_val,
        ]
        bwd_sparse_mask_block_pos_strides = [
            sparse_mask_hq_val * num_sparse_q_blocks_val * num_sparse_kv_blocks_val,
            num_sparse_q_blocks_val * num_sparse_kv_blocks_val,
            num_sparse_kv_blocks_val,
            1,
        ]
        bwd_sparse_mask_block_pos_layout = FixedLayout(
            query.get_device(),
            torch.int32,
            bwd_sparse_mask_block_pos_size,
            stride=[sympy.sympify(s) for s in bwd_sparse_mask_block_pos_strides],
        )

        bwd_query_call_size_hints = V.graph.sizevars.size_hints(
            query.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        bwd_key_call_size_hints = V.graph.sizevars.size_hints(
            key.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        (
            bwd_batch_size_hint,
            bwd_q_heads_hint,
            bwd_num_queries_hint,
            _,
        ) = bwd_query_call_size_hints
        _, bwd_kv_heads_hint, bwd_num_key_value_hint, _ = bwd_key_call_size_hints
        bwd_num_cube_core = _get_num_cube_core()

        log.debug(
            "flex_attention_backward lowering: query=%s key=%s SPARSE_Q=%s SPARSE_KV=%s use_config_generator=%s",
            query.get_size(), key.get_size(), SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE,
            npu_config.flex_attention.use_config_generator
        )

        dq_choices: list[Any] = []
        dkdv_choices: list[Any] = []
        bwd_sparse_mask_choices = []
        bwd_sparse_mask_block_pos_choices = []
        bwd_sparse_mask_base_kernel_options = {
            "SPARSE_Z": V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[0]),
            "SPARSE_HQ": kernel_options.get(
                "SPARSE_MASK_HQ",
                V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[1]),
            ),
            "NUM_SPARSE_Q_BLOCKS": V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[2]),
            "MAX_NORMAL_BLOCKS": kernel_options.get(
                "SPARSE_MASK_MAX_NORMAL_BLOCKS",
                V.graph.sizevars.evaluate_static_shape(kv_indices.get_size()[3]),
            ),
            "SPARSE_Q_BLOCK_SIZE": SPARSE_Q_BLOCK_SIZE,
            "SPARSE_KV_BLOCK_SIZE": SPARSE_KV_BLOCK_SIZE,
            "Q_LEN": V.graph.sizevars.evaluate_static_shape(seq_len_q),
            "KV_LEN": V.graph.sizevars.evaluate_static_shape(seq_len_kv),
            "TOTAL_FLAT_ENTRIES": kernel_options[
                _COMPACT_SPARSE_MASK_TOTAL_BLOCKS_OPTION
            ],
            "SPARSE_MASK_STRIDE_BLK": bwd_sparse_mask_strides[0],
            "SPARSE_MASK_STRIDE_M": bwd_sparse_mask_strides[1],
        }
        bwd_sparse_mask_tiling_configs = build_sparse_mask_candidate_configs(
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )
        for bwd_sparse_mask_tiling_config in bwd_sparse_mask_tiling_configs:
            bwd_sparse_mask_kernel_options = bwd_sparse_mask_base_kernel_options.copy()
            bwd_sparse_mask_kernel_options.update(bwd_sparse_mask_tiling_config)
            bwd_sparse_mask_choice_count = len(bwd_sparse_mask_choices)
            bwd_sparse_mask_kernel_compact_template.maybe_append_choice(
                choices=bwd_sparse_mask_choices,
                input_nodes=[
                    compact_q_offsets,
                    compact_flat_to_row,
                    compact_flat_to_blk,
                    kv_num_blocks,
                    kv_indices,
                ],
                layout=bwd_sparse_mask_layout,
                subgraphs=[mask_graph_buffer],
                call_sizes=[
                    bwd_sparse_mask_base_kernel_options["TOTAL_FLAT_ENTRIES"],
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                ],
                **bwd_sparse_mask_kernel_options,
            )
            if len(bwd_sparse_mask_choices) > bwd_sparse_mask_choice_count:
                _tag_choice_configs(
                    bwd_sparse_mask_choices[bwd_sparse_mask_choice_count:],
                    "_bwd_sparse_mask_report_config",
                    bwd_sparse_mask_tiling_config,
                )
                if prefer_max_tiling_without_benchmark():
                    _tag_choice_attr(
                        bwd_sparse_mask_choices[bwd_sparse_mask_choice_count:],
                        "_nobench_select_first_compilable",
                        True,
                    )
                log.info(
                    "Backward sparse mask kernel choice created successfully: %s",
                    bwd_sparse_mask_tiling_config,
                )
        if not bwd_sparse_mask_choices:
            raise RuntimeError(
                "Backward mask-out mode could not create compact sparse mask kernel choices."
            )

        bwd_sparse_mask_block_pos_kernel_options = {
            "SPARSE_Z": bwd_sparse_mask_base_kernel_options["SPARSE_Z"],
            "SPARSE_HQ": bwd_sparse_mask_base_kernel_options["SPARSE_HQ"],
            "NUM_SPARSE_Q_BLOCKS": bwd_sparse_mask_base_kernel_options[
                "NUM_SPARSE_Q_BLOCKS"
            ],
            "MAX_NORMAL_BLOCKS": bwd_sparse_mask_base_kernel_options[
                "MAX_NORMAL_BLOCKS"
            ],
            "NUM_SPARSE_KV_BLOCKS": num_sparse_kv_blocks_val,
            "SPARSE_MASK_BLOCK_POS_STRIDE_Z": bwd_sparse_mask_block_pos_strides[0],
            "SPARSE_MASK_BLOCK_POS_STRIDE_H": bwd_sparse_mask_block_pos_strides[1],
            "SPARSE_MASK_BLOCK_POS_STRIDE_Q": bwd_sparse_mask_block_pos_strides[2],
            "NUM_Q_SUB_BLOCKS": 1,
            "NUM_KV_SUB_BLOCKS": 1,
            "num_stages": 1,
            "num_warps": 4,
        }
        sparse_mask_block_pos_template.maybe_append_choice(
            choices=bwd_sparse_mask_block_pos_choices,
            input_nodes=[
                kv_num_blocks,
                kv_indices,
            ],
            layout=bwd_sparse_mask_block_pos_layout,
            call_sizes=[
                bwd_sparse_mask_block_pos_kernel_options["SPARSE_Z"],
                bwd_sparse_mask_block_pos_kernel_options["SPARSE_HQ"],
                bwd_sparse_mask_block_pos_kernel_options["NUM_SPARSE_Q_BLOCKS"],
                bwd_sparse_mask_block_pos_kernel_options["MAX_NORMAL_BLOCKS"],
            ],
            **bwd_sparse_mask_block_pos_kernel_options,
        )
        if not bwd_sparse_mask_block_pos_choices:
            raise RuntimeError(
                "Backward mask-out mode could not create sparse mask block-position choices."
            )

        compact_explicit_buffers = [
            compact_q_offsets,
            compact_flat_to_row,
            compact_flat_to_blk,
        ]
        bwd_sparse_mask_autotune_other_buffers = _filter_used_subgraph_buffers(
            mask_graph_buffer,
            mask_mod_other_buffers,
        )
        bwd_sparse_mask_autotune_other_buffers = [
            buffer
            for buffer in bwd_sparse_mask_autotune_other_buffers
            if all(buffer is not explicit for explicit in compact_explicit_buffers)
        ]
        bwd_sparse_mask_inputs_for_autotuning = (
            [
                compact_q_offsets,
                compact_flat_to_row,
                compact_flat_to_blk,
                kv_num_blocks,
                kv_indices,
            ]
            + bwd_sparse_mask_autotune_other_buffers
        )
        bwd_sparse_mask_input_gen_fns = {
            0: create_compact_q_offsets_fake,
            1: create_zero_int_tensor_fake,
            2: create_zero_int_tensor_fake,
            3: _create_sparse_mask_num_blocks_fake_generator(
                bwd_sparse_mask_base_kernel_options["MAX_NORMAL_BLOCKS"]
            ),
            4: _create_sparse_mask_indices_fake_generator(),
        }
        bwd_sparse_mask_result = autotune_select_algorithm(
            "bwd_sparse_mask_kernel_compact",
            bwd_sparse_mask_choices,
            bwd_sparse_mask_inputs_for_autotuning,
            bwd_sparse_mask_layout,
            input_gen_fns=bwd_sparse_mask_input_gen_fns,
        )
        log.info(
            "Backward compact sparse mask kernel autotune completed with %d choices: %s",
            len(bwd_sparse_mask_choices),
            bwd_sparse_mask_result,
        )

        bwd_sparse_mask_block_pos_result = autotune_select_algorithm(
            "sparse_mask_block_pos",
            bwd_sparse_mask_block_pos_choices,
            [
                kv_num_blocks,
                kv_indices,
            ],
            bwd_sparse_mask_block_pos_layout,
            input_gen_fns={
                0: _create_sparse_mask_num_blocks_fake_generator(
                    bwd_sparse_mask_base_kernel_options["MAX_NORMAL_BLOCKS"]
                ),
                1: _create_sparse_mask_indices_fake_generator(),
            },
        )
        log.info(
            "Sparse mask block-position kernel autotune completed with %d choices: %s",
            len(bwd_sparse_mask_block_pos_choices),
            bwd_sparse_mask_block_pos_result,
        )

        bwd_dict_configs = generate_bwd_split_mask_out_candidate_configs(
            query_shape=query.get_size(),
            key_shape=key.get_size(),
            sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
            sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            dtype=query.get_dtype(),
            num_cube_core=bwd_num_cube_core,
        )

        log.debug(
            "bwd split dict_configs count: split=%d",
            len(bwd_dict_configs),
        )

        original_kernel_options = kernel_options.copy()
        assert bwd_sparse_mask_result is not None
        assert compact_q_offsets is not None
        assert bwd_sparse_mask_block_pos_result is not None
        bwd_mask_out_input_nodes = [
            bwd_sparse_mask_result,
            compact_q_offsets,
            bwd_sparse_mask_block_pos_result,
        ]

        def make_bwd_base_kernel_options(cfg: dict) -> dict:
            cur_kernel_options = original_kernel_options.copy()
            for k in list(cur_kernel_options.keys()):
                if k.startswith("bwd_"):
                    v = cur_kernel_options.pop(k)
                    cur_kernel_options[k[4:]] = v
                if k.startswith("fwd_"):
                    cur_kernel_options.pop(k)

            # Apply all config parameters (BLOCK_M1, BLOCK_N1, etc., NPU params)
            for k, v in cfg.items():
                cur_kernel_options.setdefault(k, v)

            for key in npu_config.FLEX_ATTENTION_NPU_COMPILE_HINT_KEYS:
                cur_kernel_options.pop(key, None)

            # Blocksparse options
            cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
            cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
            cur_kernel_options.setdefault("BWD_SCORE_MOD_IS_IDENTITY", score_mod_is_identity)
            cur_kernel_options.setdefault("BWD_GRAD_SCORE_MOD_IS_IDENTITY", grad_score_mod_is_identity)
            cur_kernel_options.setdefault("BWD_IDENTITY_SCORE_AND_GRAD", bwd_identity_score_and_grad)
            cur_kernel_options.setdefault(
                "NUM_SPARSE_Q_BLOCKS",
                V.graph.sizevars.evaluate_static_shape(kv_num_blocks.get_size()[2]),
            )
            cur_kernel_options = apply_kernel_options_from_block_sparse_mask(
                cur_kernel_options,
                kv_num_blocks,
                kv_indices,
                context="bwd",
            )
            return cur_kernel_options

        def make_bwd_dq_kernel_options(cfg: dict) -> dict:
            opts = make_bwd_base_kernel_options(cfg)
            opts.update(
                {
                    "BLOCK_M2": cfg["BLOCK_M2"],
                    "BLOCK_N2": cfg["BLOCK_N2"],
                    "num_stages": 2,
                    "num_warps": 4,
                }
            )
            opts.update(get_bwd_dq_compile_options())
            opts.update(
                _build_qmajor_dq_launch_meta(
                    batch_size_hint=bwd_batch_size_hint,
                    q_heads_hint=bwd_q_heads_hint,
                    num_queries_hint=bwd_num_queries_hint,
                    block_m=cfg["BLOCK_M2"],
                )
            )
            return opts

        def make_bwd_dkdv_kernel_options(cfg: dict) -> dict:
            opts = make_bwd_base_kernel_options(cfg)
            opts.update(
                {
                    "BLOCK_M1": cfg["BLOCK_M1"],
                    "BLOCK_N1": cfg["BLOCK_N1"],
                    "num_stages": 2,
                    "num_warps": 4,
                }
            )
            opts.update(get_bwd_dkdv_compile_options())
            opts.update(
                _build_persistent_bwd_launch_meta(
                    batch_size_hint=bwd_batch_size_hint,
                    kv_heads_hint=bwd_kv_heads_hint,
                    num_key_value_hint=bwd_num_key_value_hint,
                    block_n1=cfg["BLOCK_N1"],
                )
            )
            return opts

        def log_bwd_choice(kind: str, cfg: dict, cur_kernel_options: dict) -> None:
            bwd_grid_x = (
                (bwd_num_queries_hint + cfg["BLOCK_M2"] - 1) // cfg["BLOCK_M2"]
            ) * (bwd_q_heads_hint // bwd_kv_heads_hint) + (
                (bwd_num_key_value_hint + cfg["BLOCK_N1"] - 1) // cfg["BLOCK_N1"]
            )
            bwd_grid_y = 1
            bwd_grid_z = bwd_batch_size_hint * bwd_kv_heads_hint
            bwd_total_programs = bwd_grid_x * bwd_grid_y * bwd_grid_z
            bwd_wave_estimate = (
                bwd_total_programs + bwd_num_cube_core - 1
            ) // bwd_num_cube_core

            log.debug(
                "bwd-%s-choice: cfg=%s grid=(%d,%d,%d) total_programs=%d aicore_count=%d exceeds_aicores=%s wave_estimate=%d kv_num_blocks_type=%s kv_indices_type=%s grad_lse_type=%s final_kernel_options=%s",
                kind, cfg, bwd_grid_x, bwd_grid_y, bwd_grid_z, bwd_total_programs, bwd_num_cube_core,
                bwd_total_programs > bwd_num_cube_core, bwd_wave_estimate,
                type(kv_num_blocks).__name__, type(kv_indices).__name__, type(grad_lse).__name__, cur_kernel_options
            )

        has_captured_grad_side_effect = bool(joint_outputs.mutated_grads)
        if has_captured_grad_side_effect:
            score_mod_is_identity = False
            grad_score_mod_is_identity = False
        bwd_identity_score_and_grad = (
            score_mod_is_identity and grad_score_mod_is_identity
        )
        captured_grad_owner = "dkdv" if has_captured_grad_side_effect else None
        assert captured_grad_owner in (None, "dq", "dkdv")
        log.debug(
            "bwd split captured_grad_owner=%s mutated_grads=%d score_identity=%s grad_identity=%s identity_score_and_grad=%s",
            captured_grad_owner,
            len(joint_outputs.mutated_grads),
            score_mod_is_identity,
            grad_score_mod_is_identity,
            bwd_identity_score_and_grad,
        )

        def make_bwd_subgraphs_and_mutations(kind: str, base_mutated_inputs: list[Any]):
            subgraphs = [
                fw_subgraph_buffer,
                joint_outputs.grad_input,
                mask_graph_buffer,
            ]
            mutated_inputs = list(base_mutated_inputs)
            run_captured_grads = captured_grad_owner == kind
            if run_captured_grads:
                subgraphs.append(joint_outputs.captured_grads_compute)
                mutated_inputs.extend(joint_outputs.mutated_grads)
            return subgraphs, mutated_inputs, run_captured_grads

        dq_input_nodes = [
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_out,
            grad_query,
            *bwd_mask_out_input_nodes,
            kv_num_blocks,
            kv_indices,
            q_num_blocks,
            q_indices,
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
        dkdv_input_nodes = [
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_out,
            broadcasted_grad_value,
            broadcasted_grad_key_accum,
            *bwd_mask_out_input_nodes,
            kv_num_blocks,
            kv_indices,
            q_num_blocks,
            q_indices,
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ]

        for cfg in bwd_dict_configs:
            if not is_bwd_config_compatible(
                cfg, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE
            ):
                continue

            if (
                cfg["BLOCK_M2"] == SPARSE_Q_BLOCK_SIZE
                and cfg["BLOCK_N2"] == SPARSE_KV_BLOCK_SIZE
            ):
                dq_kernel_options = make_bwd_dq_kernel_options(cfg)
                dq_subgraphs, dq_mutated_inputs, dq_run_captured = (
                    make_bwd_subgraphs_and_mutations("dq", [grad_query])
                )
                dq_kernel_options["RUN_CAPTURED_GRADS"] = dq_run_captured
                log_bwd_choice("dq", cfg, dq_kernel_options)

                prev_dq_choice_count = len(dq_choices)
                flex_attention_backward_qmajor_dq_template.maybe_append_choice(
                    choices=dq_choices,
                    input_nodes=dq_input_nodes,
                    layout=grad_query.get_layout(),
                    subgraphs=dq_subgraphs,
                    mutated_inputs=dq_mutated_inputs,
                    reset_to_zero_arg_names=None,
                    large_input_buffers=bwd_mask_out_input_nodes,
                    call_sizes=query.get_size() + key.get_size()[1:3],
                    **dq_kernel_options,
                )
                if len(dq_choices) > prev_dq_choice_count:
                    _tag_flex_attention_report_choices(
                        dq_choices[prev_dq_choice_count:],
                        "backward_dq",
                        cfg,
                    )
            else:
                log.debug(
                    "skip bwd-dq choice requiring full sparse block: cfg=%s SPARSE_Q=%s SPARSE_KV=%s",
                    cfg,
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )

            dkdv_kernel_options = make_bwd_dkdv_kernel_options(cfg)
            dkdv_subgraphs, dkdv_mutated_inputs, dkdv_run_captured = (
                make_bwd_subgraphs_and_mutations(
                    "dkdv",
                    [broadcasted_grad_value, broadcasted_grad_key_accum],
                )
            )
            dkdv_kernel_options["RUN_CAPTURED_GRADS"] = dkdv_run_captured
            log_bwd_choice("dkdv", cfg, dkdv_kernel_options)

            prev_dkdv_choice_count = len(dkdv_choices)
            flex_attention_backward_dkdv_only_template.maybe_append_choice(
                choices=dkdv_choices,
                input_nodes=dkdv_input_nodes,
                layout=layout_broadcasted_k_accum,
                subgraphs=dkdv_subgraphs,
                mutated_inputs=dkdv_mutated_inputs,
                reset_to_zero_arg_names=["arg_DV", "arg_DK"],
                large_input_buffers=bwd_mask_out_input_nodes,
                call_sizes=query.get_size() + key.get_size()[1:3],
                **dkdv_kernel_options,
            )
            if len(dkdv_choices) > prev_dkdv_choice_count:
                _tag_flex_attention_report_choices(
                    dkdv_choices[prev_dkdv_choice_count:],
                    "backward_dkdv",
                    cfg,
                )

        dq_inputs_for_autotuning = (
            dq_input_nodes
            + list(score_mod_other_buffers)
            + (
                list(joint_outputs.mutated_grads)
                if captured_grad_owner == "dq"
                else []
            )
        )
        dq_q_offsets_input_idx = 8
        dq_block_pos_input_idx = 9
        dq_block_metadata_input_idx = 10
        dq_input_gen_fns = {
            dq_q_offsets_input_idx: create_compact_q_offsets_fake,
            dq_block_pos_input_idx: create_zero_int_tensor_fake,
            dq_block_metadata_input_idx: create_num_blocks_fake_generator(kv_indices),
            dq_block_metadata_input_idx + 1: create_indices_fake,
            dq_block_metadata_input_idx + 2: create_num_blocks_fake_generator(q_indices),
            dq_block_metadata_input_idx + 3: create_indices_fake,
            dq_block_metadata_input_idx + 4: create_num_blocks_fake_generator(full_kv_indices),
            dq_block_metadata_input_idx + 5: create_indices_fake,
            dq_block_metadata_input_idx + 6: create_num_blocks_fake_generator(full_q_indices),
            dq_block_metadata_input_idx + 7: create_indices_fake,
        }

        dkdv_inputs_for_autotuning = (
            dkdv_input_nodes
            + list(score_mod_other_buffers)
            + (
                list(joint_outputs.mutated_grads)
                if captured_grad_owner == "dkdv"
                else []
            )
        )
        dkdv_q_offsets_input_idx = 9
        dkdv_block_pos_input_idx = 10
        dkdv_block_metadata_input_idx = 11
        dkdv_input_gen_fns = {
            dkdv_q_offsets_input_idx: create_compact_q_offsets_fake,
            dkdv_block_pos_input_idx: create_zero_int_tensor_fake,
            dkdv_block_metadata_input_idx: create_num_blocks_fake_generator(kv_indices),
            dkdv_block_metadata_input_idx + 1: create_indices_fake,
            dkdv_block_metadata_input_idx + 2: create_num_blocks_fake_generator(q_indices),
            dkdv_block_metadata_input_idx + 3: create_indices_fake,
            dkdv_block_metadata_input_idx + 4: create_num_blocks_fake_generator(full_kv_indices),
            dkdv_block_metadata_input_idx + 5: create_indices_fake,
            dkdv_block_metadata_input_idx + 6: create_num_blocks_fake_generator(full_q_indices),
            dkdv_block_metadata_input_idx + 7: create_indices_fake,
        }

        log.info(
            "bwd-summary: dq_choices=%d dkdv_choices=%d query_call_size_hints=%s key_call_size_hints=%s aicore_count=%d",
            len(dq_choices),
            len(dkdv_choices),
            tuple(bwd_query_call_size_hints), tuple(bwd_key_call_size_hints), bwd_num_cube_core
        )

        autotune_select_algorithm(
            "flex_attention_backward_qmajor_dq",
            dq_choices,
            dq_inputs_for_autotuning,
            grad_query.get_layout(),
            input_gen_fns=dq_input_gen_fns,
        )

        autotune_select_algorithm(
            "flex_attention_backward_dkdv_only",
            dkdv_choices,
            dkdv_inputs_for_autotuning,
            layout_broadcasted_k_accum,
            input_gen_fns=dkdv_input_gen_fns,
        )

        if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
            grad_key_accum = broadcasted_grad_key_accum
            grad_value_accum = broadcasted_grad_value
        else:
            assert V.graph.sizevars.evaluate_expr(sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)), (
                f"Bq and Bkv must broadcastable. "
                f"Got Bq={V.graph.sizevars.evaluate_expr(Bq)} "
                f"and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}"
            )
            grad_key_accum = lowerings[aten.sum](
                broadcasted_grad_key_accum, axis=0, keepdims=True
            )
            grad_value_accum = lowerings[aten.sum](
                broadcasted_grad_value, axis=0, keepdims=True
            )
        if not kernel_options.get("PRESCALE_QK", False):
            sm_scale = kernel_options["SM_SCALE"]
            grad_key_accum = lowerings[aten.mul](grad_key_accum, sm_scale)
        grad_value = _force_fixed_layout(
            _maybe_copy_to_dtype(grad_value_accum, value.get_dtype()),
            value_strides,
        )
        grad_key = _force_fixed_layout(
            _maybe_copy_to_dtype(grad_key_accum, key.get_dtype()),
            key_strides,
        )
        grad_query = _force_fixed_layout(
            _maybe_copy_to_dtype(grad_query, query.get_dtype()),
            grad_query_strides,
        )

        return (grad_query, grad_key, grad_value, tuple(joint_outputs.captured_grads))
