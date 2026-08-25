""" Triton Implementation of the flex_attention Kernel"""

import math
from collections.abc import Sequence
from functools import wraps
from typing import Any, Dict, Optional, Union

import sympy

import torch
from torch._inductor.virtualized import V, ops
from torch.utils._pytree import tree_map

from torch._inductor import config
from torch_npu._inductor import config as npu_config
from torch_npu._inductor.config import log
from torch_npu._inductor.flex_attention_tasklist import (
    FlexAttentionDkdvDispatchSpec,
    RuntimeTemplateArg,
    is_dkdv_tasklist_codegen_compatible,
)
from torch_npu._inductor.kernel.flexattention_template import (
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

from torch._inductor.ir import (
    AssertScalar,
    ComputedBuffer,
    DynamicScalar,
    ExternKernel,
    FixedLayout,
    get_fill_order,
    StorageBox,
    TensorBox,
)
from torch._inductor.lowering import (
    empty,
    empty_strided,
    lowerings,
    register_lowering,
    to_dtype,
)
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch.nn.attention import flex_attention as flex_attention_module
try:
    from torch._inductor.kernel.flex.common import (
        construct_strides,
        create_indices_fake,
        create_num_blocks_fake_generator,
        create_placeholder,
        infer_dense_strides,
        maybe_realize,
        set_head_dim_values,
    )
    from torch._inductor.kernel.flex.flex_attention import (
        flex_attention_backward_template as upstream_flex_attention_backward_template,
        flex_attention_template as upstream_flex_attention_template,
        get_bwd_subgraph_outputs,
        get_float32_precision,
        get_fwd_subgraph_outputs,
        process_joint_outputs,
        validate_joint_graph,
    )
except ImportError:
    from torch._inductor.kernel.flex_attention import (
        construct_strides,
        create_indices_fake,
        create_num_blocks_fake_generator,
        create_placeholder,
        flex_attention_backward_template as upstream_flex_attention_backward_template,
        flex_attention_template as upstream_flex_attention_template,
        get_bwd_subgraph_outputs,
        get_float32_precision,
        get_fwd_subgraph_outputs,
        infer_dense_strides,
        maybe_realize,
        process_joint_outputs,
        set_head_dim_values,
        validate_joint_graph,
    )

# PyTorch flex_attention exposes/saves LSE in log2 space, matching the
# upstream kernels that use exp2/log2. NPU templates below compute LSE with
# natural exp/log, so the lowering boundary converts between the two bases.
_LN2 = 0.6931471805599453
_LOG2E = 1.4426950408889634


def _tag_flex_attention_report_choices(new_choices, cfg):
    """Attach tiling metadata used by NPU choice diagnostics."""
    report_config = {
        key: cfg[key]
        for key in (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
            "num_warps",
            "num_stages",
        )
        if key in cfg
    }
    for choice in new_choices:
        setattr(  # noqa: B010
            choice, "_flex_attention_report_config", report_config.copy()
        )


def _tag_choice_configs(new_choices, attr_name: str, cfg: dict[str, Any]) -> None:
    """Attach tiling metadata used by NPU fallback choice ordering."""
    for choice in new_choices:
        setattr(choice, attr_name, cfg.copy())  # noqa: B010s


def _tag_choice_attr(new_choices, attr_name: str, value: Any) -> None:
    for choice in new_choices:
        setattr(choice, attr_name, value)  # noqa: B010


def _tag_choices_for_no_benchmark(new_choices) -> None:
    if prefer_max_tiling_without_benchmark():
        _tag_choice_attr(
            new_choices,
            "_nobench_select_first_compilable",
            True,
        )


def _is_named_ir_node(value: Any) -> bool:
    return hasattr(value, "get_name") and hasattr(value, "get_size")


_EXPLICIT_SCORE_MOD_OPTION = "_NPU_EXPLICIT_SCORE_MOD"
_STREAMING_BLOCK_MASK_TARGET_BYTES = 256 * 1024 * 1024
_STREAMING_BLOCK_MASK_BYTES_PER_ELEMENT = 8


def create_zero_int_tensor_fake(x) -> torch.Tensor:
    size = V.graph.sizevars.optimization_hints(
        x.get_size(),
        fallback=config.unbacked_symint_fallback,
    )
    return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())


def create_minus_one_int_tensor_fake(x) -> torch.Tensor:
    size = V.graph.sizevars.optimization_hints(
        x.get_size(),
        fallback=config.unbacked_symint_fallback,
    )
    return torch.full(size, -1, dtype=x.get_dtype(), device=x.get_device())


def create_compact_q_offsets_fake(x) -> torch.Tensor:
    size = V.graph.sizevars.optimization_hints(
        x.get_size(),
        fallback=config.unbacked_symint_fallback,
    )
    return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())


def create_sparse_mask_num_blocks_fake(x) -> torch.Tensor:
    size = V.graph.sizevars.optimization_hints(
        x.get_size(),
        fallback=config.unbacked_symint_fallback,
    )
    return torch.ones(size, dtype=x.get_dtype(), device=x.get_device())


def _create_sparse_mask_num_blocks_fake_generator(max_normal_blocks: int):
    num_blocks_for_autotuning = 1 if int(max_normal_blocks) > 0 else 0

    def create_sparse_mask_num_blocks_fake(x) -> torch.Tensor:
        size = V.graph.sizevars.optimization_hints(
            x.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        return torch.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_sparse_mask_num_blocks_fake


def _create_sparse_mask_indices_fake_generator():
    def create_sparse_mask_indices_fake(x) -> torch.Tensor:
        size = V.graph.sizevars.optimization_hints(
            x.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        return torch.zeros(size, dtype=x.get_dtype(), device=x.get_device())

    return create_sparse_mask_indices_fake


from torch_npu._inductor.kernel.flex_attention_metadata import (
    apply_kernel_options_from_eager_block_mask,
    infer_eager_block_mask_kernel_options,
)
from torch_npu._inductor.kernel.flex_attention_config_generator import (
    FlexMode,
    NO_SPARSE_BLOCK_SIZE,
    build_sparse_mask_candidate_configs,
    get_bwd_dkdv_compile_options,
    get_bwd_dq_compile_options,
    generate_bwd_candidate_configs,
    generate_fwd_candidate_configs,
    prefer_max_tiling_without_benchmark,
    sparse_mask_attention_cvpipeline_config_variants,
    validate_benchmark_config,
)
aten = torch.ops.aten
prims = torch.ops.prims
Expr = sympy.Expr


def _build_runtime_compact_sparse_mask_offsets(
    *,
    kv_num_blocks,
    kv_indices,
    device,
    context: str,
):
    q_offsets = empty_strided(
        kv_num_blocks.get_size(),
        None,
        dtype=torch.int32,
        device=device,
    )
    row_count = sympy.prod(kv_num_blocks.get_size())
    upper_capacity = row_count * kv_indices.get_size()[3]
    V.graph.sizevars.check(
        sympy.Le(
            upper_capacity,
            sympy.Integer(torch.iinfo(torch.int32).max),
        )
    )
    total_blocks = empty_strided(
        [1],
        stride=[1],
        dtype=torch.int32,
        device=device,
    )
    total_blocks = _force_fixed_layout(
        lowerings[aten.fill_](total_blocks, 0),
        [1],
    )
    total_layout = FixedLayout(device, torch.int32, [1], stride=[1])

    choices = []
    flex_attention_compact_offsets.maybe_append_choice(
        choices=choices,
        input_nodes=[
            q_offsets,
            total_blocks,
            kv_num_blocks,
        ],
        layout=total_layout,
        mutated_inputs=[
            q_offsets,
            total_blocks,
        ],
        call_sizes=[row_count],
        num_stages=1,
        num_warps=4,
        NUM_VECTOR_CORE=_get_num_vector_core(),
    )
    if not choices:
        raise RuntimeError(
            f"{context} could not create compact sparse mask metadata kernel"
        )

    autotune_select_algorithm(
        f"{context}_compact_sparse_mask_metadata",
        choices,
        [
            q_offsets,
            total_blocks,
            kv_num_blocks,
        ],
        total_layout,
        input_gen_fns={
            0: create_compact_q_offsets_fake,
            1: create_zero_int_tensor_fake,
            2: create_sparse_mask_num_blocks_fake,
        },
    )
    return q_offsets, total_blocks, row_count


def _bind_runtime_total_blocks_as_unbacked_size(
    runtime_total_blocks: TensorBox,
    *,
    max_blocks: int,
) -> sympy.Symbol:
    shape_env = V.graph.sizevars.shape_env
    symint = shape_env.create_unbacked_symint()
    symbol = symint.node.expr
    shape_env._constrain_range_for_size(symbol, min=0, max=max_blocks)

    dynamic_scalar = DynamicScalar(symbol, (), runtime_total_blocks)
    dynamic_scalar.name = V.graph.register_buffer(dynamic_scalar)
    V.graph.register_operation(dynamic_scalar)

    pending = shape_env.pending_fresh_unbacked_symbols
    shape_env.pending_fresh_unbacked_symbols = [
        value for value in pending if value != symbol
    ]

    range_assert = AssertScalar(
        sympy.And(symbol >= 0, symbol <= max_blocks),
        f"compact sparse-mask block count must be in [0, {max_blocks}]",
    )
    range_assert.name = V.graph.register_buffer(range_assert)
    V.graph.register_operation(range_assert)
    return symbol


def _build_runtime_compact_sparse_mask_mapping(
    *,
    flat_to_row,
    flat_to_blk,
    q_offsets,
    kv_num_blocks,
    row_count,
    device,
    context: str,
):
    mapping_layout = FixedLayout(device, torch.int32, [1], stride=[1])
    choices = []
    flex_attention_compact_mapping.maybe_append_choice(
        choices=choices,
        input_nodes=[flat_to_row, flat_to_blk, q_offsets, kv_num_blocks],
        layout=mapping_layout,
        mutated_inputs=[flat_to_row, flat_to_blk],
        call_sizes=[row_count],
        num_stages=1,
        num_warps=4,
        NUM_VECTOR_CORE=_get_num_vector_core(),
    )
    if not choices:
        raise RuntimeError(
            f"{context} could not create compact sparse mask mapping kernel"
        )

    autotune_select_algorithm(
        f"{context}_compact_sparse_mask_mapping",
        choices,
        [flat_to_row, flat_to_blk, q_offsets, kv_num_blocks],
        mapping_layout,
        input_gen_fns={
            0: create_zero_int_tensor_fake,
            1: create_zero_int_tensor_fake,
            2: create_compact_q_offsets_fake,
            3: create_sparse_mask_num_blocks_fake,
        },
    )


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


def _has_sparse_block_mask(
    sparse_q_block_size: Any, sparse_kv_block_size: Any
) -> bool:
    sparse_q_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_q_block_size
    )
    sparse_kv_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_kv_block_size
    )
    return not (
        sparse_q_block_size == NO_SPARSE_BLOCK_SIZE
        and sparse_kv_block_size == NO_SPARSE_BLOCK_SIZE
    )


def _is_npu_device(device: Any) -> bool:
    try:
        return torch.device(device).type == "npu"
    except (RuntimeError, TypeError):
        return False


def _create_block_mask_streaming(
    mask_mod,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
    BLOCK_SIZE: Union[int, tuple[int, int]] = 128,
    _compile: bool = False,
):
    """Create an NPU BlockMask without materializing its full dense mask.

    ``create_mask`` is evaluated in Q-block stripes and each stripe is converted
    to the standard partial/full block representation immediately. The final
    BlockMask is assembled with PyTorch's existing helper, so all downstream
    Inductor consumers keep the original BlockMask layout and semantics.
    """
    mod_type = flex_attention_module._get_mod_type(mask_mod)
    assert (
        mod_type == flex_attention_module._ModificationType.MASK_MOD
    ), f"create-block_mask requires a mask_mod function! Got {mask_mod}"

    if B is None:
        B = 1
    if H is None:
        H = 1
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE = BLOCK_SIZE
        KV_BLOCK_SIZE = BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    max_rows_per_stripe = max(
        1,
        _STREAMING_BLOCK_MASK_TARGET_BYTES
        // (B * H * KV_LEN * _STREAMING_BLOCK_MASK_BYTES_PER_ELEMENT),
    )
    stripe_q_blocks = max(1, max_rows_per_stripe // Q_BLOCK_SIZE)

    partial_block_masks = []
    full_block_masks = []
    for q_start in range(0, Q_LEN, stripe_q_blocks * Q_BLOCK_SIZE):
        stripe_q_len = min(stripe_q_blocks * Q_BLOCK_SIZE, Q_LEN - q_start)

        def shifted_mask_mod(
            b,
            h,
            q_idx,
            kv_idx,
            _mask_mod=mask_mod,
            _q_start=q_start,
        ):
            return _mask_mod(b, h, q_idx + _q_start, kv_idx)

        stripe_mask = flex_attention_module.create_mask(
            shifted_mask_mod,
            B,
            H,
            stripe_q_len,
            KV_LEN,
            device,
        )
        partial_blocks, full_blocks = (
            flex_attention_module._convert_mask_to_block_mask(
                stripe_mask,
                Q_BLOCK_SIZE=Q_BLOCK_SIZE,
                KV_BLOCK_SIZE=KV_BLOCK_SIZE,
                separate_full_blocks=True,
            )
        )
        partial_block_masks.append(partial_blocks)
        full_block_masks.append(full_blocks)
        del stripe_mask

    partial_block_mask = torch.cat(partial_block_masks, dim=2)
    full_block_mask = torch.cat(full_block_masks, dim=2)
    return flex_attention_module._create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )


def _should_use_streaming_block_mask(args, kwargs) -> bool:
    if (
        torch.compiler.is_dynamo_compiling()
        or kwargs.get("_compile", False)
        or (len(args) > 7 and args[7])
    ):
        return False
    device = kwargs.get("device", args[5] if len(args) > 5 else "cuda")
    q_len = kwargs.get("Q_LEN", args[3] if len(args) > 3 else None)
    kv_len = kwargs.get("KV_LEN", args[4] if len(args) > 4 else None)
    return (
        _is_npu_device(device)
        and isinstance(q_len, int)
        and isinstance(kv_len, int)
        and q_len > 0
        and kv_len > 0
    )


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
        updated_kernel_options = dict(updated_kernel_options)
        updated_kernel_options[_EXPLICIT_SCORE_MOD_OPTION] = score_mod is not None
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
            if _should_use_streaming_block_mask(args, kwargs):
                try:
                    block_mask = _create_block_mask_streaming(*args, **kwargs)
                except Exception as exc:
                    log.warning(  # noqa: G200
                        "NPU streaming create_block_mask failed; falling back to "
                        "the PyTorch implementation: %s: %s",
                        type(exc).__name__,
                        str(exc),
                    )
                    block_mask = current_create_block_mask(*args, **kwargs)
            else:
                block_mask = current_create_block_mask(*args, **kwargs)
            try:
                kernel_options = infer_eager_block_mask_kernel_options(block_mask)
                merged_kernel_options = dict(kernel_options) if kernel_options else {}
                setattr(  # noqa: B010
                    block_mask,
                    "_npu_flex_attention_kernel_options",
                    merged_kernel_options,
                )
                if kernel_options:
                    log.info(
                        "[flex_attention][create_block_mask] cached kernel options: %s",
                        merged_kernel_options,
                    )
            except Exception as exc:
                log.debug(  # noqa: G200
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

    These lowerings allow supported fallback operations to be lowered as pointwise
    ops in the score_mod and mask_mod subgraphs.
    """
    from torch._inductor.lowering import make_pointwise, index_impl

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

    remainder_fn = make_pointwise(ops.remainder)

    def integer_remainder(a, b):
        # Avoid ops.remainder here. Vector integer remainder currently lowers to
        # arith.remsi on NPU and can produce incorrect FlexAttention score_mod
        # results. Build Python remainder from truncating division instead:
        #   r = a - trunc(a / b) * b
        # and adjust r when its sign differs from the divisor.
        quotient = ops.truncdiv(a, b)
        remainder = ops.sub(a, ops.mul(quotient, b))
        zero = ops.constant(0, torch.int32)
        needs_adjustment = ops.and_(
            ops.ne(remainder, zero),
            ops.ne(ops.lt(remainder, zero), ops.lt(b, zero)),
        )
        return ops.where(needs_adjustment, ops.add(remainder, b), remainder)

    integer_remainder_fn = make_pointwise(integer_remainder)

    def remainder_scalar(a, b):
        if not a.get_dtype().is_floating_point and isinstance(b, int):
            return integer_remainder_fn(a, b)
        return remainder_fn(a, b)

    additional_lowerings[aten.bitwise_and.Tensor] = bitwise_and_tensor
    additional_lowerings[aten.bitwise_or.Tensor] = bitwise_or_tensor
    additional_lowerings[aten.bitwise_not.default] = bitwise_not_default
    additional_lowerings[aten.remainder.Scalar] = remainder_scalar

    return additional_lowerings


def _build_subgraph_buffer_with_additional_lowerings(args, subgraph):
    """
    Build subgraph buffer with additional lowerings for flex_attention.

    This function creates a PointwiseSubgraphLowering with additional_lowerings
    to handle supported fallback operations as pointwise ops.
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


def _sequence_lengths_are_statically_divisible(seq_len_q, seq_len_kv) -> bool:
    return V.graph.sizevars.statically_known_multiple_of(
        seq_len_q, 128
    ) and V.graph.sizevars.statically_known_multiple_of(seq_len_kv, 128)


def _validate_device(query, key, value):
    return


def _get_num_cube_core() -> int:
    return max(int(getattr(npu_config, "num_cube_core", 1)), 1)


def _get_num_vector_core() -> int:
    return max(int(getattr(npu_config, "num_vector_core", 1)), 1)


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

    used_buffers = []
    unused_buffer_names = []
    for buffer in other_buffers:
        if isinstance(buffer, sympy.Expr):
            continue
        if not _is_named_ir_node(buffer):
            raise TypeError(
                "flex attention captured input must be an IR buffer or sympy expression, "
                f"got {type(buffer).__name__}"
            )

        get_name = getattr(buffer, "get_name", None)
        buffer_name = None
        if get_name is not None:
            try:
                buffer_name = get_name()
            except (AssertionError, NotImplementedError):
                buffer_name = None

        if not read_names or buffer_name is None or buffer_name in read_names:
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
        "DQ_LAUNCH_PROGRAMS": launch_programs,
    }


def _unpack_npu_block_mask(block_mask):
    if len(block_mask) == 13:
        return block_mask
    if len(block_mask) == 17:
        return (*block_mask[:10], *block_mask[14:])
    raise ValueError(
        f"Unsupported FlexAttention BlockMask tuple length: {len(block_mask)}"
    )


def _lower_flex_attention_mask_in(
    *,
    query,
    key,
    value,
    scale,
    kernel_options,
    subgraph_buffer,
    mask_graph_buffer,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    kv_num_blocks,
    kv_indices,
    q_num_blocks,
    q_indices,
    full_kv_num_blocks,
    full_kv_indices,
    full_q_num_blocks,
    full_q_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
):
    """Lower mask-in forward with PyTorch's Triton template and NPU tiling."""
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
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)), (
        "Query length must be greater than 0"
    )
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)), (
        "Key length must be greater than 0"
    )

    kernel_options.setdefault(
        "IS_DIVISIBLE",
        _sequence_lengths_are_statically_divisible(seq_len_q, seq_len_kv),
    )

    out_size = [Bq, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        out_size,
        stride=[sympy.sympify(s) for s in out_strides],
    )
    logsumexp = empty_strided(
        [Bq, Hq, seq_len_q],
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )
    max_scores = empty_strided(
        [Bq, Hq, seq_len_q],
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )
    kernel_options.setdefault("SM_SCALE", scale)
    kernel_options.setdefault("GQA_SHARED_HEADS", Hq // Hkv)

    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    set_head_dim_values(
        kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars
    )
    sparse_q_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_q_block_size
    )
    sparse_kv_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_kv_block_size
    )
    configs = generate_fwd_candidate_configs(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )
    if not configs:
        raise RuntimeError(
            "No compatible mask-in forward tiling configs for "
            f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size} and "
            f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}."
        )

    input_nodes = [
        query,
        key,
        value,
        logsumexp,
        max_scores,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ]
    choices: list[Any] = []
    original_kernel_options = kernel_options.copy()
    for cfg in configs:
        cur_kernel_options = original_kernel_options.copy()
        cur_kernel_options.update(cfg)
        cur_kernel_options.setdefault(
            "SPARSE_Q_BLOCK_SIZE", sparse_q_block_size
        )
        cur_kernel_options.setdefault(
            "SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size
        )
        choice_count = len(choices)
        error = upstream_flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            subgraphs=[subgraph_buffer, mask_graph_buffer],
            mutated_inputs=[logsumexp, max_scores],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error
        if len(choices) > choice_count:
            _tag_flex_attention_report_choices(
                choices[choice_count:], cfg
            )
            if prefer_max_tiling_without_benchmark():
                _tag_choice_attr(
                    choices[choice_count:],
                    "_nobench_select_first_compilable",
                    True,
                )

    inputs_for_autotuning = (
        input_nodes
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )
    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }
    out, _ = autotune_select_algorithm(
        "flex_attention",
        choices,
        [x for x in inputs_for_autotuning if _is_named_ir_node(x)],
        layout,
        input_gen_fns=input_gen_fns,
    )
    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )
    return (out, logsumexp, max_scores)


def _lower_flex_attention_backward_mask_in(
    *,
    query,
    key,
    value,
    out,
    logsumexp,
    grad_out,
    grad_logsumexp,
    kernel_options,
    scale,
    fw_subgraph_buffer,
    joint_outputs,
    mask_graph_buffer,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    kv_num_blocks,
    kv_indices,
    q_num_blocks,
    q_indices,
    full_kv_num_blocks,
    full_kv_indices,
    full_q_num_blocks,
    full_q_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
):
    """Lower mask-in backward with PyTorch's fused Triton template."""
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())
    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
    )

    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[prims.convert_element_type](delta, torch.float32)
    if grad_logsumexp is not None:
        grad_lse_exp2 = lowerings[aten.mul](
            grad_logsumexp, 1 / math.log(2)
        )
        grad_lse_exp2 = ExternKernel.require_contiguous(grad_lse_exp2)
        delta = lowerings[aten.sub](delta, grad_lse_exp2)
        delta = ExternKernel.require_contiguous(delta)
        delta, grad_lse_exp2 = maybe_realize([delta, grad_lse_exp2])
    else:
        delta = ExternKernel.require_contiguous(delta)
        (delta,) = maybe_realize([delta])

    query_size = [Bq, Hq, seq_len_q, qk_head_dim]
    grad_query_strides = infer_dense_strides(
        query_size, query.get_stride()
    )
    grad_query = empty_strided(
        query_size,
        stride=[sympy.sympify(s) for s in grad_query_strides],
        dtype=query.get_dtype(),
        device=query.get_device(),
    )
    value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())
    broadcasted_grad_value = empty_strided(
        value_size,
        stride=[sympy.sympify(s) for s in value_strides],
        dtype=value.get_dtype(),
        device=value.get_device(),
    )

    kernel_options = kernel_options.copy()
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    kernel_options.setdefault("SM_SCALE", scale)
    kernel_options.setdefault("GQA_SHARED_HEADS", Hq // Hkv)
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        (
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = (empty(0, device=query.get_device()) for _ in range(4))

    set_head_dim_values(
        kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars
    )
    sparse_q_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_q_block_size
    )
    sparse_kv_block_size = V.graph.sizevars.evaluate_static_shape(
        sparse_kv_block_size
    )
    configs = generate_bwd_candidate_configs(
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        mode=FlexMode.BWD,
    )
    if not configs:
        raise RuntimeError(
            "No compatible mask-in backward tiling configs for "
            f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size} and "
            f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}."
        )

    input_nodes = [
        query,
        key,
        value,
        logsumexp,
        delta,
        grad_out,
        grad_query,
        broadcasted_grad_value,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        full_kv_num_blocks,
        full_kv_indices,
        full_q_num_blocks,
        full_q_indices,
    ]
    choices: list[Any] = []
    for cfg in configs:
        cur_kernel_options = kernel_options.copy()
        cur_kernel_options.update(cfg)
        cur_kernel_options.setdefault(
            "SPARSE_Q_BLOCK_SIZE", sparse_q_block_size
        )
        cur_kernel_options.setdefault(
            "SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size
        )
        choice_count = len(choices)
        upstream_flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout_broadcasted_k,
            subgraphs=[
                fw_subgraph_buffer,
                joint_outputs.grad_input,
                mask_graph_buffer,
                joint_outputs.captured_grads_compute,
            ],
            mutated_inputs=[
                grad_query,
                broadcasted_grad_value,
                *joint_outputs.mutated_grads,
            ],
            call_sizes=query.get_size() + key.get_size()[1:3],
            **cur_kernel_options,
        )
        if len(choices) > choice_count:
            _tag_flex_attention_report_choices(
                choices[choice_count:], cfg
            )
            if prefer_max_tiling_without_benchmark():
                _tag_choice_attr(
                    choices[choice_count:],
                    "_nobench_select_first_compilable",
                    True,
                )

    if not choices:
        raise RuntimeError(
            f"All {len(configs)} mask-in backward configs failed to compile."
        )
    inputs_for_autotuning = (
        input_nodes
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
        + joint_outputs.mutated_grads
    )
    input_gen_fns = {
        8: create_num_blocks_fake_generator(kv_indices),
        9: create_indices_fake,
        10: create_num_blocks_fake_generator(q_indices),
        11: create_indices_fake,
        12: create_num_blocks_fake_generator(full_kv_indices),
        13: create_indices_fake,
        14: create_num_blocks_fake_generator(full_q_indices),
        15: create_indices_fake,
    }
    broadcasted_grad_key, _ = autotune_select_algorithm(
        "flex_attention_backward",
        choices,
        [x for x in inputs_for_autotuning if _is_named_ir_node(x)],
        layout_broadcasted_k,
        input_gen_fns=input_gen_fns,
    )
    broadcasted_grad_key.data.data.subgraph_inps = list(
        score_mod_other_buffers
    ) + list(mask_mod_other_buffers)
    broadcasted_grad_key.data.data.subgraph_outs = get_bwd_subgraph_outputs(
        fw_subgraph_buffer, mask_graph_buffer, joint_outputs
    )

    if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value
    else:
        assert V.graph.sizevars.evaluate_expr(
            sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)
        ), f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        grad_key = lowerings[aten.sum](
            broadcasted_grad_key, axis=0, keepdims=True
        )
        grad_value = lowerings[aten.sum](
            broadcasted_grad_value, axis=0, keepdims=True
        )

    captured_grads = tuple(
        to_dtype(grad, original.get_dtype())
        if grad is not None and grad.get_dtype() != original.get_dtype()
        else grad
        for grad, original in zip(
            joint_outputs.captured_grads, score_mod_other_buffers
        )
    )
    return (grad_query, grad_key, grad_value, captured_grads)


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
        small_dqk = V.graph.sizevars.evaluate_expr(
            sympy.Lt(query.get_size()[-1], 16)
        )
        small_dv = V.graph.sizevars.evaluate_expr(
            sympy.Lt(value.get_size()[-1], 16)
        )
        if small_dqk or small_dv:
            raise NotImplementedError(
                "NYI: embedding dimension of the query, key, and value must be "
                f"at least 16 but got E={query.get_size()[-1]} and "
                f"Ev={value.get_size()[-1]}"
            )

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
        ) = _unpack_npu_block_mask(block_mask)

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
        # torch 2.10 upstream flex_attention adds kernel options that the NPU
        # templates do not consume (they are GPU backend dispatch knobs).
        # Strip them before they leak into triton constexprs and cause
        # NameError("AUTO is not defined") during compilation.
        # NB: WRITE_DQ is kept (consumed by NPU templates); OUTPUT_MAX is
        # stripped (torch-2.10-only knob absent in 2.7 codegen).
        for _unsupported in ("BACKEND", "OUTPUT_MAX", "generate_with_caching"):
            kernel_options.pop(_unsupported, None)
        has_explicit_score_mod = bool(
            kernel_options.pop(_EXPLICIT_SCORE_MOD_OPTION, False)
        )
        # Mark symbols in custom kernel options as static shapes and add guards.
        kernel_options = {
            k: V.graph.sizevars.guard_int(v)
            if isinstance(v, sympy.Symbol)
            else v
            for k, v in kernel_options.items()
        }
        kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
        score_mod_is_identity = _is_score_mod_identity_graph(subgraph)
        configured_mask_out = bool(
            npu_config.flex_attention.flexattention_mask_out
        )
        use_mask_out = (
            configured_mask_out
            and (has_explicit_score_mod and score_mod_is_identity)
            and _has_sparse_block_mask(
                SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE
            )
        )
        if not use_mask_out:
            if not bool(
                kernel_options.get(
                    "HAS_FULL_BLOCKS", full_kv_num_blocks is not None
                )
            ):
                full_kv_num_blocks = None
                full_kv_indices = None
                full_q_num_blocks = None
                full_q_indices = None
            for option in (
                "SPARSE_MASK_HEAD_SHARED",
                "SPARSE_MASK_HQ",
                "SPARSE_MASK_MAX_NORMAL_BLOCKS",
            ):
                kernel_options.pop(option, None)
            return _lower_flex_attention_mask_in(
                query=query,
                key=key,
                value=value,
                scale=scale,
                kernel_options=kernel_options,
                subgraph_buffer=subgraph_buffer,
                mask_graph_buffer=mask_graph_buffer,
                score_mod_other_buffers=score_mod_other_buffers,
                mask_mod_other_buffers=mask_mod_other_buffers,
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                full_q_num_blocks=full_q_num_blocks,
                full_q_indices=full_q_indices,
                sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
                sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            )

        kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
        kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
        kernel_options["TORCHINDUCTOR_FLEXATTENTION_MASKOUT"] = True

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

        kernel_options.setdefault(
            "IS_DIVISIBLE",
            _sequence_lengths_are_statically_divisible(seq_len_q, seq_len_kv),
        )

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
        max_scores = empty_strided(
            logsumexp_shape,
            None,
            dtype=torch.float32,
            device=query.get_device(),
        )
        kernel_options.setdefault("SM_SCALE", scale)

        # Determine GQA broadcast factor.
        gqa_shared_heads = Hq // Hkv
        kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

        flexattention_mask_out = use_mask_out
        has_full_blocks = full_kv_num_blocks is not None
        has_full_blocks = bool(
            kernel_options.get("HAS_FULL_BLOCKS", has_full_blocks)
        )
        kernel_options["HAS_FULL_BLOCKS"] = has_full_blocks
        if not has_full_blocks:
            full_kv_num_blocks, full_kv_indices = (
                empty(0, device=query.get_device()) for _ in range(2)
            )

        set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

        # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
        SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
        SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)

        compact_q_offsets = None
        compact_flat_to_row = None
        compact_flat_to_blk = None
        runtime_total_blocks = None
        actual_blocks = None
        sparse_mask_layout = None
        sparse_mask_buffer = None
        sparse_mask_strides = None

        if flexattention_mask_out:
            (
                compact_q_offsets,
                runtime_total_blocks,
                row_count,
            ) = _build_runtime_compact_sparse_mask_offsets(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                device=query.get_device(),
                context="forward",
            )
            max_runtime_blocks = torch.iinfo(torch.int32).max // (
                SPARSE_Q_BLOCK_SIZE * SPARSE_KV_BLOCK_SIZE
            )
            actual_blocks = _bind_runtime_total_blocks_as_unbacked_size(
                runtime_total_blocks,
                max_blocks=max_runtime_blocks,
            )
            compact_flat_to_row = empty_strided(
                [actual_blocks],
                [1],
                dtype=torch.int32,
                device=query.get_device(),
            )
            compact_flat_to_blk = empty_strided(
                [actual_blocks],
                [1],
                dtype=torch.int32,
                device=query.get_device(),
            )
            _build_runtime_compact_sparse_mask_mapping(
                flat_to_row=compact_flat_to_row,
                flat_to_blk=compact_flat_to_blk,
                q_offsets=compact_q_offsets,
                kv_num_blocks=kv_num_blocks,
                row_count=row_count,
                device=query.get_device(),
                context="forward",
            )
            sparse_mask_size = [
                actual_blocks,
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
            kernel_options.setdefault(
                "SPARSE_MASK_STRIDE_BLK", sparse_mask_strides[0]
            )
            kernel_options.setdefault(
                "SPARSE_MASK_STRIDE_M", sparse_mask_strides[1]
            )

        fwd_call_size_hints = V.graph.sizevars.optimization_hints(
            query.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        fwd_batch_size_hint, fwd_q_heads_hint, fwd_num_queries_hint, _ = fwd_call_size_hints
        fwd_num_cube_core = _get_num_cube_core()

        log.debug(
            "flex_attention lowering: query=%s key=%s value=%s SPARSE_Q=%s SPARSE_KV=%s kernel_options=%s",
            query.get_size(), key.get_size(), value.get_size(),
            SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE, kernel_options)

        # Validate benchmark configuration before autotuning
        log.debug("Benchmark Configuration Validation")
        validate_benchmark_config()  # Now only warns, doesn't raise errors

        choices: list[Any] = []
        assert sparse_mask_buffer is not None
        assert compact_q_offsets is not None
        forward_input_nodes = [
            query,
            key,
            value,
            sparse_mask_buffer,
            compact_q_offsets,
            kv_num_blocks,
            kv_indices,
            logsumexp,
            max_scores,
            full_kv_num_blocks,
            full_kv_indices,
        ]

        dict_configs = generate_fwd_candidate_configs(
            sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
            sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
        )

        if not dict_configs:
            raise RuntimeError(
                "No compatible flex attention forward tiling configs for "
                f"SPARSE_Q_BLOCK_SIZE={SPARSE_Q_BLOCK_SIZE} and "
                f"SPARSE_KV_BLOCK_SIZE={SPARSE_KV_BLOCK_SIZE}."
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

            log.debug(
                "Processing config: BLOCK_M=%d BLOCK_N=%d",
                BLOCK_M,
                BLOCK_N,
            )

            cur_kernel_options = original_kernel_options.copy()

            # Generated tiling config is authoritative for this choice.
            cur_kernel_options.update(cfg)

            # Blocksparse options
            cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
            cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
            cur_kernel_options.setdefault(
                "TORCHINDUCTOR_FLEXATTENTION_MASKOUT",
                True,
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
                "fwd-choice: cfg=%s grid=(%d,%d,%d) total_programs=%d aicore_count=%d"
                "exceeds_aicores=%s wave_estimate=%d kv_num_blocks_type=%s kv_indices_type=%s final_kernel_options=%s",
                cfg, fwd_grid_x, fwd_grid_y, fwd_grid_z, fwd_total_programs,
                fwd_num_cube_core, fwd_total_programs > fwd_num_cube_core,
                fwd_wave_estimate,
                type(kv_num_blocks).__name__,
                type(kv_indices).__name__, cur_kernel_options)


            try:
                forward_kernel_options = cur_kernel_options.copy()
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
                    error = flex_attention_fwd_mask_out.maybe_append_choice(
                        choices=choices,
                        input_nodes=forward_input_nodes,
                        layout=layout,
                        subgraphs=[subgraph_buffer],
                        mutated_inputs=[logsumexp, max_scores],
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
                    cfg,
                )
                if prefer_max_tiling_without_benchmark():
                    _tag_choice_attr(
                        choices[choice_count:],
                        "_nobench_select_first_compilable",
                        True,
                    )
            except Exception as e:
                # Catch compilation errors and skip this config
                log.warning("Config %s compilation failed: %s: %s", cfg, type(e).__name__, str(e)[:200])
                # Continue to next config instead of raising
                continue

        sparse_mask_choices = []
        sparse_mask_base_kernel_options = {
            "SPARSE_Q_BLOCK_SIZE": SPARSE_Q_BLOCK_SIZE,
            "SPARSE_KV_BLOCK_SIZE": SPARSE_KV_BLOCK_SIZE,
            "SPARSE_MASK_STRIDE_BLK": sparse_mask_strides[0],
            "SPARSE_MASK_STRIDE_M": sparse_mask_strides[1],
        }

        sparse_mask_tiling_configs = build_sparse_mask_candidate_configs(
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )
        log.info(
            "Generated %d sparse mask kernel tiling configs from "
            "SPARSE_Q_BLOCK_SIZE=%d, SPARSE_KV_BLOCK_SIZE=%d: %s",
            len(sparse_mask_tiling_configs),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            sparse_mask_tiling_configs,
        )

        for sparse_mask_tiling_config in sparse_mask_tiling_configs:
            sparse_mask_kernel_options = sparse_mask_base_kernel_options.copy()
            sparse_mask_kernel_options.update(sparse_mask_tiling_config)
            num_choices_before = len(sparse_mask_choices)
            sparse_mask_template = flex_attention_fwd_mask_compact
            sparse_mask_input_nodes = [
                sparse_mask_buffer,
                compact_flat_to_row,
                compact_flat_to_blk,
                query,
                key,
                kv_indices,
            ]
            try:
                sparse_mask_template.maybe_append_choice(
                    choices=sparse_mask_choices,
                    input_nodes=sparse_mask_input_nodes,
                    layout=sparse_mask_layout,
                    subgraphs=[mask_graph_buffer],
                    mutated_inputs=[sparse_mask_buffer],
                    call_sizes=[actual_blocks],
                    NUM_VECTOR_CORE=_get_num_vector_core(),
                    **sparse_mask_kernel_options,
                )
                if len(sparse_mask_choices) > num_choices_before:
                    _tag_choice_configs(
                        sparse_mask_choices[num_choices_before:],
                        "_sparse_mask_report_config",
                        sparse_mask_tiling_config,
                    )
                    _tag_choices_for_no_benchmark(
                        sparse_mask_choices[num_choices_before:]
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
            5: create_sparse_mask_num_blocks_fake,
            6: _create_sparse_mask_indices_fake_generator(),
            9: create_num_blocks_fake_generator(full_kv_indices),
            10: create_indices_fake,
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
        sparse_mask_inputs_for_autotuning = (
            [
                sparse_mask_buffer,
                compact_flat_to_row,
                compact_flat_to_blk,
                query,
                key,
                kv_indices,
            ]
            + sparse_mask_autotune_other_buffers
        )
        sparse_mask_input_gen_fns = {
            1: create_zero_int_tensor_fake,
            2: create_zero_int_tensor_fake,
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

        result, _ = autotune_select_algorithm(
            "flex_attention",
            choices,
            inputs_for_autotuning,
            layout,
            input_gen_fns=input_gen_fns,
        )

        return (
            result,
            lowerings[aten.mul](logsumexp, _LOG2E),
            lowerings[aten.mul](max_scores, _LOG2E),
        )


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
        ) = _unpack_npu_block_mask(block_mask)

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
        # torch 2.10 upstream adds GPU dispatch knobs the NPU templates don't
        # consume; strip them before they leak into triton constexprs.
        # NB: WRITE_DQ is kept (consumed by NPU templates); OUTPUT_MAX is
        # stripped (torch-2.10-only knob absent in 2.7 codegen).
        for _unsupported in ("BACKEND", "OUTPUT_MAX", "generate_with_caching"):
            kernel_options.pop(_unsupported, None)
        has_explicit_score_mod = bool(
            kernel_options.pop(_EXPLICIT_SCORE_MOD_OPTION, False)
        )
        configured_mask_out = bool(
            npu_config.flex_attention.flexattention_mask_out
        )
        # Mark symbols in custom kernel options as static shapes and add guards.
        kernel_options = {
            k: V.graph.sizevars.guard_int(v)
            if isinstance(v, sympy.Symbol)
            else v
            for k, v in kernel_options.items()
        }
        kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
        kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
        kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
        kernel_options.setdefault(
            "IS_DIVISIBLE",
            _sequence_lengths_are_statically_divisible(seq_len_q, seq_len_kv),
        )

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
        joint_placeholder_inps = fwd_placeholder_inps + [
            create_placeholder("grad_score_mod", dtype, device)
        ]
        # Sometimes we have weird unused nodes here
        joint_graph.graph_module.graph.eliminate_dead_code()

        # It is hard to raise nice errors for some joint graphs during subgraph lowering
        # This lets us do some checks before attempting to lower
        validate_joint_graph(joint_graph.graph_module.graph)
        log.debug(
            "flex_attention_backward joint_graph:\n%s",
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

        flexattention_mask_out = (
            configured_mask_out
            and (has_explicit_score_mod and score_mod_is_identity)
            and _has_sparse_block_mask(
                SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE
            )
        )
        log.info(
            "flex_attention_backward mask route: configured_mask_out=%s "
            "has_explicit_score_mod=%s score_mod_is_identity=%s "
            "flexattention_mask_out=%s",
            configured_mask_out,
            has_explicit_score_mod,
            score_mod_is_identity,
            flexattention_mask_out,
        )
        if not flexattention_mask_out:
            if not bool(
                kernel_options.get(
                    "HAS_FULL_BLOCKS", full_kv_num_blocks is not None
                )
            ):
                full_kv_num_blocks = None
                full_kv_indices = None
                full_q_num_blocks = None
                full_q_indices = None
            for option in (
                "SPARSE_MASK_HEAD_SHARED",
                "SPARSE_MASK_HQ",
                "SPARSE_MASK_MAX_NORMAL_BLOCKS",
            ):
                kernel_options.pop(option, None)
            return _lower_flex_attention_backward_mask_in(
                query=query,
                key=key,
                value=value,
                out=out,
                logsumexp=logsumexp,
                grad_out=grad_out,
                grad_logsumexp=grad_logsumexp,
                kernel_options=kernel_options,
                scale=scale,
                fw_subgraph_buffer=fw_subgraph_buffer,
                joint_outputs=joint_outputs,
                mask_graph_buffer=mask_graph_buffer,
                score_mod_other_buffers=score_mod_other_buffers,
                mask_mod_other_buffers=mask_mod_other_buffers,
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                full_q_num_blocks=full_q_num_blocks,
                full_q_indices=full_q_indices,
                sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
                sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            )

        kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
        kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
        kernel_options["TORCHINDUCTOR_FLEXATTENTION_MASKOUT"] = True

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

        # Saved statistics use log2 at the PyTorch boundary; NPU templates use
        # natural logarithms internally.
        logsumexp = lowerings[aten.mul](logsumexp, _LN2)
        mul_delta = lowerings[aten.mul](out, grad_out)
        delta = lowerings[aten.sum](mul_delta, axis=-1)
        delta = lowerings[prims.convert_element_type](delta, torch.float32)
        grad_lse = (
            lowerings[aten.mul](grad_logsumexp, _LOG2E)
            if grad_logsumexp is not None
            else None
        )
        if grad_lse is not None:
            grad_lse = lowerings[aten.mul](grad_lse, _LOG2E)
            delta = lowerings[aten.sub](delta, grad_lse)
            delta = ExternKernel.require_contiguous(delta)
            logsumexp, grad_lse, delta = maybe_realize([logsumexp, grad_lse, delta])
        else:
            delta = ExternKernel.require_contiguous(delta)
            (delta,) = maybe_realize([delta])
        (logsumexp,) = maybe_realize([logsumexp])

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

        set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

        SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
        SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)

        compact_q_offsets = None
        compact_flat_to_row = None
        compact_flat_to_blk = None
        runtime_total_blocks = None
        actual_blocks = None

        sparse_z = kv_num_blocks.get_size()[0]
        metadata_sparse_hq = kv_num_blocks.get_size()[1]
        bwd_dynamic_dims = (
            *query.get_size()[:3],
            *key.get_size()[:3],
            *kv_num_blocks.get_size(),
            kv_indices.get_size()[3],
        )
        bwd_has_dynamic_shape = any(
            bool(getattr(dim, "free_symbols", ())) for dim in bwd_dynamic_dims
        )
        sparse_z_val = V.graph.sizevars.optimization_hint(
            sparse_z,
            fallback=config.unbacked_symint_fallback,
        )
        sparse_mask_hq_val = V.graph.sizevars.optimization_hint(
            metadata_sparse_hq,
            fallback=config.unbacked_symint_fallback,
        )

        bwd_sparse_mask_layout = None
        bwd_sparse_mask_buffer = None
        bwd_sparse_mask_strides = None
        bwd_sparse_mask_block_pos_layout = None
        bwd_sparse_mask_block_pos_buffer = None
        bwd_sparse_mask_block_pos_strides = None
        if flexattention_mask_out:
            (
                compact_q_offsets,
                runtime_total_blocks,
                row_count,
            ) = _build_runtime_compact_sparse_mask_offsets(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                device=query.get_device(),
                context="backward",
            )
            max_runtime_blocks = torch.iinfo(torch.int32).max // (
                SPARSE_Q_BLOCK_SIZE * SPARSE_KV_BLOCK_SIZE
            )
            actual_blocks = _bind_runtime_total_blocks_as_unbacked_size(
                runtime_total_blocks,
                max_blocks=max_runtime_blocks,
            )
            compact_flat_to_row = empty_strided(
                [actual_blocks],
                [1],
                dtype=torch.int32,
                device=query.get_device(),
            )
            compact_flat_to_blk = empty_strided(
                [actual_blocks],
                [1],
                dtype=torch.int32,
                device=query.get_device(),
            )
            _build_runtime_compact_sparse_mask_mapping(
                flat_to_row=compact_flat_to_row,
                flat_to_blk=compact_flat_to_blk,
                q_offsets=compact_q_offsets,
                kv_num_blocks=kv_num_blocks,
                row_count=row_count,
                device=query.get_device(),
                context="backward",
            )
            bwd_sparse_mask_size = [
                actual_blocks,
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
            bwd_sparse_mask_buffer = empty_strided(
                bwd_sparse_mask_size,
                stride=bwd_sparse_mask_strides,
                dtype=torch.bool,
                device=query.get_device(),
            )
            kernel_options.setdefault(
                "SPARSE_MASK_STRIDE_BLK", bwd_sparse_mask_strides[0]
            )
            kernel_options.setdefault(
                "SPARSE_MASK_STRIDE_M", bwd_sparse_mask_strides[1]
            )

            num_sparse_kv_blocks = (
                seq_len_kv + SPARSE_KV_BLOCK_SIZE - 1
            ) // SPARSE_KV_BLOCK_SIZE
            bwd_sparse_mask_block_pos_size = [
                *kv_num_blocks.get_size(),
                num_sparse_kv_blocks,
            ]
            bwd_sparse_mask_block_pos_strides = [
                metadata_sparse_hq
                * kv_num_blocks.get_size()[2]
                * num_sparse_kv_blocks,
                kv_num_blocks.get_size()[2] * num_sparse_kv_blocks,
                num_sparse_kv_blocks,
                1,
            ]
            bwd_sparse_mask_block_pos_layout = FixedLayout(
                query.get_device(),
                torch.int32,
                bwd_sparse_mask_block_pos_size,
                stride=[
                    sympy.sympify(s)
                    for s in bwd_sparse_mask_block_pos_strides
                ],
            )
            bwd_sparse_mask_block_pos_buffer = empty_strided(
                bwd_sparse_mask_block_pos_size,
                stride=bwd_sparse_mask_block_pos_strides,
                dtype=torch.int32,
                device=query.get_device(),
            )
            bwd_sparse_mask_block_pos_buffer = _force_fixed_layout(
                lowerings[aten.fill_](bwd_sparse_mask_block_pos_buffer, -1),
                bwd_sparse_mask_block_pos_strides,
            )

        bwd_query_call_size_hints = V.graph.sizevars.optimization_hints(
            query.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        bwd_key_call_size_hints = V.graph.sizevars.optimization_hints(
            key.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        (
            bwd_batch_size_hint,
            bwd_q_heads_hint,
            bwd_num_queries_hint,
            _,
        ) = bwd_query_call_size_hints
        (
            bwd_kv_batch_size_hint,
            bwd_kv_heads_hint,
            bwd_num_key_value_hint,
            _,
        ) = bwd_key_call_size_hints
        bwd_num_cube_core = _get_num_cube_core()

        log.debug(
            "flex_attention_backward lowering: query=%s key=%s SPARSE_Q=%s SPARSE_KV=%s",
            query.get_size(), key.get_size(), SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE,
        )

        dq_choices: list[Any] = []
        dkdv_choices: list[Any] = []
        bwd_sparse_mask_choices = []
        bwd_sparse_mask_block_pos_choices = []
        bwd_sparse_mask_base_kernel_options = {}
        bwd_sparse_mask_tiling_configs = []
        if flexattention_mask_out:
            assert bwd_sparse_mask_strides is not None
            bwd_sparse_mask_base_kernel_options = {
                "SPARSE_Q_BLOCK_SIZE": SPARSE_Q_BLOCK_SIZE,
                "SPARSE_KV_BLOCK_SIZE": SPARSE_KV_BLOCK_SIZE,
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
            assert compact_flat_to_row is not None
            assert compact_flat_to_blk is not None
            assert bwd_sparse_mask_layout is not None
            flex_attention_bwd_mask_compact.maybe_append_choice(
                choices=bwd_sparse_mask_choices,
                input_nodes=[
                    compact_flat_to_row,
                    compact_flat_to_blk,
                    query,
                    key,
                    kv_indices,
                ],
                layout=bwd_sparse_mask_layout,
                subgraphs=[mask_graph_buffer],
                call_sizes=[actual_blocks],
                NUM_VECTOR_CORE=_get_num_vector_core(),
                **bwd_sparse_mask_kernel_options,
            )
            if len(bwd_sparse_mask_choices) > bwd_sparse_mask_choice_count:
                _tag_choice_configs(
                    bwd_sparse_mask_choices[bwd_sparse_mask_choice_count:],
                    "_bwd_sparse_mask_report_config",
                    bwd_sparse_mask_tiling_config,
                )
                _tag_choices_for_no_benchmark(
                    bwd_sparse_mask_choices[bwd_sparse_mask_choice_count:]
                )
                log.info(
                    "Backward sparse mask kernel choice created successfully: %s",
                    bwd_sparse_mask_tiling_config,
                )
        if not bwd_sparse_mask_choices and flexattention_mask_out:
            raise RuntimeError(
                "Backward mask-out mode could not create compact sparse mask kernel choices."
            )

        if flexattention_mask_out:
            assert bwd_sparse_mask_block_pos_strides is not None
            assert bwd_sparse_mask_block_pos_layout is not None
            assert bwd_sparse_mask_block_pos_buffer is not None
            bwd_sparse_mask_block_pos_kernel_options = {
                "num_stages": 1,
                "num_warps": 4,
            }
            flex_attention_bwd_mask_pos.maybe_append_choice(
                choices=bwd_sparse_mask_block_pos_choices,
                input_nodes=[
                    compact_flat_to_row,
                    compact_flat_to_blk,
                    kv_indices,
                    bwd_sparse_mask_block_pos_buffer,
                ],
                layout=bwd_sparse_mask_block_pos_layout,
                mutated_inputs=[bwd_sparse_mask_block_pos_buffer],
                call_sizes=[actual_blocks],
                NUM_VECTOR_CORE=_get_num_vector_core(),
                **bwd_sparse_mask_block_pos_kernel_options,
            )
        if (
            not bwd_sparse_mask_block_pos_choices
            and flexattention_mask_out
        ):
            raise RuntimeError(
                "Backward mask-out mode could not create sparse mask block-position choices."
            )

        mask_out_input_nodes = []
        if flexattention_mask_out:
            assert compact_q_offsets is not None
            assert compact_flat_to_row is not None
            assert compact_flat_to_blk is not None
            assert bwd_sparse_mask_layout is not None
            assert bwd_sparse_mask_block_pos_layout is not None
            assert bwd_sparse_mask_block_pos_buffer is not None
            bwd_sparse_mask_autotune_other_buffers = (
                _filter_used_subgraph_buffers(
                    mask_graph_buffer,
                    mask_mod_other_buffers,
                )
            )
            bwd_sparse_mask_inputs_for_autotuning = [
                compact_flat_to_row,
                compact_flat_to_blk,
                query,
                key,
                kv_indices,
                *bwd_sparse_mask_autotune_other_buffers,
            ]
            bwd_sparse_mask_input_gen_fns = {
                0: create_zero_int_tensor_fake,
                1: create_zero_int_tensor_fake,
                4: _create_sparse_mask_indices_fake_generator(),
            }
            bwd_sparse_mask_result, _ = autotune_select_algorithm(
                "bwd_sparse_mask_kernel_compact",
                bwd_sparse_mask_choices,
                bwd_sparse_mask_inputs_for_autotuning,
                bwd_sparse_mask_layout,
                input_gen_fns=bwd_sparse_mask_input_gen_fns,
            )
            log.info(
                "Backward compact sparse mask kernel autotune completed "
                "with %d choices: %s",
                len(bwd_sparse_mask_choices),
                bwd_sparse_mask_result,
            )

            bwd_sparse_mask_block_pos_result, _ = autotune_select_algorithm(
                "sparse_mask_block_pos",
                bwd_sparse_mask_block_pos_choices,
                [
                    compact_flat_to_row,
                    compact_flat_to_blk,
                    kv_indices,
                    bwd_sparse_mask_block_pos_buffer,
                ],
                bwd_sparse_mask_block_pos_layout,
                input_gen_fns={
                    0: create_zero_int_tensor_fake,
                    1: create_zero_int_tensor_fake,
                    2: _create_sparse_mask_indices_fake_generator(),
                    3: create_minus_one_int_tensor_fake,
                },
            )
            log.info(
                "Sparse mask block-position kernel autotune completed "
                "with %d choices: %s",
                len(bwd_sparse_mask_block_pos_choices),
                bwd_sparse_mask_block_pos_result,
            )
            mask_out_input_nodes = [
                bwd_sparse_mask_result,
                compact_q_offsets,
                bwd_sparse_mask_block_pos_buffer,
            ]

        bwd_dq_dict_configs = generate_bwd_candidate_configs(
            sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
            sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            mode=FlexMode.BWDDQ,
        )
        bwd_dkdv_dict_configs = generate_bwd_candidate_configs(
            sparse_q_block_size=SPARSE_Q_BLOCK_SIZE,
            sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
            mode=FlexMode.BWDDKDV,
        )

        log.debug(
            "bwd dict_configs count: dq=%d dkdv=%d",
            len(bwd_dq_dict_configs),
            len(bwd_dkdv_dict_configs),
        )

        original_kernel_options = kernel_options.copy()

        def make_bwd_base_kernel_options(cfg: dict) -> dict:
            cur_kernel_options = original_kernel_options.copy()

            # Generated tiling config is authoritative for this choice.
            cur_kernel_options.update(cfg)

            for key in npu_config.FLEX_ATTENTION_NPU_COMPILE_HINT_KEYS:
                cur_kernel_options.pop(key, None)

            # Blocksparse options
            cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
            cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
            cur_kernel_options.setdefault(
                "TORCHINDUCTOR_FLEXATTENTION_MASKOUT",
                flexattention_mask_out,
            )
            cur_kernel_options.setdefault(
                "NUM_SPARSE_Q_BLOCKS",
                V.graph.sizevars.guard_int(kv_num_blocks.get_size()[2]),
            )
            return cur_kernel_options

        def make_bwd_dq_kernel_options(cfg: dict) -> dict:
            opts = make_bwd_base_kernel_options(cfg)
            opts.update(
                {
                    "TORCHINDUCTOR_FLEXATTENTION_MASKOUT": (
                        flexattention_mask_out
                    ),
                    "num_stages": 1,
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
            if kind == "dq":
                bwd_grid_x = (
                    bwd_num_queries_hint + cfg["BLOCK_M2"] - 1
                ) // cfg["BLOCK_M2"]
            else:
                bwd_grid_x = (
                    bwd_num_key_value_hint + cfg["BLOCK_N1"] - 1
                ) // cfg["BLOCK_N1"]
            bwd_grid_y = 1
            bwd_grid_z = bwd_batch_size_hint * bwd_kv_heads_hint
            bwd_total_programs = bwd_grid_x * bwd_grid_y * bwd_grid_z
            bwd_wave_estimate = (
                bwd_total_programs + bwd_num_cube_core - 1
            ) // bwd_num_cube_core

            log.debug(
                "bwd-%s-choice: cfg=%s grid=(%d,%d,%d) total_programs=%d aicore_count=%d exceeds_aicores=%s"
                "wave_estimate=%d kv_num_blocks_type=%s kv_indices_type=%s grad_lse_type=%s final_kernel_options=%s",
                kind, cfg, bwd_grid_x, bwd_grid_y, bwd_grid_z,
                bwd_total_programs, bwd_num_cube_core, bwd_total_programs
                > bwd_num_cube_core, bwd_wave_estimate,
                type(kv_num_blocks).__name__,
                type(kv_indices).__name__,
                type(grad_lse).__name__, cur_kernel_options)

        has_captured_grad_side_effect = bool(joint_outputs.mutated_grads)
        captured_grad_owner = "dkdv" if has_captured_grad_side_effect else None
        assert captured_grad_owner in (None, "dq", "dkdv")
        log.debug(
            "bwd split captured_grad_owner=%s mutated_grads=%d",
            captured_grad_owner,
            len(joint_outputs.mutated_grads),
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
            *mask_out_input_nodes,
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
            *mask_out_input_nodes,
            kv_num_blocks,
            kv_indices,
            q_num_blocks,
            q_indices,
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ]

        def make_dkdv_composite_choice_options(
            dkdv_kernel_options,
            dkdv_subgraphs,
        ):
            if (
                not flexattention_mask_out
                or not npu_config.flex_attention.bwd_dkdv_tasklist
                or bwd_has_dynamic_shape
            ):
                return {}

            try:
                compatible = is_dkdv_tasklist_codegen_compatible(
                    cpp_wrapper=V.graph.cpp_wrapper,
                    aot_mode=getattr(V.graph, "aot_mode", False),
                    bq=bwd_batch_size_hint,
                    bkv=bwd_kv_batch_size_hint,
                    sparse_z=sparse_z_val,
                    sparse_hq=sparse_mask_hq_val,
                    sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
                    block_n1=dkdv_kernel_options["BLOCK_N1"],
                    q_num_blocks_dtype=q_num_blocks.get_dtype(),
                    full_q_num_blocks_dtype=full_q_num_blocks.get_dtype(),
                    q_num_blocks_contiguous=(
                        q_num_blocks.get_layout().is_contiguous()
                    ),
                    full_q_num_blocks_contiguous=(
                        full_q_num_blocks.get_layout().is_contiguous()
                    ),
                    accum_dtype=broadcasted_grad_key_accum.get_dtype(),
                )
                if not compatible:
                    return {}

                partial_dk_stride = V.graph.sizevars.guard_int(
                    layout_broadcasted_k_accum.storage_size()
                )
                partial_dv_stride = V.graph.sizevars.guard_int(
                    broadcasted_grad_value.get_layout().storage_size()
                )
            except (AssertionError, NotImplementedError, TypeError, ValueError):
                log.info(
                    "dK/dV task-list codegen disabled for non-static metadata",
                    exc_info=True,
                )
                return {}

            block_n1 = dkdv_kernel_options["BLOCK_N1"]
            call_sizes = query.get_size() + key.get_size()[1:3]
            common_meta = dict(dkdv_kernel_options)
            num_stages = common_meta.pop("num_stages")
            num_warps = common_meta.pop("num_warps")
            common_meta.update(
                {
                    "PARTIAL_DK_STRIDE": partial_dk_stride,
                    "PARTIAL_DV_STRIDE": partial_dv_stride,
                    "TASKLIST_NO_SPLIT": False,
                }
            )

            tasklist_renderer_factory = (
                flex_attention_bwd_dkdv_tasklist.make_runtime_renderer_factory(
                    input_nodes=dkdv_input_nodes,
                    runtime_args=(
                        RuntimeTemplateArg(
                            "WORK_ITEMS", torch.int32, 2, "work_items_t"
                        ),
                        RuntimeTemplateArg(
                            "TASK_OFFSETS", torch.int32, 1, "task_offsets_t"
                        ),
                        RuntimeTemplateArg(
                            "DK_PARTIAL", torch.float32, 5, "dk_partial"
                        ),
                        RuntimeTemplateArg(
                            "DV_PARTIAL", torch.float32, 5, "dv_partial"
                        ),
                    ),
                    layout=layout_broadcasted_k_accum,
                    num_stages=num_stages,
                    num_warps=num_warps,
                    call_sizes=call_sizes,
                    subgraphs=dkdv_subgraphs,
                    reset_to_zero_arg_names=["arg_DV", "arg_DK"],
                    **common_meta,
                )
            )
            no_split_meta = dict(common_meta)
            no_split_meta["TASKLIST_NO_SPLIT"] = True
            tasklist_no_split_renderer_factory = (
                flex_attention_bwd_dkdv_tasklist_no_split.make_runtime_renderer_factory(
                    input_nodes=dkdv_input_nodes,
                    runtime_args=(
                        RuntimeTemplateArg(
                            "WORK_ITEMS", torch.int32, 2, "work_items_t"
                        ),
                        RuntimeTemplateArg(
                            "TASK_OFFSETS", torch.int32, 1, "task_offsets_t"
                        ),
                        RuntimeTemplateArg(
                            "DK_PARTIAL", torch.float32, 5, "dk_partial"
                        ),
                        RuntimeTemplateArg(
                            "DV_PARTIAL", torch.float32, 5, "dv_partial"
                        ),
                    ),
                    layout=layout_broadcasted_k_accum,
                    num_stages=num_stages,
                    num_warps=num_warps,
                    call_sizes=call_sizes,
                    subgraphs=dkdv_subgraphs,
                    reset_to_zero_arg_names=["arg_DV", "arg_DK"],
                    **no_split_meta,
                )
            )
            reduce_renderer_factory = (
                flex_attention_bwd_dkdv_reduce.make_runtime_renderer_factory(
                    input_nodes=[
                        broadcasted_grad_key_accum,
                        broadcasted_grad_value,
                    ],
                    runtime_args=(
                        RuntimeTemplateArg(
                            "DK_PARTIAL", torch.float32, 5, "dk_partial"
                        ),
                        RuntimeTemplateArg(
                            "DV_PARTIAL", torch.float32, 5, "dv_partial"
                        ),
                        RuntimeTemplateArg(
                            "SPLIT_BASES", torch.int32, 2, "split_bases_t"
                        ),
                    ),
                    layout=layout_broadcasted_k_accum,
                    num_stages=1,
                    num_warps=num_warps,
                    call_sizes=call_sizes,
                    **common_meta,
                )
            )

            def runtime_renderer_factory(out_node):
                return {
                    "tasklist": tasklist_renderer_factory(out_node),
                    "tasklist_no_split": tasklist_no_split_renderer_factory(
                        out_node
                    ),
                    "reduce": reduce_renderer_factory(out_node),
                }

            dispatch_spec = FlexAttentionDkdvDispatchSpec(
                launch_programs=dkdv_kernel_options["LAUNCH_PROGRAMS"],
                batch_size=bwd_batch_size_hint,
                num_kv_heads=bwd_kv_heads_hint,
                num_kv_blocks=(
                    bwd_num_key_value_hint + block_n1 - 1
                )
                // block_n1,
                sparse_kv_multiple=SPARSE_KV_BLOCK_SIZE // block_n1,
                sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
                block_n1=block_n1,
                partial_dk_stride=partial_dk_stride,
                partial_dv_stride=partial_dv_stride,
            )
            return {
                "runtime_renderer_factory": runtime_renderer_factory,
                "dispatch_spec": dispatch_spec,
            }

        for cfg in bwd_dq_dict_configs:
            dq_kernel_options = make_bwd_dq_kernel_options(cfg)
            dq_subgraphs, dq_mutated_inputs, dq_run_captured = (
                make_bwd_subgraphs_and_mutations("dq", [grad_query])
            )
            dq_kernel_options["RUN_CAPTURED_GRADS"] = dq_run_captured
            log_bwd_choice("dq", cfg, dq_kernel_options)

            prev_dq_choice_count = len(dq_choices)
            flex_attention_bwd_dq_mask_out.maybe_append_choice(
                choices=dq_choices,
                input_nodes=dq_input_nodes,
                layout=grad_query.get_layout(),
                subgraphs=dq_subgraphs,
                mutated_inputs=dq_mutated_inputs,
                reset_to_zero_arg_names=None,
                large_input_buffers=mask_out_input_nodes,
                call_sizes=query.get_size() + key.get_size()[1:3],
                **dq_kernel_options,
            )
            if len(dq_choices) > prev_dq_choice_count:
                _tag_flex_attention_report_choices(
                    dq_choices[prev_dq_choice_count:],
                    cfg,
                )
                if prefer_max_tiling_without_benchmark():
                    _tag_choice_attr(
                        dq_choices[prev_dq_choice_count:],
                        "_nobench_select_first_compilable",
                        True,
                    )

        for cfg in bwd_dkdv_dict_configs:
            dkdv_kernel_options = make_bwd_dkdv_kernel_options(cfg)
            dkdv_subgraphs, dkdv_mutated_inputs, dkdv_run_captured = (
                make_bwd_subgraphs_and_mutations(
                    "dkdv",
                    [broadcasted_grad_value, broadcasted_grad_key_accum],
                )
            )
            dkdv_kernel_options["RUN_CAPTURED_GRADS"] = dkdv_run_captured
            log_bwd_choice("dkdv", cfg, dkdv_kernel_options)
            dkdv_composite_choice_options = make_dkdv_composite_choice_options(
                dkdv_kernel_options,
                dkdv_subgraphs,
            )

            prev_dkdv_choice_count = len(dkdv_choices)
            flex_attention_bwd_dkdv_mask_out.maybe_append_choice(
                choices=dkdv_choices,
                input_nodes=dkdv_input_nodes,
                layout=layout_broadcasted_k_accum,
                subgraphs=dkdv_subgraphs,
                mutated_inputs=dkdv_mutated_inputs,
                reset_to_zero_arg_names=["arg_DV", "arg_DK"],
                large_input_buffers=mask_out_input_nodes,
                call_sizes=query.get_size() + key.get_size()[1:3],
                **dkdv_composite_choice_options,
                **dkdv_kernel_options,
            )
            if len(dkdv_choices) > prev_dkdv_choice_count:
                _tag_flex_attention_report_choices(
                    dkdv_choices[prev_dkdv_choice_count:],
                    cfg,
                )
                _tag_choices_for_no_benchmark(
                    dkdv_choices[prev_dkdv_choice_count:]
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
        dq_block_metadata_input_idx = 10 if flexattention_mask_out else 7
        dq_input_gen_fns = {
            dq_block_metadata_input_idx: create_num_blocks_fake_generator(kv_indices),
            dq_block_metadata_input_idx + 1: create_indices_fake,
            dq_block_metadata_input_idx + 2: create_num_blocks_fake_generator(q_indices),
            dq_block_metadata_input_idx + 3: create_indices_fake,
            dq_block_metadata_input_idx + 4: create_num_blocks_fake_generator(full_kv_indices),
            dq_block_metadata_input_idx + 5: create_indices_fake,
            dq_block_metadata_input_idx + 6: create_num_blocks_fake_generator(full_q_indices),
            dq_block_metadata_input_idx + 7: create_indices_fake,
        }
        if flexattention_mask_out:
            dq_input_gen_fns.update(
                {
                    8: create_compact_q_offsets_fake,
                    9: create_zero_int_tensor_fake,
                }
            )

        dkdv_inputs_for_autotuning = (
            dkdv_input_nodes
            + list(score_mod_other_buffers)
            + (
                list(joint_outputs.mutated_grads)
                if captured_grad_owner == "dkdv"
                else []
            )
        )
        dkdv_block_metadata_input_idx = 11 if flexattention_mask_out else 8
        dkdv_input_gen_fns = {
            dkdv_block_metadata_input_idx: create_num_blocks_fake_generator(kv_indices),
            dkdv_block_metadata_input_idx + 1: create_indices_fake,
            dkdv_block_metadata_input_idx + 2: create_num_blocks_fake_generator(q_indices),
            dkdv_block_metadata_input_idx + 3: create_indices_fake,
            dkdv_block_metadata_input_idx + 4: create_num_blocks_fake_generator(full_kv_indices),
            dkdv_block_metadata_input_idx + 5: create_indices_fake,
            dkdv_block_metadata_input_idx + 6: create_num_blocks_fake_generator(full_q_indices),
            dkdv_block_metadata_input_idx + 7: create_indices_fake,
        }
        if flexattention_mask_out:
            dkdv_input_gen_fns.update(
                {
                    9: create_compact_q_offsets_fake,
                    10: create_zero_int_tensor_fake,
                }
            )

        log.info(
            "bwd-summary: dq_choices=%d dkdv_choices=%d query_call_size_hints=%s key_call_size_hints=%s aicore_count=%d",
            len(dq_choices),
            len(dkdv_choices),
            tuple(bwd_query_call_size_hints), tuple(bwd_key_call_size_hints), bwd_num_cube_core
        )

        autotune_select_algorithm(
            "flex_attention_backward_dkdv_only",
            dkdv_choices,
            dkdv_inputs_for_autotuning,
            layout_broadcasted_k_accum,
            input_gen_fns=dkdv_input_gen_fns,
        )

        autotune_select_algorithm(
            "flex_attention_backward_qmajor_dq",
            dq_choices,
            dq_inputs_for_autotuning,
            grad_query.get_layout(),
            input_gen_fns=dq_input_gen_fns,
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
        grad_key = _maybe_copy_to_dtype(grad_key_accum, key.get_dtype())
        grad_value = _maybe_copy_to_dtype(
            grad_value_accum, value.get_dtype()
        )
        grad_key = _force_fixed_layout(grad_key, key_strides)
        grad_value = _force_fixed_layout(grad_value, value_strides)
        grad_query = _force_fixed_layout(
            _maybe_copy_to_dtype(grad_query, query.get_dtype()),
            grad_query_strides,
        )

        return (grad_query, grad_key, grad_value, tuple(joint_outputs.captured_grads))
