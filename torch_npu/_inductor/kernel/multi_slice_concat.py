"""Indexed slice loads and a static multi-source slice-cat template."""

__all__ = ["_register_npu_inductor_multi_slice_concat"]

import functools
import hashlib
import logging

import sympy
import torch

from torch._inductor import ir
from torch._inductor.ir import FixedLayout
from torch._inductor.lowering import empty_strided, fallback_handler, register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
)
from torch._inductor.utils import triton_type
from torch._inductor.virtualized import V

from ..select_algorithm import NPUTritonTemplate


log = logging.getLogger(__name__)


@SymbolicGridFn
def _indexed_slice_concat_grid(rows, cols, meta, *, cdiv):
    return (cdiv(rows * cols, meta["BLOCK"]), 1, 1)


@functools.lru_cache(maxsize=1)
def _indexed_template():
    return NPUTritonTemplate(
        name="npu_indexed_slice_concat",
        grid=_indexed_slice_concat_grid,
        source=r'''
{{def_kernel("SRC", "INDEX")}}
    rows = tl.full([], {{size("SRC", 0)}}, INDEX_DTYPE)
    total = {{size("INDEX", 0)}}
    source_cols = {{size("SRC", 1)}}
    stride_row = tl.full([], {{stride("SRC", 0)}}, INDEX_DTYPE)
    stride_col = tl.full([], {{stride("SRC", 1)}}, INDEX_DTYPE)
    linear = (tl.program_id(0).to(INDEX_DTYPE) * BLOCK
              + tl.arange(0, BLOCK).to(INDEX_DTYPE))
    row = linear // total
    col = linear % total
    mask = linear < rows * total
    source_col = tl.load(INDEX + col, mask=mask, other=0)
    # Fusion benchmarks can replace the constant index table with random data.
    valid = (source_col >= 0) & (source_col < source_cols)
    source_col = tl.where(valid, source_col, 0).to(INDEX_DTYPE)
    value = tl.load(SRC + row * stride_row + source_col * stride_col,
                    mask=mask & valid, other=0).to(ACC_TYPE)
    {{store_output(("row", "col"), "value", "mask", val_shape=("BLOCK",))}}
''',
    )


def _indexed_block_size(numel, source_dtype, output_dtype):
    from .. import config as npu_config

    cores = max(1, npu_config.num_vector_core)
    source_bytes, output_bytes = source_dtype.itemsize, output_dtype.itemsize
    min_block = _next_pow2((_ALIGN_BYTES + min(source_bytes, output_bytes) - 1)
                           // min(source_bytes, output_bytes))
    # Index, row, column and address; mask, source, conversion and output values.
    live_bytes = 4 * 8 + 1 + source_bytes + max(4, source_bytes, output_bytes) + output_bytes
    block_cap = _prev_pow2(max(min_block, _ub_bytes() // live_bytes))
    hint = max(1, V.graph.sizevars.size_hint(numel, fallback=cores * min_block))
    return min(block_cap, max(min_block, _next_pow2((hint + cores - 1) // cores)))


def _register_indexed_slice_concat(target):
    @register_lowering(target, type_promotion_kind=None)
    def indexed_slice_concat(src, offsets, widths, dtype):
        src = ir.ExternKernel.realize_input(src)
        rows, total = src.get_size()[0], sympy.Integer(sum(widths))
        layout = FixedLayout(src.get_device(), dtype, [rows, total],
                             stride=[total, sympy.Integer(1)])
        if V.graph.sizevars.statically_known_equals(rows, 0):
            return empty_strided(layout.size, layout.stride, dtype=dtype, device=src.get_device())
        columns = [column for offset, width in zip(offsets, widths)
                   for column in range(offset, offset + width)]
        # A compiler-owned constant, built once rather than on every invocation.
        with torch.utils._python_dispatch._disable_current_modes():
            value = torch.tensor(columns, dtype=torch.int64, device=src.get_device())
        index = ir.ExternKernel.realize_input(V.graph.add_tensor_constant(value))
        choices = []
        _indexed_template().maybe_append_choice(
            choices=choices, input_nodes=[src, index], layout=layout,
            BLOCK=_indexed_block_size(rows * total, src.get_dtype(), dtype),
            ACC_TYPE=triton_type(dtype), num_stages=2, num_warps=4,
        )
        if not choices:
            return fallback_handler(target, add_to_fallback_set=False)(
                src, list(offsets), list(widths), dtype,
            )
        # One static configuration: no lowering-time search or benchmark.
        out = choices[0].output_node()
        # Keep dtype conversion materialized so consumers cannot remove its rounding.
        if src.get_dtype() != dtype:
            V.graph.no_fuse_buffer_names.add(out.get_name())
        return out


@SymbolicGridFn
def _multi_slice_concat_grid(rows, cols, meta, *, cdiv):
    return (cdiv(rows, meta["BLOCK_ROWS"]), 1, 1)


_ALIGN_BYTES = 32


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pad_width(width, esize):
    """Use a power-of-two tile of at least 32 bytes; mask the padded columns."""
    return max(_next_pow2(width), _next_pow2(max(1, _ALIGN_BYTES // esize)))


def _out_index(cat_dim, idx, prefix, width, total):
    """Output row stride and segment base offset, as source text."""
    if cat_dim == 1:
        return str(total), str(prefix)
    # seg_rows is ROWS in INDEX_DTYPE; ROWS itself may be an int32 argument.
    return str(width), "0" if idx == 0 else f"seg_rows * {idx * width}"


def _build_source(n_srcs, n_masks, src_idx, offsets, widths, mask_idx, esize, cat_dim):
    total = sum(widths)
    src_names = [f"SRC{i}" for i in range(n_srcs)]
    mask_names = [f"MASK{i}" for i in range(n_masks)]

    lines = ['{{def_kernel(%s)}}' % ", ".join(f'"{n}"' for n in src_names + mask_names)]
    # SRC*/MASK* aliases include view offsets; arg_* pointers do not.
    lines.append("    OUT = arg_OUT")
    lines.append('    ROWS = {{size("%s", 0)}}' % src_names[0])
    for i, name in enumerate(src_names):
        lines.append('    stride_src%d_r = {{stride("%s", 0)}}' % (i, name))
        lines.append('    stride_src%d_c = {{stride("%s", 1)}}' % (i, name))
    for i, name in enumerate(mask_names):
        lines.append('    stride_mask%d = {{stride("%s", 0)}}' % (i, name))
    lines.append("")
    # Every index term carries INDEX_DTYPE: rows * stride overflows int32 long before
    # the buffer needs 64-bit addressing.
    lines.append("    pid = tl.program_id(0).to(INDEX_DTYPE)")
    lines.append("    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(INDEX_DTYPE)")
    lines.append("    row_mask = rows < ROWS")
    if cat_dim == 0 and len(widths) > 1:
        lines.append("    seg_rows = tl.full([], ROWS, INDEX_DTYPE)")

    for i in range(n_masks):
        lines.append("")
        lines.append(
            f"    m{i} = tl.load("
            f"MASK{i} + rows[:, None] * stride_mask{i}, "
            "mask=row_mask[:, None], other=0)"
        )

    prefix = 0
    for idx, (si, off, width, mi) in enumerate(
        zip(src_idx, offsets, widths, mask_idx)
    ):
        pad = _pad_width(width, esize)
        access = (
            "row_mask[:, None]"
            if pad == width
            else f"row_mask[:, None] & (cols{idx}[None, :] < {width})"
        )
        row_stride, out_base = _out_index(cat_dim, idx, prefix, width, total)
        dest = (
            f"out[:, {prefix}:{prefix + width}]"
            if cat_dim == 1
            else f"out rows [{idx} * ROWS, {idx + 1} * ROWS)"
        )
        lines.append("")
        lines.append(
            f"    # segment {idx}: SRC{si}[:, {off}:{off + width}]"
            f"{'' if mi < 0 else f' masked by MASK{mi}'}"
            f" -> {dest}"
        )
        lines.append(f"    cols{idx} = tl.arange(0, {pad}).to(INDEX_DTYPE)")
        lines.append(f"    val{idx} = tl.load(")
        lines.append(
            f"        SRC{si} + rows[:, None] * stride_src{si}_r "
            f"+ ({off} + cols{idx}[None, :]) * stride_src{si}_c,"
        )
        lines.append(f"        mask={access},")
        lines.append("        other=0,")
        lines.append("    )")
        if mi >= 0:
            lines.append(f"    val{idx} = tl.where(m{mi}, 0, val{idx})")
        lines.append("    tl.store(")
        lines.append(
            f"        OUT + rows[:, None] * {row_stride} "
            f"+ ({out_base} + cols{idx}[None, :]),"
        )
        lines.append(f"        val{idx},")
        lines.append(f"        mask={access},")
        lines.append("    )")
        prefix += width

    return "\n".join(lines) + "\n"


@functools.lru_cache(maxsize=None)
def _template_for(n_srcs, n_masks, src_idx, offsets, widths, mask_idx, esize, cat_dim):
    plan = (n_srcs, n_masks, src_idx, offsets, widths, mask_idx, esize, cat_dim)
    # Template names are global and must not collide; hash() is not stable across processes.
    digest = hashlib.sha256(repr(plan).encode()).hexdigest()[:16]
    return NPUTritonTemplate(
        name=f"npu_multi_slice_concat_{digest}",
        grid=_multi_slice_concat_grid,
        source=_build_source(n_srcs, n_masks, src_idx, offsets, widths, mask_idx,
                             esize, cat_dim),
        manual_output_buffer="arg_OUT",
    )


# Reserve UB for up to eight simultaneously live segment tiles.
_ASSUMED_LIVE_TILES = 8

# The smallest UB among known targets, used when the backend cannot tell us.
_FALLBACK_UB_BYTES = 192 * 1024

_MAX_BLOCK_ROWS = 256


@functools.lru_cache(maxsize=None)
def _ub_bytes():
    """Usable UB bytes, from the backend's budget rather than the raw capacity."""
    try:
        from triton.backends.ascend.utils import graph_ub_budget_bytes_for_arch
        from triton.runtime import driver

        arch = getattr(driver.active.get_current_target(), "arch", "")
        budget = graph_ub_budget_bytes_for_arch(arch) or 0
        if budget > 0:
            return budget
    except Exception:  # no device, or a backend without the resolver
        pass
    log.warning("multi_slice_concat: could not read the UB budget, assuming %d bytes",
                _FALLBACK_UB_BYTES)
    return _FALLBACK_UB_BYTES


def _max_tile_elems(esize, n_segments=_ASSUMED_LIVE_TILES):
    """Element budget for one segment tile; a run cannot hold more tiles than segments."""
    live = min(max(1, n_segments), _ASSUMED_LIVE_TILES)
    return max(1, _ub_bytes() // live // esize)


def _get_configs(max_width, esize, n_segments, rows=None):
    """Keep UB-bounded tiles; distribute narrow column segments across cores."""
    from .. import config as npu_config

    row_cap = _max_tile_elems(esize, n_segments) // _pad_width(max_width, esize)
    block_rows = _MAX_BLOCK_ROWS
    if isinstance(rows, (int, sympy.Integer)) and rows > 0:
        tile_rows = int(rows)
        # Wide segments benefit from larger transfers; keep their original tile.
        if max_width * esize <= _ALIGN_BYTES:
            cores = max(1, npu_config.num_vector_core)
            tile_rows = (tile_rows + cores - 1) // cores
        block_rows = min(block_rows, _next_pow2(tile_rows))
    block_rows = min(block_rows, _prev_pow2(max(1, row_cap)))
    return [
        {
            "BLOCK_ROWS": max(1, block_rows),
            "num_stages": 2,
            "num_warps": 4,
        }
    ]


def _dedup_inputs(nodes, indices):
    """Deduplicate identical layouts; reject differing views of the same buffer.

    def_kernel keys arguments by buffer name, so it cannot distinguish those views.
    """
    unique, remap, seen = [], [], {}
    for node in nodes:
        layout = node.get_layout()
        key = (
            node.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            getattr(layout, "offset", 0),
        )
        if key not in seen:
            seen[key] = len(unique)
            unique.append(node)
        remap.append(seen[key])

    names = [node.get_name() for node in unique]
    if len(set(names)) != len(names):
        return None, None
    return unique, [remap[i] if i >= 0 else i for i in indices]


def _register_npu_inductor_multi_slice_concat():
    from ..fx_passes.ascend_custom_passes.ascend_graph_pass import (
        INDEXED_SLICE_CONCAT_TARGET,
        MULTI_SLICE_CONCAT_DTYPES,
        MULTI_SLICE_CONCAT_TARGET,
    )

    _register_indexed_slice_concat(INDEXED_SLICE_CONCAT_TARGET)

    @register_lowering(MULTI_SLICE_CONCAT_TARGET, type_promotion_kind=None)
    def multi_slice_concat(srcs, masks, src_idx, offsets, widths, mask_idx, cat_dim=1):
        srcs = [ir.ExternKernel.realize_input(src) for src in srcs]
        masks = [ir.ExternKernel.realize_input(mask) for mask in masks]

        def per_slice_copies(srcs, masks, src_idx, mask_idx):
            return fallback_handler(MULTI_SLICE_CONCAT_TARGET, add_to_fallback_set=False)(
                srcs, masks, list(src_idx), list(offsets), list(widths), list(mask_idx),
                cat_dim,
            )

        uniq_srcs, uniq_src_idx = _dedup_inputs(srcs, src_idx)
        uniq_masks, uniq_mask_idx = _dedup_inputs(masks, mask_idx)
        if uniq_srcs is None or uniq_masks is None:
            log.warning(
                "multi_slice_concat: inputs contain same-named views with different "
                "layouts, falling back to per-slice copies"
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)
        srcs, src_idx = uniq_srcs, uniq_src_idx
        masks, mask_idx = uniq_masks, uniq_mask_idx

        src_idx, offsets = tuple(src_idx), tuple(offsets)
        widths, mask_idx = tuple(widths), tuple(mask_idx)

        if cat_dim not in (0, 1):
            log.warning(
                "multi_slice_concat: cat_dim=%s is not 0 or 1, falling back", cat_dim
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)

        # The op is public, so these are rechecked for direct callers.
        dtypes = {src.get_dtype() for src in srcs}
        if len(dtypes) > 1 or not dtypes <= MULTI_SLICE_CONCAT_DTYPES:
            log.warning(
                "multi_slice_concat: source dtypes %s are mixed or unsupported by the "
                "template, falling back",
                sorted(str(d) for d in dtypes),
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)
        if any(mask.get_dtype() is not torch.bool for mask in masks):
            log.warning(
                "multi_slice_concat: row masks must be bool, falling back"
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)
        if cat_dim == 0 and len(set(widths)) > 1:
            log.warning(
                "multi_slice_concat: cat_dim=0 needs one width for all segments, "
                "got %d distinct, falling back",
                len(set(widths)),
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)

        total = sum(widths)
        rows = srcs[0].get_size()[0]
        out_size = (
            [rows, sympy.Integer(total)]
            if cat_dim == 1
            else [rows * len(widths), sympy.Integer(widths[0])]
        )
        out_cols = total if cat_dim == 1 else widths[0]
        layout = FixedLayout(
            srcs[0].get_device(),
            srcs[0].get_dtype(),
            out_size,
            stride=[sympy.Integer(out_cols), sympy.Integer(1)],
        )

        # Must match the def_kernel parameter order: sources, then masks.
        input_nodes = [*srcs, *masks]
        esize = srcs[0].get_dtype().itemsize

        # BLOCK_ROWS cannot go below 1, so a single row over budget cannot fit in UB.
        widest_tile = max(_pad_width(width, esize) for width in widths)
        tile_budget = _max_tile_elems(esize, len(widths))
        if widest_tile > tile_budget:
            log.warning(
                "multi_slice_concat: widest segment tile is %d elements, over the %d "
                "the UB budget allows for %d segments of this dtype, falling back to "
                "per-slice copies",
                widest_tile,
                tile_budget,
                len(widths),
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)

        template = _template_for(
            len(srcs), len(masks), src_idx, offsets, widths, mask_idx, esize, cat_dim,
        )
        choices = []
        # The grid covers the segment rows, not the output rows of a row concat.
        call_sizes = [rows, sympy.Integer(out_cols)]
        static_rows = int(rows) if isinstance(rows, (int, sympy.Integer)) else None
        for cfg in _get_configs(max(widths), esize, len(widths), static_rows):
            template.maybe_append_choice(
                choices=choices,
                input_nodes=input_nodes,
                layout=layout,
                call_sizes=call_sizes,
                **cfg,
            )

        if not choices:
            log.warning(
                "multi_slice_concat: no template config compiled for %d segments, "
                "falling back to per-slice copies",
                len(widths),
            )
            return per_slice_copies(srcs, masks, src_idx, mask_idx)

        out = autotune_select_algorithm(
            "multi_slice_concat", choices, input_nodes, layout
        )
        # No store_output hook: a fused epilogue would be silently dropped by codegen.
        V.graph.no_fuse_buffer_names.add(out.get_name())
        return out
