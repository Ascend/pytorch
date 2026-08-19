"""Triton template and lowering for ``npu_ext::multi_slice_concat``.

A run of fixed-width column slices over several wide tables, lowered to a single
kernel: blocked by rows, each program handling all segments for ``BLOCK_ROWS`` rows,
with segments unrolled statically at render time. Source indices, offsets and widths
are compile-time constants, so addressing is affine fixed-width copies, not a gather.

Masked segments add a ``tl.where``. The mask is a ``[rows, 1]`` boolean loaded once up
front and reused across segments.

Output goes through ``manual_output_buffer``: the full output can be thousands of
columns wide, too much to stage on chip for a single ``store_output``, so the template
stores per segment instead. That gives up epilogue fusion: with no ``store_output`` there
is no hook to render an epilogue into, and one fused in anyway would be skipped by codegen
rather than emitted, silently dropping the consumer's computation. The lowering therefore
marks the output in ``V.graph.no_fuse_buffer_names``, which the scheduler honours before
it asks the backend, so the concat is fused into nothing at all.

The segment plan is baked into the template source and the instance cached by plan, so
autotune kwargs stay down to ``BLOCK_ROWS`` and the kernel name does not grow with the
offset count.
"""

__all__ = ["_register_npu_inductor_multi_slice_concat"]

import functools
import hashlib
import logging

import sympy

from torch._inductor import ir
from torch._inductor.ir import FixedLayout
from torch._inductor.lowering import fallback_handler, register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
)
from torch._inductor.virtualized import V

from ..select_algorithm import NPUTritonTemplate


log = logging.getLogger(__name__)


@SymbolicGridFn
def _multi_slice_concat_grid(rows, cols, meta, *, cdiv):
    """Program count depends only on the row count, not the segment count."""
    return (cdiv(rows, meta["BLOCK_ROWS"]), 1, 1)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _build_source(n_srcs, n_masks, src_idx, offsets, widths, mask_idx):
    """Render the kernel source; all indices and widths are compile-time constants."""
    total = sum(widths)
    src_names = [f"SRC{i}" for i in range(n_srcs)]
    mask_names = [f"MASK{i}" for i in range(n_masks)]

    lines = ['{{def_kernel(%s)}}' % ", ".join(f'"{n}"' for n in src_names + mask_names)]
    lines.append("    OUT = arg_OUT")
    lines.append('    ROWS = {{size("%s", 0)}}' % src_names[0])
    for i, name in enumerate(src_names):
        lines.append('    stride_src%d_r = {{stride("%s", 0)}}' % (i, name))
        lines.append('    stride_src%d_c = {{stride("%s", 1)}}' % (i, name))
    # Only column 0 of a mask is read, so the column stride is unused. A broadcast mask
    # has row stride 0, making every row read the same element, which is correct.
    for i, name in enumerate(mask_names):
        lines.append('    stride_mask%d = {{stride("%s", 0)}}' % (i, name))
    lines.append("")
    lines.append("    pid = tl.program_id(0).to(tl.int32)")
    lines.append("    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)")
    lines.append("    row_mask = rows < ROWS")

    # Load masks once up front so segments sharing one do not reload it.
    for i in range(n_masks):
        lines.append("")
        lines.append(
            f"    m{i} = tl.load("
            f"arg_MASK{i} + rows[:, None] * stride_mask{i}, "
            "mask=row_mask[:, None], other=0)"
        )

    prefix = 0
    for idx, (si, off, width, mi) in enumerate(
        zip(src_idx, offsets, widths, mask_idx)
    ):
        pad = _next_pow2(width)
        # tl.arange needs a power of two, so over-read and trim with a column mask.
        access = (
            "row_mask[:, None]"
            if pad == width
            else f"row_mask[:, None] & (cols{idx}[None, :] < {width})"
        )
        lines.append("")
        lines.append(
            f"    # segment {idx}: SRC{si}[:, {off}:{off + width}]"
            f"{'' if mi < 0 else f' masked by MASK{mi}'}"
            f" -> out[:, {prefix}:{prefix + width}]"
        )
        lines.append(f"    cols{idx} = tl.arange(0, {pad})")
        lines.append(f"    val{idx} = tl.load(")
        lines.append(
            f"        arg_SRC{si} + rows[:, None] * stride_src{si}_r "
            f"+ ({off} + cols{idx}[None, :]) * stride_src{si}_c,"
        )
        lines.append(f"        mask={access},")
        lines.append("        other=0,")
        lines.append("    )")
        if mi >= 0:
            # Zero the whole row where the mask is true, matching where(mask, 0, x).
            lines.append(f"    val{idx} = tl.where(m{mi}, 0, val{idx})")
        lines.append("    tl.store(")
        lines.append(
            f"        OUT + rows[:, None] * {total} + ({prefix} + cols{idx}[None, :]),"
        )
        lines.append(f"        val{idx},")
        lines.append(f"        mask={access},")
        lines.append("    )")
        prefix += width

    return "\n".join(lines) + "\n"


@functools.lru_cache(maxsize=None)
def _template_for(n_srcs, n_masks, src_idx, offsets, widths, mask_idx):
    """Cache the template by plan so a repeated plan reuses one instance."""
    plan = (n_srcs, n_masks, src_idx, offsets, widths, mask_idx)
    # TritonTemplate keys globally on name and raises on a collision, so the name has to
    # track the plan. A content digest rather than hash(), which is only stable in-process.
    digest = hashlib.sha256(repr(plan).encode()).hexdigest()[:16]
    return NPUTritonTemplate(
        name=f"npu_multi_slice_concat_{digest}",
        grid=_multi_slice_concat_grid,
        source=_build_source(n_srcs, n_masks, src_idx, offsets, widths, mask_idx),
        manual_output_buffer="arg_OUT",
    )


# Only one segment is resident at a time: BLOCK_ROWS x next_pow2(width) elements.
# Capping on element count keeps a wide segment plus a large BLOCK_ROWS on chip.
_MAX_TILE_ELEMS = 32768


def _get_configs(max_width):
    """Row-blocking configs. For a pure copy kernel only the rows per program matter;
    num_stages / num_warps reuse the mm template's range rather than widening the search."""
    row_cap = max(1, _MAX_TILE_ELEMS // _next_pow2(max_width))
    configs = []
    for block_rows in (8, 32, 64, 128, 256):
        if block_rows > row_cap:
            break
        for num_stages in (2, 3):
            for num_warps in (4, 8):
                configs.append(
                    {
                        "BLOCK_ROWS": block_rows,
                        "num_stages": num_stages,
                        "num_warps": num_warps,
                    }
                )
    # A very wide segment can push row_cap below 8; keep one config so choices is non-empty.
    if not configs:
        configs = [{"BLOCK_ROWS": max(1, row_cap), "num_stages": 2, "num_warps": 4}]
    return configs


def _dedup_inputs(nodes, indices):
    """Merge inputs backed by the same buffer and remap the plan indices onto them.

    ``def_kernel`` registers arguments in two differently keyed dicts:
    ``named_input_nodes`` by template parameter name and ``args.input_buffers`` by buffer
    name. One buffer appearing twice makes the latter overwrite itself, leaving the
    earlier ``arg_`` name undefined and the rendered kernel raising NameError, while the
    stride lines still work because they come from the other dict.

    This is common in real graphs: one row mask squeezed and reshaped at two sites is two
    fx nodes, which inductor then CSEs back into a single ComputedBuffer.

    Two views sharing a name but not a layout cannot both be expressed, since
    ``input_buffers`` still has one slot, so return None and let the caller fall back.
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
    # The op is defined on the fx_passes side; import it so torch.ops.npu_ext is populated.
    from ..fx_passes.ascend_custom_passes.ascend_graph_pass import (
        MULTI_SLICE_CONCAT_TARGET,
    )

    @register_lowering(MULTI_SLICE_CONCAT_TARGET, type_promotion_kind=None)
    def multi_slice_concat(srcs, masks, src_idx, offsets, widths, mask_idx):
        srcs = [ir.ExternKernel.realize_input(src) for src in srcs]
        masks = [ir.ExternKernel.realize_input(mask) for mask in masks]

        def per_slice_copies(srcs, masks, src_idx, mask_idx):
            """Fall back to the eager decomposition: still correct, one Slice per segment."""
            return fallback_handler(MULTI_SLICE_CONCAT_TARGET, add_to_fallback_set=False)(
                srcs, masks, list(src_idx), list(offsets), list(widths), list(mask_idx)
            )

        # Dedup before templating: the plan is both the cache key and the pointer count.
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

        total = sum(widths)
        rows = srcs[0].get_size()[0]
        layout = FixedLayout(
            srcs[0].get_device(),
            srcs[0].get_dtype(),
            [rows, sympy.Integer(total)],
            stride=[sympy.Integer(total), sympy.Integer(1)],
        )

        # def_kernel orders parameters as all sources then all masks; input_nodes must match.
        input_nodes = [*srcs, *masks]
        template = _template_for(
            len(srcs), len(masks), src_idx, offsets, widths, mask_idx
        )
        choices = []
        for cfg in _get_configs(max(widths)):
            template.maybe_append_choice(
                choices=choices, input_nodes=input_nodes, layout=layout, **cfg
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
        # Nothing may be fused into this kernel: it stores its own output, so it renders
        # no store_output hook, and an epilogue the scheduler fused in would be skipped by
        # codegen rather than emitted -- the kernel would return the bare concat with the
        # consumer's computation gone. ``Scheduler.can_fuse`` tests this set before it
        # consults the backend, so one name here refuses epilogue and horizontal fusion
        # alike.
        V.graph.no_fuse_buffer_names.add(out.get_name())
        return out
