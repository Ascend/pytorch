"""Experimental FlexAttention templates built on shared NPU sources."""

import ast
import inspect
import textwrap

from torch._inductor.kernel.flex.common import load_flex_template
from torch._inductor.select_algorithm import SymbolicGridFn

from torch_npu._inductor.kernel.flexattention_template import (
    _BWD_DKDV_COMPILE_OPTIONS,
    _BWD_DQ_COMPILE_OPTIONS,
    _FWD_COMPILE_OPTIONS,
    _FWD_MASK_OUT_SOURCE,
    compute_bwd_sparse_mask_kernel_compact,
    compute_compact_sparse_mask_mapping_kernel,
    compute_sparse_mask_block_pos_kernel,
    compute_sparse_mask_kernel_compact,
    compact_mapping_grid,
    compact_sparse_mask_grid,
    flex_attention_backward_dkdv_grid,
    flex_attention_backward_dq_grid,
    flex_attention_backward_dkdv_reduce_source,
    flex_attention_in_loop_grid,
    flex_attention_backward_qmajor_dq_source as _DEFAULT_BWD_DQ_SOURCE,
    flex_attention_backward_dkdv_only_source as _DEFAULT_BWD_DKDV_SOURCE,
    flex_attention_backward_dkdv_tasklist_source as _DEFAULT_BWD_TASKLIST_SOURCE,
    sparse_mask_block_pos_grid,
)
from torch_npu._inductor import select_algorithm as shared_selector
from torch_npu._inductor.select_algorithm import NPUTemplateCompileOption


class ExperimentalTemplateKernel(shared_selector.NPUTritonTemplateKernel):
    def jit_lines(self):
        lines = super().jit_lines()
        marker = "@triton_heuristics.template("
        if lines.count(marker) != 1:
            raise RuntimeError("default NPU template decorator changed")
        return lines.replace(marker, "@npu_triton_heuristics.template(")


def _experimental_def_kernel():
    function = shared_selector.NPUTritonTemplateKernel.def_kernel
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    namespace = dict(vars(shared_selector))
    namespace["_gen_npu_template_triton_imports"] = lambda: (
        shared_selector._gen_npu_template_triton_imports()
        + "\nfrom torch_npu._inductor.triton_experimental import npu_triton_heuristics\n"
    )
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


ExperimentalTemplateKernel.def_kernel = _experimental_def_kernel()


def _experimental_generate():
    tree = ast.parse(textwrap.dedent(inspect.getsource(shared_selector.NPUTritonTemplate.generate)))
    matches = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "make_kernel_render":
            continue
        if len(node.args.args) != 1 or node.args.args[0].arg != "out_node":
            raise RuntimeError("default NPU template renderer signature changed")
        node.args.args.append(ast.arg(arg="hint_override"))
        node.args.defaults.append(ast.Constant(value=None))
        node.args.kwarg = ast.arg(arg="_kwargs")
        node.body[:0] = ast.parse("""
render_options = dict(kernel_options)
if hint_override is not None:
    render_options['hint_override'] = hint_override
""").body
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Name) or call.func.id != "NPUTritonTemplateKernel":
                continue
            for keyword in call.keywords:
                if keyword.arg is None and isinstance(keyword.value, ast.Name):
                    if keyword.value.id == "kernel_options":
                        keyword.value.id = "render_options"
                        matches += 1
    if matches != 1:
        raise RuntimeError("default NPU template renderer structure changed")
    ast.fix_missing_locations(tree)
    namespace = dict(vars(shared_selector))
    namespace["NPUTritonTemplateKernel"] = ExperimentalTemplateKernel
    exec(compile(tree, inspect.getsourcefile(shared_selector.NPUTritonTemplate.generate), "exec"), namespace)
    return namespace["generate"]


def _experimental_runtime_renderer():
    function = shared_selector.NPUTritonTemplate.make_runtime_renderer_factory
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    namespace = dict(vars(shared_selector))
    namespace["NPUTritonTemplateKernel"] = ExperimentalTemplateKernel
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


class NPUTritonTemplate(shared_selector.NPUTritonTemplate):
    generate = _experimental_generate()
    make_runtime_renderer_factory = _experimental_runtime_renderer()


@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)


@SymbolicGridFn
def flex_attention_backward_grid(
    batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv
):
    return (
        cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
        + cdiv(num_key_value, meta["BLOCK_N1"]),
        batch_size,
        kv_heads,
    )


@SymbolicGridFn
def flex_decoding_grid(
    batch_size, kv_heads, gqa_group_size, seq_len_q, d_model, meta, *, cdiv
):
    return (
        cdiv(seq_len_q * gqa_group_size, meta["BLOCK_M"]),
        batch_size * kv_heads,
        meta["SPLIT_KV"],
    )


def _without_nan_propagation(source):
    return source.replace("propagate_nan=tl.PropagateNan.ALL,", "").replace("propagate_nan=True,", "")


def _template(name, grid, source, *, compile_options=None, manual_output_buffer=None):
    kwargs = {"name": f"te_{name}", "grid": grid, "source": _without_nan_propagation(source)}
    if compile_options is not None:
        kwargs["compile_options"] = compile_options
    if manual_output_buffer is not None:
        kwargs["manual_output_buffer"] = manual_output_buffer
    return NPUTritonTemplate(**kwargs)


def _guard_empty_backward_blocks(source):
    replacements = (
        (
            "        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)",
            "        has_q_blocks = False\n"
            "        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)",
        ),
        (
            "            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)",
            "            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)\n"
            "            has_q_blocks = has_q_blocks | (sparse_q_num_blocks > 0)",
        ),
        (
            "                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_full_q_num_blks_offset)",
            "                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_full_q_num_blks_offset)\n"
            "                has_q_blocks = has_q_blocks | (sparse_q_num_blocks > 0)",
        ),
        (
            "        dk *= SM_SCALE",
            "        dk *= SM_SCALE\n"
            "        dk = tl.where(has_q_blocks, dk, tl.zeros_like(dk))",
        ),
        (
            "            tl.store(dv_ptrs, dv)",
            "            if has_q_blocks:\n"
            "                tl.store(dv_ptrs, dv)",
        ),
        (
            "            tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))",
            "            if has_q_blocks:\n"
            "                tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("default FlexAttention backward template changed")
        source = source.replace(old, new)
    return source


compute_compact_sparse_mask_offsets_kernel = r"""
{{def_kernel("Q_OFFSETS", "TOTAL_BLOCKS", "KV_NUM_BLKS")}}
    sparse_z = {{size("KV_NUM_BLKS", 0)}}
    sparse_h = {{size("KV_NUM_BLKS", 1)}}
    sparse_q = {{size("KV_NUM_BLKS", 2)}}
    row_count = sparse_z * sparse_h * sparse_q
    stride_num_z, stride_num_h, stride_num_q = {{stride("KV_NUM_BLKS")}}
    stride_off_z, stride_off_h, stride_off_q = {{stride("Q_OFFSETS")}}
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row in range(pid, row_count, num_programs):
        q_idx = row % sparse_q
        tmp = row // sparse_q
        h_idx = tmp % sparse_h
        z_idx = tmp // sparse_h
        num_offset = z_idx * stride_num_z + h_idx * stride_num_h + q_idx * stride_num_q
        num_blocks = tl.load(KV_NUM_BLKS + num_offset).to(tl.int32)
        base = tl.atomic_add(TOTAL_BLOCKS, num_blocks)
        offset = z_idx * stride_off_z + h_idx * stride_off_h + q_idx * stride_off_q
        tl.store(Q_OFFSETS + offset, base)
"""


@SymbolicGridFn
def compact_offsets_grid(row_count, meta, *, min, max):
    return (max(1, min(row_count, meta["NUM_VECTOR_CORE"])), 1, 1)


flex_attention_template = _template(
    "flex_attention", flex_attention_grid,
    load_flex_template("flex_attention") + load_flex_template("utilities") + load_flex_template("common"),
    compile_options=NPUTemplateCompileOption(),
)
flex_attention_backward_template = _template(
    "flex_attention_backward", flex_attention_backward_grid,
    _guard_empty_backward_blocks(load_flex_template("flex_backwards")) + load_flex_template("utilities"),
    compile_options=NPUTemplateCompileOption(),
)
flex_decoding_template = _template(
    "flex_decoding", flex_decoding_grid,
    load_flex_template("flex_decode") + load_flex_template("utilities") + load_flex_template("common"),
    compile_options=NPUTemplateCompileOption(),
)

for _name, _grid, _source, _options, _buffer in (
    ("flex_attention_fwd_mask_out", flex_attention_in_loop_grid, _FWD_MASK_OUT_SOURCE, _FWD_COMPILE_OPTIONS, None),
    ("flex_attention_compact_offsets", compact_offsets_grid, compute_compact_sparse_mask_offsets_kernel, None, None),
    ("flex_attention_compact_mapping", compact_mapping_grid, compute_compact_sparse_mask_mapping_kernel, None, None),
    ("flex_attention_fwd_mask_compact", compact_sparse_mask_grid, compute_sparse_mask_kernel_compact, None, None),
    ("flex_attention_bwd_mask_compact", compact_sparse_mask_grid,
     compute_bwd_sparse_mask_kernel_compact, None, "arg_SPARSE_MASK"),
    ("flex_attention_bwd_mask_pos", sparse_mask_block_pos_grid,
     compute_sparse_mask_block_pos_kernel, None, None),
    ("flex_attention_bwd_dq_mask_out", flex_attention_backward_dq_grid,
     _DEFAULT_BWD_DQ_SOURCE, _BWD_DQ_COMPILE_OPTIONS, None),
    ("flex_attention_bwd_dkdv_mask_out", flex_attention_backward_dkdv_grid,
     _DEFAULT_BWD_DKDV_SOURCE, _BWD_DKDV_COMPILE_OPTIONS, None),
    ("flex_attention_bwd_dkdv_tasklist", flex_attention_backward_dkdv_grid,
     _DEFAULT_BWD_TASKLIST_SOURCE, _BWD_DKDV_COMPILE_OPTIONS, None),
    ("flex_attention_bwd_dkdv_tasklist_no_split", flex_attention_backward_dkdv_grid,
     _DEFAULT_BWD_TASKLIST_SOURCE, _BWD_DKDV_COMPILE_OPTIONS, None),
    ("flex_attention_bwd_dkdv_reduce", flex_attention_backward_dkdv_grid,
     flex_attention_backward_dkdv_reduce_source, _BWD_DKDV_COMPILE_OPTIONS, None),
):
    globals()[_name] = _template(_name, _grid, _source, compile_options=_options, manual_output_buffer=_buffer)
