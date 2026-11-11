"""FX graph construction, dumping and define_kernel hookup for template
kernels (triton_tem_fused_*) in the accuracy tool.

Background: max-autotune GEMM template kernels bypass the linear-path
graph construction (codegen_node_schedule) at scheduling time and reach
define_kernel with traced_graph_hash=None, so the accuracy tool
(INDUCTOR_ASCEND_DUMP_FX_GRAPH / INDUCTOR_ASCEND_CHECK_ACCURACY) cannot
dump/compare this part of the computation and records one "no valid
dump_path" per forward.

This module does two things:
1. Build the missing traced graph for a template kernel (boundary
   placeholder graphs for the inputs, the template op taken from the
   buffer origins, fused epilogue nodes grafted by name on their own
   traced_graph where available and rebuilt from origins otherwise),
   inject hash placeholders and let the original define_kernel logic
   perform the placeholder substitution (TRACED_GRAPH_HASH / TRACED_GRAPH_DIR);
2. After define_kernel returns, dump the FX module under the same
   conventions as the pointwise path.

Hookup: wrap NPUTritonScheduling.define_kernel via monkeypatch (same
convention as scheduler.patch_scheduler for the stock Scheduler), so
the codegen/ sources stay untouched. The hook only fires when (tool on
&& hash missing && template buffer); everything else passes through
unchanged, and construction/dump failures only warn and fall back to
the original behavior without affecting compilation.
"""

import os

import torch
from torch._inductor.codecache import code_hash
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.ir import TemplateBuffer
from torch._inductor.virtualized import V

from . import config as npu_config
from .config import log
from .lowering_fx import (
    dump_fx_graph_code,
    generate_fx_graph_code,
    get_last_node,
    merge_traced_graphs,
    node_id,
)

# Hashes already dumped: skip re-dumping when the same kernel is recompiled
# via the src_to_kernel cache, matching the pointwise path's "dump each
# kernel once" behavior.
_dumped_hashes = set()


def maybe_prepare_template_fx_dump(src_code, node_schedule):
    """Template-kernel scene check and graph construction entry.

    Fires only when all of the following hold (otherwise returns None and
    the caller passes through unchanged):
      - the tool is on (npu_config.dump_fx_graph)
      - node_schedule is non-empty and its first entry is a template buffer
        (MultiTemplateBuffer is a TemplateBuffer subclass)
    On construction failure, warn and return None to fall back to the
    original behavior (no dump for this kernel, same as tool-off), never
    affecting the compile flow.
    """
    if not (
        node_schedule
        and npu_config.dump_fx_graph
        and isinstance(getattr(node_schedule[0], "node", None), TemplateBuffer)
    ):
        return None
    try:
        return _prepare_template_fx_dump(src_code, node_schedule)
    except Exception as e:
        log.warning("[ACC_DEBUG] could not build traced fx graph for template kernel %s: %s",
                    node_schedule[0].node.get_name(), repr(e))
        return None


def _prepare_template_fx_dump(src_code, node_schedule):
    """Build a traced fx graph for a (multi-)template kernel and compute its hash.

    Template inputs at scheduling time are plain InputBuffers without a
    traced graph, so each input gets a boundary placeholder graph (same as
    lowering_common.fetch_graphs' fallback for graph-less inputs). The
    template op is taken from the buffer's origins rather than assuming mm
    (pattern-match templates such as mm+bias expose addmm). Fused epilogue
    nodes from node_schedule[1:] are attached in order: prefer grafting by
    name on the epilogue's own traced_graph (merge_fx_graphs, the same
    mechanism as pointwise multi-snode fusion); only nodes without a
    traced_graph (stock lowering output) are rebuilt from origins, with their
    extra read buffers wired in as boundary graphs. Any failure raises; the
    caller falls back to the original behavior.
    """
    from .lowering_common import TracedGraph, create_fake_input, merge_fx_graphs
    from torch.fx.experimental.proxy_tensor import make_fx

    tnode = node_schedule[0].node
    if not npu_config.traced_fx_graph_cache:
        npu_config.traced_fx_graph_cache = os.path.join(
            os.getenv("TORCHINDUCTOR_CACHE_DIR", "."), "traced_fx_graph_cache")

    def _boundary_graph(inp):
        tg_i = TracedGraph()
        nd = tg_i.graph.placeholder(inp.get_name())
        nd.meta["val"] = create_fake_input(
            inp.get_size(), inp.get_stride(), inp.get_device(), inp.get_dtype()
        )
        tg_i.last_node = nd
        return tg_i

    def _ir_buffer_by_name(name):
        if name in V.graph.name_to_buffer:
            return V.graph.name_to_buffer[name]
        return V.graph.graph_inputs.get(name)

    raw_inputs = (getattr(tnode, "inputs", None) or getattr(tnode, "_inputs", None)
                  or getattr(tnode, "original_inputs", None))
    inputs = list(raw_inputs)
    graphs = [_boundary_graph(inp) for inp in inputs]
    in_names = [inp.get_name() for inp in inputs]

    origins = [o.target for o in (getattr(tnode, "origins", None) or [])
               if hasattr(o, "target")]
    if not origins:
        raise RuntimeError(f"template buffer {tnode.get_name()} has no origin op")
    origin_fn = origins[0]
    node_name = f"{str(origin_fn).split('.')[-2]}_{next(node_id)}"
    tg = merge_traced_graphs(graphs, origin_fn, node_name)

    out_name = tnode.get_name()
    for e in node_schedule[1:]:
        e_node = getattr(e, "node", e)
        # Main path: the epilogue node carries its own traced_graph linked at
        # lowering time (a multi-op fused node is the whole compute chain);
        # graft it onto the template graph by name (merge_fx_graphs, same
        # mechanism as pointwise multi-snode fusion): the placeholder matching
        # out_name (the template output) is wired to the current last node
        # automatically, and extra reads (e.g. addmm's bias) naturally become
        # new placeholders. Rebuilding from origins is not viable: a fused
        # epilogue's origins contain multiple ops in no guaranteed order, and
        # wiring origins[0] with "one positional arg per input graph" would
        # synthesize wrong calls like relu(mm_out, bias).
        e_tg = getattr(e_node, "traced_graph", None)
        if e_tg is None and hasattr(e_node, "data"):
            # SchedulerNode entry: node is a ComputedBuffer whose traced_graph
            # hangs off its data (Loops), same place as
            # create_fx_from_snodes_by_traced_graph.
            e_tg = getattr(e_node.data, "traced_graph", None)
        if e_tg is not None:
            tg.last_node.name = out_name
            tg = merge_fx_graphs([tg, e_tg])
            tg.last_node = (tg.last_node[-1] if isinstance(tg.last_node, list)
                            else tg.last_node)
        else:
            # Fallback: when the epilogue node has no traced_graph (stock
            # lowering output), rebuild from the origin op with extra reads
            # wired in as boundary placeholder graphs. That wiring feeds the
            # template output and extra reads positionally to the origin op,
            # which is only safe for simple single-origin epilogues — anything
            # else raises and the caller skips honestly.
            e_origins = [o.target for o in (getattr(e_node, "origins", None) or [])
                         if hasattr(o, "target")]
            if not e_origins:
                raise RuntimeError(f"epilogue node {e_node.get_name()} has no origin op")
            # Names already in the graph (placeholders + current output) are
            # not wired in again as extra reads
            existing = {nd.name for nd in tg.graph.nodes if nd.op == "placeholder"}
            existing.add(out_name)
            extra_reads = []
            for d in e.read_writes.reads:
                if d.name not in existing and d.name not in extra_reads:
                    extra_reads.append(d.name)
            extra_graphs = []
            for rn in extra_reads:
                buf = _ir_buffer_by_name(rn)
                if buf is None:
                    raise RuntimeError(f"epilogue extra input {rn} not found in IR buffers")
                extra_graphs.append(_boundary_graph(buf))
            tg = merge_traced_graphs([tg] + extra_graphs, e_origins[0],
                                     f"epilogue_{next(node_id)}")
        out_name = e_node.get_name()

    # TracedGraph -> executable fx module (same as
    # create_fx_from_snodes_by_traced_graph)
    tg.last_node.name = out_name
    tg.graph.output((tg.last_node,))
    tg.graph.lint()
    # Defensively fill in missing meta["val"] (grafted placeholders should
    # carry the lowering-time fake values; rebuild from IR buffer info when
    # missing, so graph construction degrades to an honest skip instead of
    # dying on KeyError).
    for nd in tg.graph.nodes:
        if nd.op == "placeholder" and nd.meta.get("val") is None:
            buf = _ir_buffer_by_name(nd.name)
            if buf is None:
                raise RuntimeError(
                    f"merged graph placeholder {nd.name} not found in IR buffers")
            nd.meta["val"] = create_fake_input(
                buf.get_size(), buf.get_stride(), buf.get_device(), buf.get_dtype()
            )
    # in_names follows the final graph's placeholders (extra reads on the
    # graft path are introduced by the epilogue graph's own placeholders,
    # consistent with call_args_mapping's by-name mapping).
    in_names = [nd.name for nd in tg.graph.nodes if nd.op == "placeholder"]
    # Template input layout recipe (1-1 with in_names): a template kernel's
    # runtime arguments are the base storages of views/expands (e.g. the
    # broadcast bmm's mat2 is an expanded 2D weight, a transposed mat1 is the
    # base buffer); feeding the base directly into the fx graph through the
    # by-name position mapping would mismatch the placeholder's shape/layout
    # ("mat2 must be a 3D tensor"). Record each placeholder's IR view
    # (size, stride) here; at runtime and in offline replay, torch.as_strided
    # rebuilds the view the kernel actually reads from the base storage.
    # Identity when the view itself is contiguous — behavior unchanged.
    in_layouts = []
    for nd in tg.graph.nodes:
        if nd.op != "placeholder":
            continue
        val = nd.meta.get("val")
        if isinstance(val, torch.Tensor):
            size = list(val.shape)
            stride = list(val.stride())
            # Dynamic shapes put SymInt symbols in size/stride: their repr is
            # a bare name (s77 etc.) that makes the dumped .py fail to import
            # (NameError) and break the forward at comparison time — the same
            # failure class the -1 scalar recipe guards against. Record a
            # layout only when every element is a concrete int; otherwise None
            # (= no as_strided rebuild, identity fallback; symbolic shapes
            # cannot be rebuilt honestly at runtime anyway).
            if all(isinstance(s, int) for s in size + stride):
                in_layouts.append({
                    "size": size,
                    "stride": stride,
                    # Offset of the placeholder's view within its storage,
                    # consumed by the compare side only when the runtime
                    # argument does not already carry the recorded shape
                    # (base-storage rebuild path). 0 for fresh fake
                    # placeholders; recorded for forward-compat.
                    "offset": val.storage_offset(),
                })
            else:
                in_layouts.append(None)
        else:
            in_layouts.append(None)
    gm0 = torch.fx.GraphModule(torch.nn.Module(), tg.graph)
    gm0.recompile()
    fx_inputs = [n.meta["val"] for n in tg.graph.nodes if n.op == "placeholder"]
    num_inputs = len(fx_inputs)

    def _runnable(*args):
        return torch.fx.Interpreter(gm0).run(*args)

    with V.graph.fake_mode:
        gm = make_fx(_runnable)(*fx_inputs)
    view_to_reshape(gm)
    fx_code = gm.code
    last_node = get_last_node(gm)
    fx_outputs = [n.meta["val"] for n in last_node.args[0]]

    # Inject the hash placeholders, then compute the hash with the same
    # formula as the pointwise path: kernel source (placeholders still
    # literal) + fx code + argument shapes.
    marked_src = src_code.replace(
        "inductor_meta={",
        'inductor_meta={"traced_graph_hash": "TRACED_GRAPH_HASH", '
        '"traced_graph_dir": "TRACED_GRAPH_DIR",',
        1,
    )
    if marked_src == src_code:
        raise RuntimeError("template kernel source has no inductor_meta to inject")
    fx_arg_shapes_str = str([getattr(a, "shape", a) for a in fx_inputs + fx_outputs])
    traced_graph_hash = code_hash(marked_src + fx_code + fx_arg_shapes_str)

    non_contiguous_inputs = [
        i for i, inp in enumerate(fx_inputs)
        if torch.is_tensor(inp) and not inp.is_contiguous()
    ]
    return {
        "src_code": marked_src,
        "traced_graph_hash": traced_graph_hash,
        "fx_code": fx_code,
        "in_names": in_names,
        "out_name": out_name,
        "num_inputs": num_inputs,
        "num_outputs": 1,
        "non_contiguous_inputs": non_contiguous_inputs,
        "in_layouts": in_layouts,
    }


def dump_template_fx_graph(template_dump, src_code, kernel_name, kernel):
    """Dump the FX module after define_kernel returns; failures only warn."""
    if template_dump["traced_graph_hash"] in _dumped_hashes:
        return
    try:
        _dump_template_fx_graph(template_dump, src_code, kernel_name, kernel)
        _dumped_hashes.add(template_dump["traced_graph_hash"])
    except Exception as e:
        log.warning("[ACC_DEBUG] could not dump traced fx graph for template kernel %s: %s",
                    template_dump["out_name"], repr(e))


def _dump_template_fx_graph(template_dump, src_code, kernel_name, kernel):
    # Build the fx placeholder order -> kernel argument position mapping from
    # the kernel's call argument names (python_argdefs); the runtime feeds
    # real arguments into the fx recompute through it.
    _, call_args, _, _ = kernel.args.python_argdefs()
    ca = list(call_args)
    mapping = [ca.index(n) for n in template_dump["in_names"]]
    mapping.append(ca.index(template_dump["out_name"]))
    compile_kwargs = {
        "call_args_mapping": mapping,
        "num_inputs": template_dump["num_inputs"],
        "num_outputs": template_dump["num_outputs"],
        "non_contiguous_indices": {
            "inputs": template_dump["non_contiguous_inputs"],
            "outputs": [],
        },
        "mismatch_indices_shapes": {},
        "template_input_layouts": template_dump["in_layouts"],
    }
    # The compile-time matmul precision switch (allow_hf32 → CANN
    # ALLOW_MATMUL_HF32) is baked into the kernel by template rendering
    # (bmm/mm templates use it to choose tl.dot input_precision="tf32"),
    # while the eager reference in the comparison process follows that
    # process's own switch — a replay process in a different state mismatches
    # the kernel's precision mode and fails by construction. Record it in the
    # dump so the replay side can reproduce the compile-time environment; on
    # lookup failure record nothing (replay None = switch untouched, old
    # behavior).
    try:
        from torch_npu.npu import matmul
        compile_kwargs["allow_hf32"] = bool(matmul.allow_hf32)
    except Exception:
        pass
    dump_dir = os.path.join(npu_config.traced_fx_graph_cache,
                            template_dump["traced_graph_hash"])
    os.makedirs(dump_dir, exist_ok=True)
    fx_code = generate_fx_graph_code(template_dump["fx_code"], src_code,
                                     kernel_name, compile_kwargs)
    dump_fx_graph_code(fx_code, dump_dir, template_dump["traced_graph_hash"])
    log.debug("[ACC_DEBUG] dumped traced fx graph for template kernel %s to %s "
              "(call_args_mapping=%s)", kernel_name, dump_dir, mapping)


def patch_npu_template_fx():
    """Wrap NPUTritonScheduling.define_kernel: build the graph + inject hash
    placeholders up front, dump afterwards.

    The original function body (torch_npu/_inductor/codegen/scheduling.py)
    stays untouched; placeholder substitution is done by the stock
    define_kernel's existing logic. When the tool is off or the kernel is a
    normal one, maybe_prepare_template_fx_dump returns None and the wrapper
    passes its arguments through unmodified.
    """
    from .codegen.scheduling import NPUTritonScheduling

    orig_define_kernel = NPUTritonScheduling.define_kernel

    def define_kernel(self, src_code, node_schedule, kernel,
                      traced_graph_hash: str, kernel_graph_hash: str = None):
        # Accuracy-tool hook: only template kernels (missing hash) enter the
        # tool branch; everything else passes through.
        dump_ctx = None
        if traced_graph_hash is None:
            dump_ctx = maybe_prepare_template_fx_dump(src_code, node_schedule)
        if dump_ctx is not None:
            src_code = dump_ctx["src_code"]
            traced_graph_hash = dump_ctx["traced_graph_hash"]

        kernel_name, src_code = orig_define_kernel(
            self, src_code, node_schedule, kernel, traced_graph_hash, kernel_graph_hash)

        if dump_ctx is not None:
            dump_template_fx_graph(dump_ctx, src_code, kernel_name, kernel)
        return kernel_name, src_code

    NPUTritonScheduling.define_kernel = define_kernel
