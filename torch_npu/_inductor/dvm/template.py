import torch
from torch._inductor import config, ir, scheduler
from torch._inductor.dependencies import MemoryDep
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch import (
    ir as npu_ir,
    lowering as npu_lowering,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering import (
    fetch_graphs,
    merge_traced_graphs,
)
from torch_npu._inductor.lowering_common import TracedGraph, create_fake_input

from .op_emitter import mm_rule

aten = torch.ops.aten
_orig_npu_subtract_graph = npu_ir.subtract_graph


def _make_template_input_graph(inp, name=None):
    traced_graph = TracedGraph()
    placeholder = traced_graph.graph.placeholder(name or inp.get_name())
    placeholder.meta["val"] = create_fake_input(
        inp.get_size(), inp.get_stride(), inp.get_device(), inp.get_dtype()
    )
    traced_graph.last_node = placeholder
    return traced_graph


def _make_matmul_input_graphs(inputs, node_name):
    graphs = []
    bindings = {}
    for index, inp in enumerate(inputs):
        logical_name = f"_dvm_{node_name}_mat_input_{index}"
        graphs.append(_make_template_input_graph(inp, logical_name))
        bindings[logical_name] = inp
    return graphs, bindings


class _DvmTemplateGraph(TracedGraph):
    def __init__(self, traced_graph, input_bindings):
        self.graph = traced_graph.graph
        self.last_node = traced_graph.last_node
        self.sym_nodes = traced_graph.sym_nodes
        self.input_bindings = input_bindings

    def get_placeholder_names(self):
        return {
            self.input_bindings[name].get_name()
            if name in self.input_bindings
            else name
            for name in super().get_placeholder_names()
        }


class DvmTemplateBuffer(ir.TemplateBuffer):
    def __init__(self, layout, inputs, traced_graph, input_bindings):
        self.traced_graph = _DvmTemplateGraph(traced_graph, input_bindings)
        self.input_bindings = input_bindings
        super().__init__(layout, inputs, make_kernel_render=None)
        # Reuse NPU meta_kernel's snode.node.data.traced_graph rebuild path.
        self.data = self

    def get_traced_graph(self):
        return _make_template_input_graph(self)


def _subtract_dvm_template_graph(graph1, graph2, node_name=None):
    if (
        node_name is not None
        and isinstance(V.graph.try_get_buffer(node_name), DvmTemplateBuffer)
        and graph2.last_node.op == "placeholder"
        and graph2.last_node.name == node_name
    ):
        node_name = None
    return _orig_npu_subtract_graph(graph1, graph2, node_name)


def _register_dvm_mm_template_lowerings():
    for op in (aten.mm, aten.bmm, aten.addmm, aten.baddbmm):
        if op in anir_config.FALLBACK_LIST:
            anir_config.FALLBACK_LIST.remove(op)
        if op not in anir_config.GENERATE_LIST:
            anir_config.GENERATE_LIST.append(op)

    def make_mm_template(op, mat1, mat2, *, layout=None):
        if V.graph.cpp_wrapper:
            return npu_lowering.fallback_handler(op)(mat1, mat2)

        _, _, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
        current_node = V.graph.current_node
        if current_node is None or "val" not in current_node.meta:
            return npu_lowering.fallback_handler(op)(mat1, mat2)
        if k == 1:
            mat1 = npu_lowering.expand(ir.TensorBox.create(mat1), layout.size)
            mat2 = npu_lowering.expand(ir.TensorBox.create(mat2), layout.size)
            return npu_lowering.lowerings[aten.mul.Tensor](mat1, mat2)

        inputs = [mat1, mat2]
        input_graphs, input_bindings = _make_matmul_input_graphs(
            inputs, current_node.name
        )
        traced_graph = merge_traced_graphs(input_graphs, op, current_node.name)
        traced_graph.last_node.meta["val"] = current_node.meta["val"]
        if not mm_rule(traced_graph.last_node):
            return npu_lowering.fallback_handler(op)(mat1, mat2)

        return ir.TensorBox.create(
            DvmTemplateBuffer(
                layout,
                inputs,
                traced_graph,
                input_bindings=input_bindings,
            )
        )

    @npu_lowering.register_lowering([aten.mm.default], type_promotion_kind=None)
    def dvm_mm(mat1, mat2, *, layout=None):
        return make_mm_template(aten.mm.default, mat1, mat2, layout=layout)

    @npu_lowering.register_lowering([aten.bmm.default], type_promotion_kind=None)
    def dvm_bmm(mat1, mat2, *, layout=None):
        return make_mm_template(aten.bmm.default, mat1, mat2, layout=layout)

    def make_addmm_template(
        op,
        inp,
        mat1,
        mat2,
        *,
        alpha=1,
        beta=1,
        layout=None,
    ):
        _, _, k, layout, mat1, mat2, expanded_inp = mm_args(
            mat1, mat2, inp, layout=layout
        )
        if op is aten.addmm.default and k == 1:
            mul = npu_lowering.lowerings[aten.mul.Tensor]
            add = npu_lowering.lowerings[aten.add.Tensor]
            inp = ir.TensorBox.create(expanded_inp)
            product = mul(mat1, mat2)
            if alpha != 1:
                product = mul(product, alpha)
            if beta != 1:
                inp = mul(inp, beta)
            return add(product, inp)

        current_node = V.graph.current_node
        matrix_inputs = [mat1, mat2]
        matrix_graphs, input_bindings = _make_matmul_input_graphs(
            matrix_inputs, current_node.name
        )
        input_graphs = [*fetch_graphs([inp]), *matrix_graphs]
        traced_graph = merge_traced_graphs(
            input_graphs, op, current_node.name, alpha=alpha, beta=beta
        )
        traced_graph.last_node.meta["val"] = current_node.meta["val"]
        if not mm_rule(traced_graph.last_node):
            return npu_lowering.fallback_handler(op)(
                inp, mat1, mat2, alpha=alpha, beta=beta
            )

        return ir.TensorBox.create(
            DvmTemplateBuffer(
                layout,
                [inp, *matrix_inputs],
                traced_graph,
                input_bindings=input_bindings,
            )
        )

    @npu_lowering.register_lowering([aten.addmm.default], type_promotion_kind=None)
    def dvm_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
        return make_addmm_template(
            aten.addmm.default,
            inp,
            mat1,
            mat2,
            alpha=alpha,
            beta=beta,
            layout=layout,
        )

    @npu_lowering.register_lowering([aten.baddbmm.default], type_promotion_kind=None)
    def dvm_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
        return make_addmm_template(
            aten.baddbmm.default,
            inp,
            mat1,
            mat2,
            alpha=alpha,
            beta=beta,
            layout=layout,
        )


def _keep_addmm_for_dvm_template():
    from torch._inductor.fx_passes import post_grad

    for entries in post_grad.pass_patterns[2].patterns.values():
        for entry in entries:
            if entry.extra_check is post_grad.should_prefer_unfused_addmm:
                entry.extra_check = lambda _match: False
                return


def patch_dvm_matmul_template_fusion() -> None:
    _keep_addmm_for_dvm_template()
    npu_ir.subtract_graph = _subtract_dvm_template_graph
    _register_dvm_mm_template_lowerings()


def _buffer_numel(buffer):
    buffer = getattr(buffer, "node", buffer)
    return sympy_product(buffer.get_size())


def _has_unsupported_epilogue_broadcast(node: scheduler.SchedulerNode) -> bool:
    output_shape = tuple(node.node.get_size())
    if len(output_shape) <= 2:
        return False

    for dep in node.read_writes.reads:
        if not isinstance(dep, MemoryDep):
            continue

        input_buffer = V.graph.try_get_buffer(dep.name)
        input_buffer = getattr(input_buffer, "node", input_buffer)
        if input_buffer is None or not hasattr(input_buffer, "get_size"):
            return True

        input_shape = tuple(input_buffer.get_size())
        if len(input_shape) > len(output_shape):
            return True
        input_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
        outer_shape_mismatch = any(
            not V.graph.sizevars.statically_known_equals(input_dim, output_dim)
            for input_dim, output_dim in zip(input_shape[:-2], output_shape[:-2])
        )
        is_broadcast_access = any(
            var not in dep.index.free_symbols for var in dep.var_names
        )
        if outer_shape_mismatch and is_broadcast_access:
            return True
    return False


def can_fuse_dvm_epilogue(
    template_node: scheduler.BaseSchedulerNode,
    epilogue_node: scheduler.BaseSchedulerNode,
) -> bool:
    if not config.epilogue_fusion:
        return False
    template_buffer = template_node.get_template_node()
    if not isinstance(template_buffer, DvmTemplateBuffer):
        return False
    if isinstance(epilogue_node.get_template_node(), DvmTemplateBuffer):
        return False
    if epilogue_node.is_reduction():
        return False
    _, (numel1, rnumel1) = template_node.group
    _, (numel2, rnumel2) = epilogue_node.group
    if numel1 != numel2 or rnumel1 != rnumel2:
        return False

    matmul_numel = _buffer_numel(template_buffer)
    if not all(
        V.graph.sizevars.statically_known_equals(
            matmul_numel, _buffer_numel(output)
        )
        for output in epilogue_node.get_outputs()
    ):
        return False
    if not (template_node.get_buffer_names() & epilogue_node.used_buffer_names()):
        return False

    for node in epilogue_node.get_nodes():
        if (
            not isinstance(node, scheduler.SchedulerNode)
            or not isinstance(node.node, ir.ComputedBuffer)
            or not isinstance(node.node.data, ir.Pointwise)
        ):
            return False
        if _has_unsupported_epilogue_broadcast(node):
            return False

    return True
