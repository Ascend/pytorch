import collections
import functools
import itertools
import os
import math
import textwrap
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import sympy
import torch._ops
from sympy.core import Expr, Integer, Symbol
from torch._inductor import ir
from torch._inductor import lowering
from torch._inductor import scheduler
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.ir import (
    ExpandView,
    IndexingConstant,
    is_triton,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    SqueezeView,
    TensorBox,
    IRNode,
    validate_ir,
    View,
)
from torch._inductor.scheduler import (
    Dep,
    WeakDep,
    Scheduler,
    SchedulerNode,
    SchedulerBuffer,
    FusedSchedulerNode,
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
    )
from torch._inductor.utils import ModularIndexing, FloorDiv
from torch._inductor.utils import (
    decode_device,
    sympy_product,
)
from torch._inductor.virtualized import ops, V
from torch._prims_common import (
    canonicalize_dims,
    check,
    dtype_to_type,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    get_computation_dtype,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    FloorDiv,
    Identity,
    ModularIndexing,
)
from .config import log
from .lowering_op_list import GENERATE_LIST, GENERATE_LIST2, LOWERING_OVERLOAD_OP

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu


fn_to_aten_fn = {}
node_id = itertools.count(0)
snodes_to_fx = {}


def register_fn_to_aten_fn(fn, aten_fn=None):
    if fn not in fn_to_aten_fn:
        fn_to_aten_fn[fn] = aten_fn
    return fn


def register_to_aten(aten_fn=None):
    def decorator(fn):
        if fn not in fn_to_aten_fn:
            fn_to_aten_fn[fn] = aten_fn
        return fn

    return decorator


reduction_type_to_aten_fn = {
    "sum": aten.sum,
    "prod": aten.prod,
    "xor_sum": prims.xor_sum,
    "any": aten.any,
    "max": aten.amax,
    "min": aten.amin,
    "argmax": aten.argmax,
    "argmin": aten.argmin
}

operator_to_string = {
    '+': 'a',
    '-': 'sub',
    '*': 'm',
    '/': 'd',
    '(': 'l',
    ')': 'r',
    '.': 'p',
}


def npu_compute_ancestors(self) -> None:
    """
    Populate each node.ancestors
    """
    # note self.nodes is topologically sorted
    name_to_ancestors: Dict[str, OrderedSet[str]] = {}
    for node in self.nodes:
        ancestors: OrderedSet[str] = OrderedSet()
        for dep in node.unmet_dependencies:
            if dep.name not in self.name_to_buf:
                continue
            dep_node_name = self.name_to_buf[dep.name].defining_op.get_name()
            ancestors.add(dep_node_name)
            ancestors |= name_to_ancestors[dep_node_name]
        name_to_ancestors[node.get_name()] = ancestors
        node.ancestors = ancestors

    for order, node in enumerate(self.nodes):
        node.min_order = order
        node.max_order = order


def _npu_prune_redundant_deps(
    node: BaseSchedulerNode,
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
) -> None:
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """
    name_to_dep_count: collections.Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep) and dep.name in name_to_buf:
            op = name_to_buf[dep.name].defining_op
            name_to_dep_count[name_to_fused_node[op.get_name()].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        if isinstance(dep, WeakDep) and dep.name in name_to_buf:
            op_name = name_to_buf[dep.name].defining_op.get_name()
            is_redundant = name_to_dep_count[name_to_fused_node[op_name].get_name()] > 0
            # These can occur because fused nodes always gather deps from their snodes
            # If B has a weakdep on A
            # B gets fused with C, then any time BC is fused, the weakdep will reappear
            is_self_dep = name_to_fused_node[op_name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = OrderedSet(
        dep for dep in node.unmet_dependencies if should_prune(dep)
    )

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


def _npu_get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> List[BaseSchedulerNode]:
    unmet_deps = set()
    if isinstance(
        snode,
        (
            SchedulerNode,
            ExternKernelSchedulerNode,
            NopKernelSchedulerNode,
            FusedSchedulerNode,
        ),
    ):
        for dep in snode.unmet_dependencies:
            unmet_deps.add(dep.name)
    else:
        raise RuntimeError(
            f"get_unmet_dep_nodes is not implemented for {type(snode)}."
        )
    unmet_dep_ops = (self.name_to_buf[dep].defining_op for dep in unmet_deps if dep in self.name_to_buf)
    return list({self.name_to_fused_node[n.get_name()] for n in unmet_dep_ops})

string_to_operator = {v: k for k, v in operator_to_string.items()}


def map_operators_to_strings(expr_str: str):
    expr_str = expr_str.replace(' ', '')
    for op, string in operator_to_string.items():
        expr_str = expr_str.replace(op, string)
    return '_' + expr_str


def map_strings_to_operators(expr_str: str):
    for op, string in string_to_operator.items():
        expr_str = expr_str.replace(op, string)
    return expr_str[1:]


class TracedGraph:
    def __init__(self):
        self.graph = torch.fx.Graph()
        self.last_node: Optional[torch.fx.Node] = None
        self.sym_nodes: Dict[str, torch.fx.Node] = {}

    def __str__(self):
        return str(self.graph)

    def get_placeholder_names(self):
        placeholder_names = set()
        for node in self.graph.nodes:
            if node.op == 'placeholder' and node.name not in self.sym_nodes:
                placeholder_names.add(node.name)
        return placeholder_names

    __repr__ = __str__


def create_fake_input(size, stride, device, dtype):
    size = [V.graph.sizevars.shape_env.create_symintnode(s, hint=None) \
                if isinstance(s, Expr) and not isinstance(s, Integer) else s for s in size]
    stride = [V.graph.sizevars.shape_env.create_symintnode(s, hint=None) \
                  if isinstance(s, Expr) and not isinstance(s, Integer) else s for s in stride]
    with V.graph.fake_mode:
        fake_input = torch.empty_strided(size, stride, device=device, dtype=dtype)
    return fake_input


def create_sym_inputs(traced_graph: TracedGraph, size: List[Expr]):
    for s in size:
        if isinstance(s, (List, Tuple)):
            create_sym_inputs(traced_graph, s)
            continue
        if isinstance(s, Expr) and not isinstance(s, Integer):
            s_name = str(s)
            if not isinstance(s, Symbol):
                s_name = map_operators_to_strings(s_name)
            if s_name in traced_graph.sym_nodes:
                continue
            new_node = traced_graph.graph.placeholder(s_name)
            new_node.meta['val'] = V.graph.sizevars.shape_env.create_symintnode(s, hint=None)
            traced_graph.sym_nodes.update({s_name: new_node})


def process_ir_constant(inp: ExpandView) -> Union[TracedGraph, int, float]:
    skip = False
    if isinstance(inp.data, IndexingConstant):
        dtype = inp.data.dtype
        inp = inp.data.index
        # convert to original dtype.
        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            # sympy inputs
            if isinstance(inp, Expr) and not isinstance(inp, sympy.core.numbers.Number):
                traced_graph = TracedGraph()
                create_sym_inputs(traced_graph, [inp])
                s_name = str(inp)
                if not isinstance(inp, Symbol):
                    s_name = map_operators_to_strings(str(inp))
                traced_graph.last_node = traced_graph.sym_nodes[s_name]
                inp = traced_graph
            else:
                inp = float(inp)
    elif isinstance(inp.data, ir.Constant):
        dtype = inp.data.dtype
        inp = inp.data.value
    else:
        skip = True
    return inp, skip


def fetch_graphs(inputs: Optional[List[TensorBox]]):
    if isinstance(inputs, (TensorBox, ir.StorageBox, ir.View, sympy.Symbol, ir.Constant, ir.ReinterpretView)):
        inputs = [inputs]
    input_graphs = []
    for inp in inputs:
        if isinstance(inp, List):
            input_graphs.append(fetch_graphs(inp))
            continue
        if not isinstance(inp, (
        TensorBox, ir.StorageBox, ir.View, ir.ReinterpretView, ir.PermuteView, ir.SliceView, ir.ExpandView)):
            input_graphs.append(inp)
            continue
        if isinstance(inp, ExpandView):
            inp, skip = process_ir_constant(inp)
            if not skip:
                input_graphs.append(inp)
                continue
        name = inp.get_name()
        traced_graph = inp.get_traced_graph()
        if traced_graph is not None:
            input_graphs.append(traced_graph)
            continue
        traced_graph = TracedGraph()
        device = inp.get_device()
        dtype = inp.get_dtype()
        size = inp.get_size()
        stride = inp.get_stride()
        new_node = traced_graph.graph.placeholder(name)
        fake_input = create_fake_input(size, stride, device, dtype)
        new_node.meta['val'] = fake_input
        traced_graph.last_node = new_node
        input_graphs.append(traced_graph)
    return input_graphs


def merge_traced_graphs(input_graphs: List[TracedGraph], origin_fn, node_name, **kwargs):
    new_graph = TracedGraph()
    exist_nodes: Dict[str, torch.fx.Node] = {}

    def merge_graph(input_graphs: List[TracedGraph]):
        for input_graph in input_graphs:
            if isinstance(input_graph, List):
                merge_graph(input_graph)
                continue
            if not isinstance(input_graph, TracedGraph):
                continue
            for node in input_graph.graph.nodes:
                if node.name in exist_nodes:
                    continue
                new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes[n.name])
                exist_nodes[node.name] = new_node
                if node.name in input_graph.sym_nodes:
                    new_graph.sym_nodes.update({node.name: new_node})

    def parse_args(input_graphs, exist_nodes):
        args = []
        for input_graph in input_graphs:
            if isinstance(input_graph, TracedGraph):
                args.append(exist_nodes[input_graph.last_node.name])
            elif isinstance(input_graph, (List, Tuple)):
                args.append(parse_args(input_graph, exist_nodes))
            else:
                if isinstance(input_graph, Expr) and not isinstance(input_graph, Integer):
                    if not isinstance(input_graph, Symbol):
                        input_graph = map_operators_to_strings(str(input_graph))
                    args.append(new_graph.sym_nodes[str(input_graph)])
                else:
                    args.append(input_graph)
        return args

    num_args = len(input_graphs)

    for k, v in kwargs.items():
        if isinstance(v, Expr) and not isinstance(v, Integer):
            traced_graph = TracedGraph()
            create_sym_inputs(traced_graph, [v])
            s_name = str(v)
            if not isinstance(v, Symbol):
                s_name = map_operators_to_strings(str(v))
            traced_graph.last_node = traced_graph.sym_nodes[s_name]
            kwargs[k] = traced_graph.sym_nodes[s_name]
            input_graphs.append(traced_graph)
    merge_graph(input_graphs)
    input_graphs = input_graphs[:num_args]
    # if inputs do not have any valid graphs, like full/iota
    create_sym_inputs(new_graph, input_graphs)
    args = parse_args(input_graphs, exist_nodes)
    with new_graph.graph.inserting_after(new_graph.last_node):
        new_node = new_graph.graph.call_function(origin_fn, args=tuple(args), kwargs=kwargs)
    new_node.name = node_name
    new_graph.last_node = new_node
    return new_graph


def merge_fx_graphs(traced_graphs: List[TracedGraph]):
    new_graph = TracedGraph()
    exist_nodes: Dict[str, torch.fx.Node] = {}
    last_nodes = []

    def merge_graph(input_graphs: List[TracedGraph]):
        for input_graph in input_graphs:
            if isinstance(input_graph, List):
                merge_graph(input_graph)
                continue
            if not isinstance(input_graph, TracedGraph):
                continue
            for node in input_graph.graph.nodes:
                if node.name in exist_nodes:
                    continue
                new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes[n.name])
                exist_nodes[node.name] = new_node
            last_nodes.append(exist_nodes[input_graph.last_node.name])

    merge_graph(traced_graphs)
    new_graph.last_node = last_nodes
    return new_graph


def subtract_graph(graph1: TracedGraph, graph2: TracedGraph, node_name=None) -> Tuple[TracedGraph, torch.fx.Node]:
    new_graph = TracedGraph()
    last_node2 = graph2.last_node
    graph1_node_names = {node.name for node in graph1.graph.nodes}
    graph2_node_names = {node.name for node in graph2.graph.nodes}
    placeholder = None
    exist_nodes: Dict[str, torch.fx.Node] = {}
    if node_name not in graph1_node_names:
        placeholder = new_graph.graph.placeholder(last_node2.name if node_name is None else node_name)
        exist_nodes[last_node2.name] = placeholder
    for node in graph1.graph.nodes:
        if node.name in graph2_node_names and node.name not in graph1.sym_nodes:
            continue
        new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes[n.name])
        exist_nodes[node.name] = new_node
    new_graph.last_node = exist_nodes[graph1.last_node.name]
    new_graph.sym_nodes = graph1.sym_nodes
    return new_graph, placeholder


def get_last_node(gm: torch.fx.GraphModule):
    last_node = None
    for node in gm.graph.nodes:
        last_node = node
    return last_node


def tensor_info(tensor):
    if isinstance(tensor, (list, tuple)):
        infos = ", ".join(tensor_info(t) for t in tensor)
        return f"[{infos}]"
    if not isinstance(tensor, torch.Tensor):
        return str(tensor)
    info = str(tensor)
    info = info[:-1]
    info += f", strides={tensor.stride()})"
    return info


def create_fx_from_snodes_by_traced_graph(snodes: List[scheduler.SchedulerNode]):
    fx_call_inputs = []
    try:
        for snode in snodes:
            snode.node.data.traced_graph.last_node.name = snode.node.get_name()
    except Exception as e:
        log.warning(f"Could not rebuild fx graph for {snodes}, reason: {e}")
        return None, None, None, None

    if len(snodes) == 1:
        traced_graph = snodes[0].node.data.traced_graph
    else:
        traced_graph = merge_fx_graphs([snode.node.data.traced_graph for snode in snodes])
    fx_inputs = []
    for node in traced_graph.graph.nodes:
        if node.op == 'placeholder':
            fx_call_inputs.append(node.target)
            fx_inputs.append(node.meta['val'])
    non_contiguous_indices = {}
    non_contiguous_indices["inputs"] = [
        i
        for i, inp in enumerate(fx_inputs)
        if torch.is_tensor(inp) and not inp.is_contiguous()
    ]
    num_inputs = len(fx_call_inputs)
    fx_call_outputs = []
    for snode in snodes:
        if snode.has_aliasing_or_mutation():
            for buf in snode.get_outputs():
                if len(buf.get_mutations()):
                    fx_call_outputs.extend(buf.get_mutations())
                elif len(buf.get_aliases()):
                    fx_call_outputs.append(buf.get_name())
        elif snode.node.get_name() not in (V.graph.removed_buffers | V.graph.inplaced_to_remove):
            fx_call_outputs.append(snode.node.get_name())
    num_outputs = len(fx_call_outputs)
    outputs = traced_graph.last_node if isinstance(traced_graph.last_node, List) \
        else [traced_graph.last_node]
    outputs = [
        output
        for output in outputs
        if output.name not in (V.graph.removed_buffers | V.graph.inplaced_to_remove)
    ]
    fx_call_args = fx_call_inputs + fx_call_outputs
    traced_graph.graph.output(tuple(outputs))
    traced_graph.graph.lint()
    orig_module = torch.nn.Module()
    gm = torch.fx.GraphModule(orig_module, traced_graph.graph)
    gm.recompile()

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)

    with V.graph.fake_mode:
        gm = make_fx(runnable_gm)(*fx_inputs)
    view_to_reshape(gm)
    last_node = get_last_node(gm)
    fx_output_nodes = last_node.args[0]
    fx_outputs = [node.meta['val'] for node in fx_output_nodes]
    non_contiguous_indices["outputs"] = [
        i + num_inputs
        for i, call_output in enumerate(fx_call_outputs)
        if not V.graph.try_get_buffer(call_output).layout.is_contiguous()
    ]
    fx_args = fx_inputs + fx_outputs
    snodes_to_fx[str(snodes)] = f"{gm}\n inputs: {tensor_info(fx_inputs)}\n outputs: {tensor_info(fx_outputs)}\n"

    return gm, fx_call_args, fx_args, {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "non_contiguous_indices": non_contiguous_indices,
    }


def create_compile_kwargs(final_kernel, fx_call_args, fx_args):
    _, kernel_call_args, _, arg_types = final_kernel.args.python_argdefs()
    for idx, call_arg in enumerate(fx_call_args):
        if call_arg in final_kernel.args.inplace_buffers:
            fx_call_args[idx] = final_kernel.args.inplace_buffers[call_arg].other_names[-1]
    fx_arg_shapes = [fx_arg.shape for fx_arg in fx_args if isinstance(fx_arg, torch.Tensor)]

    if set(kernel_call_args) != set(fx_call_args):
        return None
    final_kernel.add_numel_to_call_args(final_kernel.kernel_name, kernel_call_args, arg_types)

    index_map = {element: idx for idx, element in enumerate(kernel_call_args)}
    call_args_mapping = [index_map[element] for element in fx_call_args]

    mismatch_indices_shapes = {}

    for i in range(len(fx_call_args)):
        mismatch_indices_shapes[i] = fx_arg_shapes[i]

    return {
        "call_args_mapping": call_args_mapping,
        "mismatch_indices_shapes": mismatch_indices_shapes,
    }


def generate_fx_graph_code(code, kernel_code, kernel_name, compile_kwargs):
    code = textwrap.indent(code, '    ')
    code_template = f"""
import os    
import torch
from torch._inductor.compile_fx import clone_preserve_strides
from torch._dynamo.testing import rand_strided
from torch import device

import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._inductor import get_current_raw_stream as get_raw_stream
from torch_npu._inductor import config as npu_config

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)


class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
{code}
model = GraphModule().npu()
call_args_mapping = {compile_kwargs['call_args_mapping']}
num_inputs = {compile_kwargs['num_inputs']}
num_outputs = {compile_kwargs['num_outputs']}
non_contiguous_indices = {compile_kwargs['non_contiguous_indices']}
mismatch_indices_shapes = {compile_kwargs['mismatch_indices_shapes']}

async_compile = AsyncCompile()
{kernel_name} = async_compile.triton('{kernel_name}', '''
{kernel_code}
    ''', device_str='npu')

async_compile.wait(globals())
del async_compile

def run():

    stream0 = get_raw_stream(0)

    
    args = torch.load(os.path.join(dir_path, "data.pth"))

    call_inputs_indices = call_args_mapping[:num_inputs]
    call_outputs_indices = call_args_mapping[num_inputs:]

    args = [arg.npu() if isinstance(arg, torch.Tensor) else arg for arg in args]

    fx_args = [] 
    for idx in call_args_mapping:
        arg = args[idx]
        if isinstance(arg, torch.Tensor):
            fx_arg = clone_preserve_strides(arg).float() if arg.dtype == torch.bfloat16 else clone_preserve_strides(arg)
            fx_args.append(fx_arg)

    fx_inputs = [fx_args[idx].contiguous() if idx in non_contiguous_indices['inputs'] else fx_args[idx] for idx in range(num_inputs)]
    if len(mismatch_indices_shapes):
        for ind, shape in mismatch_indices_shapes.items():
            if ind >= num_inputs:
                break
            fx_inputs[ind] = fx_inputs[ind].reshape(shape)
    model_outputs = model.forward(*fx_inputs)
    for idx, (out1, out2) in enumerate(zip(model_outputs, fx_args[num_inputs:(num_inputs + num_outputs)])):
        out1 = out1.reshape(out2.shape)
        if idx in non_contiguous_indices['outputs']:
            out2.copy_(out1)
        else: 
            out2.data = out1.data

    {kernel_name}.run(*args, stream=stream0)

    for actual, expected in zip([args[i] for i in call_outputs_indices], fx_args[num_inputs:]):
        if actual.dtype != expected.dtype:
            expected = expected.to(actual.dtype)
        acc_comp_tol = npu_config.acc_comp_tol.get(actual.dtype, npu_config.acc_comp_tol['default'])
        rtol = acc_comp_tol['rtol']
        atol = acc_comp_tol['atol']
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    run()
"""
    return code_template


def dump_fx_graph_code(code, dump_path, traced_graph_hash):
    py_path = os.path.join(dump_path, traced_graph_hash + '.py')
    with open(py_path, 'w') as f:
        f.write(code)


def clone(x, *, memory_format=None):
    # TODO(jansel): memory format
    input_graphs = fetch_graphs(x)
    node_name = f'clone_{next(node_id)}'
    new_graph = merge_traced_graphs(input_graphs, aten.clone, node_name)
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
        traced_graph=new_graph,
        node_name=node_name
    )


def make_pointwise(
        fn,
        override_return_dtype=None,
        override_device=None,
        override_fn_when_input_bool=None,
        override_fn_when_gpu_float64=None,
        allow_alpha=False,
        triton_fallback=None,
        **kwargs
):
    def inner(*inputs: TensorBox, alpha=None):
        if triton_fallback is not None and any(
                isinstance(inp, IRNode) and is_triton(inp) for inp in inputs
        ):
            # not implemented
            if allow_alpha:
                raise RuntimeError("assert allow_alpha is not allowed")
            return triton_fallback(*inputs)

        inputs = lowering.promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            if alpha is not None:
                raise RuntimeError("assert alpha is not None")
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_gpu_device = lowering.is_gpu(decode_device(inputs[0].get_device()).type)

        for other in inputs[1:]:
            if not (isinstance(other, ir.BaseConstant) or len(ranges) == len(other.get_size())):
                raise RuntimeError(f"assert ndim mismatch {fn} {ranges} {other.get_size()}")

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        emulate_precision_casts = (
                V.graph is not None
                and getattr(V.graph, "current_node", None) is not None
                and V.graph.current_node.meta is not None
                and V.graph.current_node.meta.get("low_precision_pointwise_barrier", False)
                and dtype in (torch.bfloat16, torch.float16)
        )

        def inner_fn(index):
            if len(index) != len(ranges):
                raise RuntimeError(f"assert wrong ndim {index} {ranges}")
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif (
                    override_fn_when_gpu_float64
                    and is_gpu_device
                    and dtype == torch.float64
            ):
                return override_fn_when_gpu_float64(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for load in loaders:
                    out = load(index)
                    if emulate_precision_casts:
                        downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_precision_casts:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            device = None
            for i in inputs:
                if lowering.is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        device = override_device or device

        input_graphs = fetch_graphs(inputs)
        node_name = f'pointwise_{next(node_id)}'
        origin_fn = fn_to_aten_fn[fn]
        new_graph = merge_traced_graphs(input_graphs, origin_fn, node_name, **kwargs)

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            node_name=node_name,
            traced_graph=new_graph,
        )

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype, copy=False):
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype, src_dtype=src_dtype)

    register_fn_to_aten_fn(_to_dtype, aten.to.dtype)
    return make_pointwise(_to_dtype, override_return_dtype=dtype, dtype=dtype)(x)


def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = set(lowering._validate_reduction_axis(x, axis))

    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    def loader(index, reduction_index):
        if len(reduction_index) != len(reduced_idx):
            raise RuntimeError("assert reduction index length mismatch")
        if keepdims:
            if len(index) != len(size):
                raise RuntimeError("assert index size length mismatch")
            index = [index[i] for i in kept_idx]
        if len(index) != len(kept_idx):
            raise RuntimeError("assert index kept_idx length mismatch")
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(
                zip(kept_idx, index), zip(reduced_idx, reduction_index)
        ):
            new_index[idx] = var
        return inner_loader(new_index)

    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.S.One
    else:
        new_size = kept_sizes

    inner_loader = x.make_loader()
    return dict(
        device=x.get_device(),
        dst_dtype=override_return_dtype or x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=new_size,
        reduction_ranges=reduced_sizes,
    )


def _register_npu_inductor_fallbacks_fx(make_reduction):

    def transform_args(
            args: List[Any],
            kwargs: Dict[str, Any],
            broadcast: bool,
            type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND],
            convert_input_to_bool: bool,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        args_indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
        kwargs_indices = [k for k, v in kwargs.items() if isinstance(v, TensorBox)]
        # check that there's something to transform
        if not args_indices and not kwargs_indices:
            return args, kwargs

        if type_promotion_kind or convert_input_to_bool:
            if convert_input_to_bool:
                dtype = torch.bool
            else:
                # this is a crude approximation for promoting args
                promoting_args = [
                    a
                    for a in args
                    if isinstance(a, (Number, sympy.Basic)) or hasattr(a, "dtype")
                ]
                # only consider tensor kwargs for promotion, for now
                promoting_args.extend(a for a in kwargs.values() if hasattr(a, "dtype"))
                dtype = lowering.get_promoted_dtype(
                    *promoting_args, type_promotion_kind=type_promotion_kind  # type: ignore[arg-type]
                )

            device = (
                args[args_indices[0]] if args_indices else kwargs[kwargs_indices[0]]
            ).get_device()

            # sometimes args are an immutable list so we can't mutate them
            def promote(arg):
                if isinstance(arg, TensorBox):
                    return to_dtype(arg, dtype)
                elif isinstance(arg, ir.Constant):
                    return ir.Constant(value=arg.value, dtype=dtype, device=device)
                else:
                    return arg

            args = [promote(a) for a in args]
            kwargs = {k: promote(v) for k, v in kwargs.items()}

        if broadcast:
            broadcasted = broadcast_tensors(
                *list(
                    itertools.chain(
                        (args[i] for i in args_indices),
                        (kwargs[k] for k in kwargs_indices),
                    )
                )
            )
            size = list(broadcasted[0].get_size())

            for i, x in zip(args_indices, broadcasted[: len(args_indices)]):
                args[i] = x
            for k, x in zip(kwargs_indices, broadcasted[len(args_indices):]):
                kwargs[k] = x

            for i in range(len(args)):
                if isinstance(args[i], ir.Constant):
                    args[i] = ExpandView.create(args[i], size)
            for k in kwargs:
                if isinstance(kwargs[k], ir.Constant):
                    kwargs[k] = ExpandView.create(kwargs[k], size)

        return args, kwargs

    def _register_lowering(
            aten_fn, decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool
    ):

        """
        Add a lowering to lowerings dict

        Arguments:
            aten_fn: torch.ops.aten.* fn we are lowering
            decomp_fn: alternate implementation on our IR
            broadcast: True to apply broadcasting to tensor inputs
            type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
            convert_input_to_bool: some logical ops require inputs are converted to bool
        """

        @functools.wraps(decomp_fn)
        def wrapped(*args, **kwargs):
            args: List[Any] = list(args)
            kwargs: Dict[str, Any] = dict(kwargs)
            unpacked = False
            #  maybe we need to use pytrees here
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                unpacked = True
                args = list(args[0])

            if not all(
                    (fn in lowering.fallbacks or lowering.in_namespace(fn, "_c10d_functional")) for fn in aten_fn
            ):
                # explicitly assert for "out=" ops for better error messages
                if any(x == "out" for x in kwargs.keys()):
                    raise RuntimeError("assert out= ops aren't yet supported")

            args, kwargs = transform_args(
                args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
            )

            if unpacked:
                args = [args]

            out = decomp_fn(*args, **kwargs)
            validate_ir(out)

            return out

        aten_fn = lowering.get_overloads(aten_fn)

        lowering.lowerings.update(dict.fromkeys(aten_fn, wrapped))
        return wrapped

    def register_lowering(
            aten_fn,
            broadcast=False,
            type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            convert_input_to_bool=False,
    ):

        """
        Shim to support decorator syntax.
        """
        return functools.partial(
            _register_lowering,
            aten_fn,
            broadcast=broadcast,
            type_promotion_kind=type_promotion_kind,
            convert_input_to_bool=convert_input_to_bool,
        )


    @register_lowering(prims.convert_element_type, type_promotion_kind=None)
    def _convert_element_type(x: TensorBox, dtype: torch.dtype):
        if dtype.is_complex or x.get_dtype().is_complex:
            if x.get_size():
                # Decompose since aa aten fallback is more friendly for c++ codegen.
                # This decomposition doesn't work for empty tensor, which needs more investigation.
                dst = empty_like(x, dtype=dtype)
                ir.InplaceCopyFallback.create(dst, x)
                return dst
            else:
                return lowering.fallback_handler(
                    prims.convert_element_type.default, add_to_fallback_set=False
                )(x, dtype)
        return to_dtype(x, dtype, copy=True)

    def register_pointwise(
            aten_fn,
            name=None,
            broadcast=True,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            convert_input_to_bool=False,
            override_return_dtype=None,
            override_fn_when_input_bool=None,
            allow_alpha=False,
            use_libdevice_for_f64=False,
            triton_fallback=None,
    ):
        """A pointwise function that maps ops.{name} to inputs"""
        name = name or aten_fn.__name__
        fn = ops_wrapper(name)
        if use_libdevice_for_f64:
            fn_libdevice = ops_wrapper("libdevice_" + name)
            lowering.register_op_dtype_propagation_rules(
                "libdevice_" + name, type_promotion_kind, override_return_dtype
            )

        lowering.register_op_dtype_propagation_rules(
            name, type_promotion_kind, override_return_dtype
        )

        if override_fn_when_input_bool is not None:
            override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

        fn = register_fn_to_aten_fn(fn, aten_fn)

        fn = make_pointwise(
            fn,
            override_return_dtype=override_return_dtype,
            override_fn_when_input_bool=override_fn_when_input_bool,
            override_fn_when_gpu_float64=fn_libdevice if use_libdevice_for_f64 else None,
            # type: ignore[possibly-undefined]
            allow_alpha=allow_alpha,
            triton_fallback=triton_fallback,
        )
        fn = register_lowering(
            aten_fn,
            broadcast=broadcast,
            type_promotion_kind=type_promotion_kind,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)

        if hasattr(prims, name):
            register_lowering(
                getattr(prims, name),
                type_promotion_kind=None,
                convert_input_to_bool=convert_input_to_bool,
            )(fn)
        return fn

    

    @register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
    def where(cond, a, b):
        def fn(*args):
            return ops.where(*args)

        if isinstance(a, (float, int)):
            a = lowering.constant_like(a)(b)
        if isinstance(b, (float, int)):
            b = lowering.constant_like(b)(a)

        args = [cond, a, b]
        dtype = lowering.get_promoted_dtype(
            args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        )
        indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
        for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
            args[i] = x
        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
        register_fn_to_aten_fn(fn, aten.where)
        return make_pointwise(fn, override_return_dtype=dtype)(
            args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype)
        )

    @register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
    def broadcast_tensors(*inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            return broadcast_tensors(*inputs[0])
        target: List[sympy.Expr] = functools.reduce(
            lowering.broadcast_symbolic_shapes, [x.get_size() for x in inputs], []
        )
        outputs = []
        for x in inputs:
            sizes = x.get_size()
            if len(sizes) != len(target) or any(
                    (
                            (
                                    V.graph.sizevars.shape_env.evaluate_expr(
                                        sympy.Eq(a, 1), size_oblivious=True
                                    )
                                    and not V.graph.sizevars.shape_env.evaluate_expr(
                                sympy.Eq(b, 1), size_oblivious=True
                            )
                            )
                            or (
                                    not V.graph.sizevars.shape_env.evaluate_expr(
                                        sympy.Eq(a, 1), size_oblivious=True
                                    )
                                    and V.graph.sizevars.shape_env.evaluate_expr(
                                sympy.Eq(b, 1), size_oblivious=True
                            )
                            )
                    )
                    for a, b in zip(sizes, target)
            ):
                x = expand(x, target)
            outputs.append(x)
        return outputs

    @register_lowering(aten.squeeze, type_promotion_kind=None)
    def squeeze(x, dim=None):
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be instance of TensorBox")

        if dim is None:
            return TensorBox(SqueezeView.create(x.data))

        dim = (
            V.graph.sizevars.evaluate_static_shape(dim)
            if isinstance(dim, (int, sympy.Expr))
            else tuple(V.graph.sizevars.evaluate_static_shape(d) for d in dim)
        )
        dim = canonicalize_dims(len(x.get_size()), dim)  # type: ignore[call-overload]
        dims = set((dim,) if not isinstance(dim, tuple) else dim)

        new_shape = []
        for d, s in enumerate(x.get_size()):
            if not (
                    d in dims
                    and V.graph.sizevars.evaluate_expr(sympy.Eq(s, 1), size_oblivious=True)
            ):
                new_shape.append(s)

        # squeeze does nothing if the size isn't 1
        return view(x, new_shape) if new_shape != x.get_size() else x

    @register_lowering([aten.squeeze_])
    def squeeze_(x, dim=None):
        val = squeeze(x, dim)
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        if not isinstance(val, TensorBox):
            raise RuntimeError("assert val should be  instance of TensorBox")
        x.data = val.data
        return x

    @register_lowering(aten.isinf)
    def isinf(x):
        if lowering.is_integer_type(x):
            return full_like(x, False, dtype=torch.bool)
        fn = ops_wrapper("isinf")
        register_fn_to_aten_fn(fn, aten.isinf)
        return make_pointwise(fn, override_return_dtype=torch.bool)(x)

    @register_lowering(aten.isnan)
    def isnan(x):
        if lowering.is_integer_type(x):
            return full_like(x, False, dtype=torch.bool)
        fn = ops_wrapper("isnan")
        register_fn_to_aten_fn(fn, aten.isnan)
        return make_pointwise(fn, override_return_dtype=torch.bool)(x)

    @register_lowering(aten.ceil)
    def ceil(x):
        if lowering.is_integer_type(x):
            return clone(x)
        fn = ops_wrapper("ceil")
        register_fn_to_aten_fn(fn, aten.ceil)
        return make_pointwise(fn)(x)

    @register_lowering(aten.floor)
    def floor(x):
        if lowering.is_integer_type(x):
            return clone(x)
        fn = ops_wrapper("floor")
        register_fn_to_aten_fn(fn, aten.floor)
        return make_pointwise(fn)(x)

    @register_lowering(aten.round.default)
    def round(x):
        if lowering.is_integer_type(x):
            return clone(x)
        else:
            fn = ops_wrapper("round")
            register_fn_to_aten_fn(fn, aten.round)
            return make_pointwise(fn)(x)

    @register_lowering(aten.trunc)
    def trunc(x):
        if lowering.is_integer_type(x):
            return clone(x)
        fn = ops_wrapper("trunc")
        register_fn_to_aten_fn(fn, aten.trunc)
        return make_pointwise(fn)(x)

    @register_lowering(aten.expand, type_promotion_kind=None)
    def expand(x, sizes):
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        (x,) = lowering.promote_constants([x])
        if isinstance(x, ir.BaseConstant):
            return ExpandView.create(x, tuple(sizes))
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        if not isinstance(sizes, (list, tuple)):
            raise RuntimeError("assert x should be  instance of (list, tuple)")
        if tuple(x.get_size()) == tuple(sizes):
            return x

        if not free_unbacked_symbols(x.get_size()):
            x_size_product = V.graph.sizevars.size_hint(sympy_product(x.get_size()))
            # It would be better to realize the input if any of its sizes
            # are unbacked, because typically the size will be non-zero.  However,
            # this cannot be done directly as below as we'll choke on the size_hint
            # here
            if x_size_product > 0 and not free_unbacked_symbols(sizes):
                # maybe realize input before broadcasting it
                x.mark_reuse(
                    V.graph.sizevars.size_hint(sympy_product(sizes)) // x_size_product
                )
        input_graphs = fetch_graphs([x.data, tuple(sizes)])
        node_name = f'expand_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.expand, node_name)
        return TensorBox(ExpandView.create(x.data, tuple(sizes), traced_graph=new_graph, node_name=node_name))

    @register_lowering(aten.expand_as, type_promotion_kind=None)
    def expand_as(x, y):
        return expand(x, y.get_size())

    def to_device(x: TensorBox, device: torch.device, *, copy=False, non_blocking=False):
        device = decode_device(device)
        if x.get_device() == device:
            return clone(x) if copy else x
        input_graphs = fetch_graphs([x, device])
        node_name = f'device_put_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, prims.device_put, node_name, non_blocking=non_blocking)
        out_data = ir.DeviceCopy.create(x, device, non_blocking)
        out_data._post_init_setattr("traced_graph", new_graph)
        out_data._post_init_setattr("node_name", node_name)
        out = TensorBox.create(out_data)
        return out

    @register_lowering(prims.device_put, type_promotion_kind=None)
    def _device_put(x: TensorBox, device: torch.device, non_blocking=False):
        return to_device(x, device, copy=True, non_blocking=non_blocking)
    
    @register_lowering(aten.repeat)
    def repeat(x, repeats):
        input_graphs = fetch_graphs([x, repeats])
        node_name = f'repeat_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.repeat, node_name)
        old_size = list(x.get_size())
        if len(repeats) > len(old_size):
            old_size = [sympy.S.One] * (len(repeats) - len(old_size)) + old_size
            x = view(x, list(old_size))
        if len(repeats) != len(x.get_size()):
            raise RuntimeError("assert repeat should have same size as x.size")

        new_size = list(x.get_size())

        zero_tensor = False
        for i in range(len(repeats)):
            if repeats[i] == 0:
                zero_tensor = True
            new_size[i] = new_size[i] * repeats[i]

        if zero_tensor:
            return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
        if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
            return clone(expand(x, new_size))

        x_loader: Callable[[Any], Any]

        def inner_fn(index):
            if len(index) != len(repeats):
                raise RuntimeError("assert repeat should have same length as repeats")
            index = list(index)
            for i in range(len(repeats)):
                if repeats[i] != 1:
                    if old_size[i] == 1:
                        index[i] = sympy.S.Zero
                    else:
                        index[i] = ModularIndexing(index[i], 1, old_size[i])
            return x_loader(index)

        old_size_product = V.graph.sizevars.size_hint(sympy_product(old_size))
        if old_size_product > 0:
            # maybe realize the input
            x.mark_reuse(
                V.graph.sizevars.size_hint(sympy_product(new_size)) // old_size_product
            )

        x_loader = x.make_loader()
        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=list(new_size),
            traced_graph=new_graph,
            node_name=node_name
        )

    @register_lowering(aten._unsafe_view, type_promotion_kind=None)
    @register_lowering(aten.view, type_promotion_kind=None)
    @register_lowering(aten.reshape, type_promotion_kind=None)
    def view(x, sizes):
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        if not isinstance(sizes, (list, tuple)):
            raise RuntimeError("assert sizes should be  instance of (list, tuple)")
        input_graphs = fetch_graphs([x.data, sizes])
        node_name = f'view_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.reshape, node_name)
        return TensorBox(View.create(x.data, sizes, traced_graph=new_graph, node_name=node_name))

    @register_lowering(aten.permute, type_promotion_kind=None)
    def permute(x, dims):
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        if not isinstance(dims, (list, tuple)):
            raise RuntimeError("assert dims should be  instance of (list, tuple)")
        input_graphs = fetch_graphs([x.data, dims])
        node_name = f'permute_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.permute, node_name)
        return TensorBox(PermuteView.create(x.data, tuple(dims), traced_graph=new_graph, node_name=node_name))

    @register_lowering(aten.slice, type_promotion_kind=None)
    def slice_(x, dim=0, start=0, end=2 ** 63, step=1, clamp=True):
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        dim = _validate_dim(x, dim, 0)
        input_graphs = fetch_graphs([x.data])
        node_name = f'slice_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.slice, node_name, dim=dim, start=start, end=end, step=step)

        return TensorBox(
            ir.SliceView.create(x.data, dim, start, end, step, traced_graph=new_graph, node_name=node_name))

    @register_lowering(aten.select, type_promotion_kind=None)
    def select(x, dim, idx):
        idx = View.handle_negative_index(idx, x.get_size()[dim])
        return squeeze(slice_(x, dim, idx, idx + 1), dim)

    @register_lowering(aten.unbind, type_promotion_kind=None)
    def unbind(x, dim=0):
        dim = _validate_dim(x, dim, 0)
        x_size = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim])
        result = [select(x, dim, i) for i in range(x_size)]
        return result

    @register_lowering(aten.unsqueeze, type_promotion_kind=None)
    def unsqueeze(x, dim):
        dim = _validate_dim(x, dim, 1)
        new_shape = list(x.get_size())
        new_shape.insert(dim, sympy.S.One)
        return view(x, new_shape)

    @register_lowering(aten.unsqueeze_, type_promotion_kind=None)
    def unsqueeze_(x, dim):
        val = unsqueeze(x, dim)
        if not isinstance(x, TensorBox):
            raise RuntimeError("assert x should be  instance of TensorBox")
        if not isinstance(val, TensorBox):
            raise RuntimeError("assert val should be  instance of TensorBox")
        x.data = val.data
        return x

    def _validate_dim(x, dim, offset=0):
        dim = V.graph.sizevars.shape_env.evaluate_expr(sympy.sympify(dim))
        ndim = len(x.get_size())
        if dim < 0:
            dim += ndim + offset
        if not (0 <= dim < ndim + offset):
            raise RuntimeError(f"assert  dim {dim} is out of bounds. Expected: 0 <= dim < {ndim + offset}")
        return dim

    @register_lowering(aten.copy, type_promotion_kind=None)
    def copy(self, src, non_blocking=False):
        x = src
        if self.get_device() != src.get_device():
            x = lowering.to_device(x, self.get_device())
        if self.get_dtype() != src.get_dtype():
            x = to_dtype(x, self.get_dtype())

        if self.get_size() != src.get_size():
            out = expand(x, self.get_size())
            return clone(out)
        return clone(x)

    @register_lowering(prims.iota)
    def iota(
            length,
            *,
            start,
            step,
            dtype,
            device,
            requires_grad,
    ):
        def fn(index):
            return ops.index_expr(step * index[0] + start, dtype=dtype)

        node_name = f'iota_{next(node_id)}'
        new_graph = merge_traced_graphs([length], prims.iota, node_name, \
                                        start=start, step=step, \
                                        dtype=dtype, device=device, \
                                        requires_grad=requires_grad)
        return Pointwise.create(
            device=decode_device(device),
            dtype=dtype,
            inner_fn=fn,
            ranges=[length],
            traced_graph=new_graph,
            node_name=node_name
        )

    @register_lowering(aten.select_scatter, type_promotion_kind=None)
    def select_scatter(x, src, dim: int, index: int):
        if x.get_dtype() != src.get_dtype():
            raise RuntimeError(f"assert Expected dtype {src.get_dtype()}, but got {x.get_dtype()}")
        input_graphs = fetch_graphs([x, src, dim, index])
        node_name = f'select_scatter_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.select_scatter, node_name)
        x_loader = x.make_loader()
        dim = _validate_dim(x, dim, 0)
        if V.graph.sizevars.evaluate_expr(sympy.Lt(index, 0)):
            index = index + x.get_size()[dim]
        V.graph.sizevars.guard_leq(0, index)  # type: ignore[arg-type]
        V.graph.sizevars.guard_lt(index, x.get_size()[dim])  # type: ignore[arg-type]
        src = expand(unsqueeze(src, dim), x.get_size())
        src_loader = src.make_loader()

        def inner_fn(idx):
            return ops.where(
                ops.eq(
                    ops.index_expr(idx[dim], torch.int32),
                    ops.index_expr(index, torch.int32),
                ),
                src_loader(idx),
                x_loader(idx),
            )

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=list(x.get_size()),
            traced_graph=new_graph,
            node_name=node_name
        )

    @register_lowering(aten.slice_scatter, type_promotion_kind=None)
    def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
        if x.get_dtype() != src.get_dtype():
            raise RuntimeError(f"assert Expected dtype {src.get_dtype()}, but got {x.get_dtype()}")
        input_graphs = fetch_graphs([x, src])
        node_name = f'slice_scatter_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.slice_scatter, node_name, \
                                        dim=dim,
                                        start=start,
                                        end=end,
                                        step=step)
        x_loader = x.make_loader()
        dim = _validate_dim(x, dim, 0)
        dim_size = x.get_size()[dim]

        start, end = ir.SliceView.normalize_start_end(x, dim, start, end)

        src_size = list(x.get_size())
        src_size[dim] = FloorDiv(end - start + (step - 1), step)
        src = expand(src, src_size)
        src_loader = src.make_loader()

        def inner_fn(idx):
            if start == 0 and end == dim_size and step == 1:
                # selecting every element is the same as just src.clone()
                return src_loader(idx)

            idx_dim = ops.index_expr(idx[dim], torch.int64)
            src_idx = list(idx)
            src_idx[dim] = FloorDiv(idx[dim] - start, step)

            mask = []
            if start != 0:
                mask.append(
                    ops.ge(
                        idx_dim,
                        ops.index_expr(sympy.expand(start), torch.int64),
                    )
                )
            if end != dim_size:
                mask.append(
                    ops.lt(
                        idx_dim,
                        ops.index_expr(sympy.expand(end), torch.int64),
                    )
                )
            if step != 1:
                mask.append(
                    ops.eq(
                        ops.index_expr(
                            ModularIndexing(idx[dim] - start, 1, step), torch.int64
                        ),
                        ops.constant(0, torch.int64),
                    )
                )
            if not mask:
                raise RuntimeError("assert mask cannot be empty")
            mask = functools.reduce(ops.and_, mask)
            src_val = ops.masked(
                mask,
                lambda: src_loader(src_idx),
                0 if lowering.is_integer_type(x) else 0.0,
            )
            return ops.where(
                mask,
                src_val,
                x_loader(idx),
            )

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=inner_fn,
            ranges=list(x.get_size()),
            traced_graph=new_graph,
            node_name=node_name
        )

    @register_lowering([torch.tensor, aten.scalar_tensor])
    def tensor(data, *, dtype=None, device=None, layout=None, pin_memory=False):
        lowering.assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        lowering.assert_nyi(not pin_memory, "pin_memory")
        input_graphs = fetch_graphs([data])
        node_name = f'tensor_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.scalar_tensor, node_name, \
                                        dtype=dtype,
                                        device='npu',
                                        layout=layout,
                                        pin_memory=False)
        if isinstance(lowering._unwrap(data), int):
            dtype = dtype or torch.int64
        else:
            dtype = dtype or torch.get_default_dtype()

        ranges: List[sympy.Expr] = []

        if isinstance(data, sympy.Basic):

            def inner_fn(index):
                return ops.index_expr(data, dtype)

        elif isinstance(data, (float, int)):

            def inner_fn(index):
                return ops.constant(data, dtype)

        elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
            # inline small tensors
            ranges.append(sympy.Integer(len(data)))

            def inner_fn(index):
                def binary_search(start, end):
                    if start >= end:
                        raise RuntimeError(f"assert start ({start}) must be less than end ({end})")
                    if end - start == 1:
                        return ops.constant(data[start], dtype)
                    mid = (end - start) // 2 + start
                    return ops.where(
                        ops.lt(
                            ops.index_expr(index[0], torch.int64),
                            ops.constant(mid, torch.int64),
                        ),
                        binary_search(start, mid),
                        binary_search(mid, end),
                    )

                if len(data) == 0:
                    return ops.constant(0, dtype)
                return binary_search(0, len(data))

        else:
            return V.graph.add_tensor_constant(
                torch.tensor(data, dtype=dtype, device=device)
            )

        return Pointwise.create(
            device=decode_device(device),
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            traced_graph=new_graph,
            node_name=node_name
        )

    def tensor_constructor(fill_value):
        # torch.zeros, torch.ones, etc
        def inner(
                *size,
                names=None,
                dtype=None,
                device=None,
                layout=None,
                pin_memory=False,
                memory_format=None,
        ):
            lowering.assert_nyi(names is None, "named tensors")
            lowering.assert_nyi(layout in (None, torch.strided), f"layout={layout}")
            lowering.assert_nyi(not pin_memory, "pin_memory")
            device = decode_device(device)
            dtype = dtype or torch.get_default_dtype()
            if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
                size = tuple(size[0])
            # See pytorch issues 118102
            # All sizes at lowering time should be sympy.Symbol, not SymInt!
            for s in size:
                if isinstance(s, torch.SymInt):
                    raise RuntimeError("assert s must not be of type torch.SymInt")
            size = [sympy.expand(s) for s in size]
            return _full(fill_value, device, dtype, size)

        return inner

    def _full(fill_value, device, dtype, size):
        value = fill_value
        if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
            value = value.value

        if isinstance(value, (int, float)):

            def inner_fn(index):
                return ops.constant(value, dtype)

        elif isinstance(value, sympy.Basic):

            def inner_fn(index):
                return ops.index_expr(value, dtype)

        else:
            if len(value.get_size()) != 0:
                raise RuntimeError("assert value should be equal to 0")
            value_loader = value.make_loader()

            def inner_fn(index):
                return value_loader([])

        node_name = f'full_{next(node_id)}'
        new_graph = merge_traced_graphs([size, fill_value], aten.full.default, node_name, \
                                        device='npu', dtype=dtype, layout=torch.strided, pin_memory=False)

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=list(size),
            traced_graph=new_graph,
            node_name=node_name
        )

    @register_lowering(aten.empty_strided)
    def empty_strided(
            size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
    ):
        if not isinstance(size, (list, tuple)):
            raise RuntimeError(f"assert Expected list or tuple")
        if not isinstance(stride, (list, tuple)):
            raise RuntimeError(f"assert Expected list or tuple or None")
        lowering.assert_nyi(not pin_memory, "pin_memory")
        lowering.assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        dtype = lowering.decode_dtype(dtype) or torch.get_default_dtype()
        device = device or torch.tensor(0.0).device
        device = decode_device(device)
        pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
        pointwise.realize()
        buffer = pointwise.data.data
        # explicitly set ranges to zeros in order to make a NopKernelSchedulerNode
        buffer.data = lowering.dataclasses.replace(buffer.data, ranges=[0] * len(size))
        if not isinstance(buffer, ir.ComputedBuffer):
            raise RuntimeError(f"assert Expected ir.ComputedBuffer")
        size = [sympy.expand(s) for s in size]
        stride = (
            [sympy.expand(s) for s in stride]
            if stride
            else ir.FlexibleLayout.contiguous_strides(size)
        )
        buffer.layout = ir.FixedLayout(
            device=device,
            dtype=dtype,
            size=size,
            stride=stride,
        )
        return pointwise

    @register_lowering([torch.empty, aten.empty])
    def empty(
            *size,
            names=None,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            memory_format=None,
    ):
        lowering.assert_nyi(names is None, "named tensors")
        device = decode_device(device)
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        return empty_strided(
            size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
        )

    @register_lowering([torch.full, aten.full])
    def full(size, fill_value, **kwargs):
        if kwargs.get("dtype") is None:
            raise RuntimeError("assert kwargs dtype should be handled by decomposition")
        return tensor_constructor(fill_value)(size, **kwargs)

    register_lowering(aten.clone)(clone)

    @register_lowering(aten.constant_pad_nd, type_promotion_kind=None)
    def constant_pad_nd(x, padding, fill_value=0):
        if (len(padding) % 2) != 0:
            raise RuntimeError("assert len(padding) must % 2=0")

        input_graphs = fetch_graphs([x, padding])
        node_name = f'constand_pad_nd_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.constant_pad_nd, node_name, value=fill_value)

        if all(p == 0 for p in padding):
            return clone(x)

        sizes = x.get_size()

        bounds = list(reversed(list(zip(padding[::2], padding[1::2]))))
        n = len(sizes) - len(bounds)

        # if padding is a complicated expression, hoist it
        bounds_precomp: List[Tuple[sympy.Symbol, Any]] = []
        for low, high in bounds:
            bounds_precomp.append((V.graph.sizevars.lookup_precomputed_size(low), high))  # type: ignore[arg-type]

        output_size = list(sizes[:n])
        mask_sizes = []
        for (low, high), size in zip(bounds, sizes[n:]):
            mask_sizes.append(size)
            output_size.append(sympy.expand(size + low + high))
        if len(output_size) != len(sizes):
            raise RuntimeError("assert len(output_size) must equal to len(sizes)")
        fill_value = dtype_to_type(x.get_dtype())(fill_value)

        def mask(index):
            mask = []
            for idx, (low, high), length in zip(index[n:], bounds, mask_sizes):
                if low != 0:
                    mask.append(lowering.range_mask_low(idx, 0))
                if high != 0:
                    mask.append(lowering.range_mask_high(idx, length))
            mask = functools.reduce(ops.and_, mask)
            return ops.masked(mask, lambda: x_loader(index), fill_value)

        def offset_fn(index):
            new_index = list(index[:n])
            for idx, (low, high) in zip(index[n:], bounds_precomp):
                new_index.append(idx - low)
            if len(new_index) != len(index):
                raise RuntimeError("assert len(new_index) must equal len(index)")
            return mask(new_index)

        x_loader = x.make_loader()
        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=offset_fn,
            ranges=output_size,
            traced_graph=new_graph,
            node_name=node_name
        )

    @make_pointwise
    @register_to_aten(aten_fn=aten.pow)
    def pow_native(a, b):
        return ops.pow(a, b)

    @register_lowering(aten.pow, broadcast=True)
    def pow(a, b):
        if isinstance(b, float) and b == int(b):
            return pow(a, int(b))
        elif isinstance(b, float) and b == 0.5:
            return sqrt(a)
        elif isinstance(b, int) and b == 1:
            return clone(a)

        input_graphs = fetch_graphs([a, b])
        node_name = f'pointwise_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, aten.pow, node_name)

        # Type promotion ensures all tensor arguments have the same type
        dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
        is_integer_pow = is_integer_dtype(dtype)

        # Optimize away small fixed powers, or for integers avoid falling back to ATen
        embed_exponent = isinstance(b, int) and (
                -32 < b < 32 or (is_integer_pow and b >= 0)
        )
        if embed_exponent:
            loader = a.make_loader()

            def fn(idx):
                return lowering.pow_recursive(loader(idx), b, a.get_dtype())

            return Pointwise.create(
                device=a.get_device(),
                dtype=a.get_dtype(),
                inner_fn=fn,
                ranges=a.get_size(),
                node_name=node_name,
                traced_graph=new_graph,
            )

        if isinstance(a, Number):
            if a == 1:
                return full_like(b, 1)
            if a == 2 and is_float_dtype(b.get_dtype()):
                return exp2(b)

        if is_integer_pow:
            # ops.pow doesn't work for integers
            if isinstance(a, Number):
                return lowering.fallback_pow_scalar(a, b)
            elif isinstance(b, Number):
                return lowering.fallback_pow_tensor_scalar(a, b)
            else:
                return lowering.fallback_pow_tensor_tensor(a, b)

        return pow_native(a, b)

    def mutate_to(changed, val, unsafe_alias=False):
        if isinstance(changed, TensorBox):
            changed_data = changed.data
        else:
            changed_data = changed
        if isinstance(val, TensorBox):
            val = val.data

        if not isinstance(val, ir.StorageBox):
            # introduce a copy to handle views
            input_graphs = fetch_graphs([changed, val])
            node_name = f'copy__{next(node_id)}'
            new_graph = merge_traced_graphs(input_graphs, aten.copy_, node_name)
            val = Pointwise.create(
                device=changed.get_device(),
                dtype=changed.get_dtype(),
                inner_fn=val.make_loader(),
                ranges=changed.get_size(),
                traced_graph=new_graph,
                node_name=node_name
            ).data
            if not isinstance(val, ir.StorageBox):
                raise RuntimeError("assert val should be instance of ir.StorageBox")

        if isinstance(changed_data, ir.StorageBox) and not (
                changed_data.is_input_buffer()
                # In AOTI, module parameters and buffers are not lifted as graph inputs
                or changed_data.is_module_buffer()
                or isinstance(changed_data.data, ir.NopKernel)
        ):
            # Fast path, just swing the data pointer
            val.realize()
            changed_data.data = val.data
            return changed

        ir.MutationLayoutSHOULDREMOVE.realize_into(
            val, changed_data, unsafe_alias=unsafe_alias
        )
        return changed

    empty_like = register_lowering(aten.empty_like)(lowering.create_tensor_like(empty))
    ones_like = lowering.create_tensor_like(tensor_constructor(1))
    zeros_like = lowering.create_tensor_like(tensor_constructor(0))

    @register_lowering(aten.full_like, type_promotion_kind=None)
    def full_like(x, fill_value, **kwargs):
        return lowering.create_tensor_like(tensor_constructor(fill_value))(x, **kwargs)

    @register_lowering(aten.fill_)
    def fill_(x, fill_value):
        return mutate_to(x, full_like(x, fill_value))

    @register_lowering(aten.copy_, type_promotion_kind=None)
    def copy_(dst, src, non_blocking=False):
        if dst is src:
            # dst.copy_(dst) can happen from the reinplacing pass
            return dst
        src = lowering.to_device(src, dst.get_device())
        src = to_dtype(src, dst.get_dtype())
        src = expand(src, dst.get_size())
        return mutate_to(dst, src)

    @make_pointwise
    def floordiv(a, b):
        return ops.floordiv(a, b)

    @make_pointwise
    def truncdiv(a, b):
        return ops.truncdiv(a, b)

    @register_lowering(aten.div, broadcast=True)
    def div_mode(a, b, rounding_mode=None):
        both_integer = lowering.is_integer_type(a) and lowering.is_integer_type(b)
        both_boolean = lowering.is_boolean_type(a) and lowering.is_boolean_type(b)

        # floordiv and truncdiv need special handling for integer tensors on Triton,
        # see the discussion at openai triton issues 605
        if rounding_mode == "floor":
            if both_boolean:
                raise RuntimeError("assert floordiv operands cannot be boolean at the same time")
            return floordiv(a, b) if both_integer else floor(div(a, b))
        if rounding_mode == "trunc":
            if both_boolean:
                raise RuntimeError("assert truncdiv operands can not be boolean at the same time")
            return truncdiv(a, b) if both_integer else trunc(div(a, b))
        return div(a, b)

    @register_lowering([aten.mul], broadcast=True)
    def mul(a, b):
        both_bool = lowering.is_boolean_type(a) and lowering.is_boolean_type(b)
        if both_bool:
            return logical_and(a, b)
        else:
            fn = ops_wrapper(aten.mul.__name__)
            fn = register_fn_to_aten_fn(fn, aten.mul)
            return make_pointwise(fn)(a, b)

    @register_lowering([aten.reciprocal], broadcast=True, )
    def reciprocal(a):
        return div(1.0, a)

    @register_lowering([prims.div], broadcast=True)
    def div_prim(a, b):
        is_integral = all(lowering.is_boolean_type(x) or lowering.is_integer_type(x) for x in [a, b])

        if is_integral:
            return truncdiv(a, b)

        if (divisor := lowering.get_constant_value(b)) is not None:
            # Replace divide by constant with multiply by reciprocal
            if divisor.value == 0:
                reciprocal = math.copysign(float("inf"), divisor.value)
            else:
                reciprocal = 1.0 / divisor.value
            return mul(a, reciprocal)

        def fn(*args):
            return ops.truediv(*args)

        fn = register_fn_to_aten_fn(fn, aten.div)
        return make_pointwise(fn)(a, b)

    @register_lowering(
        [aten.true_divide, aten.div.Tensor],
        broadcast=True,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    def div(a, b):
        a, b = lowering.promote_constants(
            (a, b), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )
        return div_prim(a, b)

    @register_lowering(aten.rsqrt)
    def rsqrt(x):
        dtype = x.get_dtype()
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            x = to_dtype(x, torch.get_default_dtype())

        def _rsqrt(x):
            return ops.rsqrt(x)

        register_fn_to_aten_fn(_rsqrt, aten.rsqrt)
        return make_pointwise(_rsqrt)(x)

    @register_lowering(aten.prod)
    def prod(x, axis=None, keepdims=False, *, dtype=None):
        if (
                is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            dtype = torch.int64

        fn = make_reduction("prod", override_return_dtype=dtype)
        return fn(x, axis, keepdims, dtype=dtype)

    @register_lowering(aten.any)
    def reduce_any(x, dim=None, keepdim=False):
        x = to_dtype(x, torch.bool)
        return make_reduction("any")(x, axis=dim, keepdims=keepdim)

    register_lowering(prims.xor_sum)(make_reduction("xor_sum"))
    add = register_pointwise(
        aten.add, allow_alpha=True, override_fn_when_input_bool="logical_or"
    )

    def register_pointwise_numeric(op, name=None, triton_fallback=None):
        return register_pointwise(
            op,
            name=name,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
            triton_fallback=triton_fallback,
        )

    def register_pointwise_numeric_ldf64(op):
        return register_pointwise(
            op,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
            use_libdevice_for_f64=True,
        )

    def register_inplace(aten_op, outplace_op):
        @register_lowering(aten_op, type_promotion_kind=None)
        def fn(*args, **kwargs):
            result = outplace_op(*args, **kwargs)
            result = to_dtype(result, args[0].get_dtype())
            return mutate_to(args[0], result)

        return fn

    rsqrt = register_pointwise_numeric(aten.rsqrt)
    exp = register_pointwise_numeric_ldf64(aten.exp)
    exp2 = register_pointwise_numeric(aten.exp2)
    expm1 = register_pointwise_numeric(aten.expm1)
    relu = register_pointwise(aten.relu)
    sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
    sqrt = register_pointwise_numeric_ldf64(aten.sqrt)
    square = register_pointwise(aten.square)
    sub = register_pointwise(aten.sub, allow_alpha=True)
    register_pointwise_numeric_ldf64(aten.cos)
    register_pointwise_numeric_ldf64(aten.sin)
    abs_val = register_pointwise(aten.abs)
    bitwise_and = register_pointwise(aten.bitwise_and)
    bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)
    bitwise_not = register_pointwise(
        aten.bitwise_not, override_fn_when_input_bool="logical_not"
    )
    bitwise_or = register_pointwise(aten.bitwise_or)
    bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)
    bitwise_xor = register_pointwise(aten.bitwise_xor)
    register_pointwise_numeric(aten.lgamma)
    erf = register_pointwise_numeric(aten.erf)
    register_lowering(
        aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )(erf)

    register_pointwise_numeric(aten.log1p)
    register_pointwise_numeric(aten.tan)
    register_pointwise_numeric(aten.tanh)
    register_pointwise_numeric_ldf64(aten.log)
    logical_and = register_pointwise(
        aten.logical_and,
        type_promotion_kind=None,
        convert_input_to_bool=True,
        override_return_dtype=torch.bool,
    )
    logical_not = register_pointwise(
        aten.logical_not,
        type_promotion_kind=None,
        convert_input_to_bool=True,
        override_return_dtype=torch.bool,
    )
    logical_or = register_pointwise(
        aten.logical_or,
        type_promotion_kind=None,
        convert_input_to_bool=True,
        override_return_dtype=torch.bool,
    )
    logical_xor = register_pointwise(
        aten.logical_xor,
        type_promotion_kind=None,
        convert_input_to_bool=True,
        override_return_dtype=torch.bool,
    )
    maximum = register_pointwise(aten.maximum)
    minimum = register_pointwise(aten.minimum)
    clamp_min = register_pointwise(aten.clamp_min, name='maximum')
    clamp_max = register_pointwise(aten.clamp_max, name='minimum')
    neg = register_pointwise(aten.neg)
    abs_val1 = register_pointwise(aten.abs)
    register_pointwise(aten.remainder)
    sign = register_pointwise(aten.sign, override_fn_when_input_bool="identity")
    register_pointwise(aten.ceil)
    register_pointwise(aten.signbit, override_return_dtype=torch.bool)

    register_lowering(aten._neg_view)(neg)

    register_pointwise(aten.le, override_return_dtype=torch.bool)
    register_pointwise(aten.lt, override_return_dtype=torch.bool)
    register_pointwise(aten.ge, override_return_dtype=torch.bool)
    gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
    register_pointwise(aten.eq, override_return_dtype=torch.bool)
    register_pointwise(aten.ne, override_return_dtype=torch.bool)

    register_pointwise_numeric(aten.cosh)
    register_pointwise_numeric(aten.sinh)
    register_pointwise_numeric(aten.acos)
    register_pointwise_numeric(aten.acosh)
    register_pointwise_numeric(aten.asin)
    register_pointwise_numeric(aten.asinh)
    register_pointwise_numeric(aten.atan2)
    register_pointwise_numeric(aten.atan)
    register_pointwise_numeric(aten.atanh)
    register_pointwise_numeric(aten.copysign)
    register_pointwise_numeric(aten.erfc)
    register_pointwise_numeric(aten.erfinv)
    register_pointwise_numeric(aten.hypot)
    register_pointwise_numeric(aten.log10)
    register_pointwise_numeric(aten.log2)
    register_pointwise_numeric(aten.nextafter)

    register_inplace(aten.add_, add)
    register_inplace(aten.bitwise_and_, bitwise_and)
    register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
    register_inplace(aten.bitwise_not_, bitwise_not)
    register_inplace(aten.bitwise_or_, bitwise_or)
    register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
    register_inplace(aten.bitwise_xor_, bitwise_xor)
    register_inplace(aten.mul_, mul)
    register_inplace(aten.div_.Tensor, div)
    register_inplace(aten.div_.Tensor_mode, div_mode)
    register_inplace(aten.logical_and_, logical_and)
    register_inplace(aten.logical_not_, logical_not)
    register_inplace(aten.logical_or_, logical_or)
    register_inplace(aten.logical_xor_, logical_xor)
    register_inplace(aten.sub_, sub)
    register_inplace(aten.relu_, relu)
    register_inplace(aten.sigmoid_, sigmoid)

    register_lowering(aten.__and__)(bitwise_and)
    register_lowering(aten.__lshift__)(bitwise_left_shift)
    register_lowering(aten.__or__)(bitwise_or)
    register_lowering(aten.__rshift__)(bitwise_right_shift)
    register_lowering(aten.__xor__)(bitwise_xor)

    register_inplace(aten.__iand__, aten.__and__)
    register_inplace(aten.__ilshift__, aten.__lshift__)
    register_inplace(aten.__ior__, aten.__or__)
    register_inplace(aten.__irshift__, aten.__rshift__)
    register_inplace(aten.__ixor__, aten.__xor__)

    ##########################################################################
    @register_lowering([aten.sum, prims.sum])
    def sum_(x, axis=None, keepdims=False, *, dtype=None):
        if (
            is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            dtype = torch.int64

        fn = make_reduction("sum", override_return_dtype=dtype)
        return fn(x, axis, keepdims, dtype=dtype)

    def inductor_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
        if sparse:
            raise ValueError("Sparse tensors are not supported.")
        if not isinstance(weight, TensorBox):
            raise TypeError(f"Expected weight to be a TensorBox, got {type(weight)}.")
        if not isinstance(indices, TensorBox):
            raise TypeError(f"Expected indices to be a TensorBox, got {type(indices)}.")
        if "int" not in str(indices.get_dtype()):
            raise TypeError(f"indices must have integer dtype, got {indices.get_dtype()}.")

        weight_loader = weight.make_loader()
        indices_loader = indices.make_loader()
        indices_ndim = len(indices.get_size())
        weight_size = weight.get_size()
        new_size = [*indices.get_size(), *weight_size[1:]]

        def fn(idx):
            if len(idx) != len(new_size):
                raise ValueError(f"Length mismatch: len(idx)={len(idx)} != len(new_size)={len(new_size)} ({idx} != {new_size})")
            var_index = indices_loader(idx[:indices_ndim])
            weight_idx = [ops.indirect_indexing(var_index, weight_size[0])] + [
                *idx[indices_ndim:]
            ]
            return weight_loader(weight_idx)
        
        input_graphs = fetch_graphs([weight, indices])
        node_name = f'embedding_{next(node_id)}'
        new_graph = merge_traced_graphs(input_graphs, torch.ops.aten.embedding.default, node_name, padding_idx=padding_idx, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)

        return Pointwise.create(
            device=weight.get_device(),
            dtype=weight.get_dtype(),
            inner_fn=fn,
            ranges=new_size,
            traced_graph=new_graph,
            node_name=node_name
        )

    lowering.make_fallback(aten._log_softmax)
    lowering.make_fallback(aten.nll_loss_forward)

    return (squeeze, expand, view, unsqueeze, _validate_dim, full_like, mul, div, rsqrt, add, square, sub)
