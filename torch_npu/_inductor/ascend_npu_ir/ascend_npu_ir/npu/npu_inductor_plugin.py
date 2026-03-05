import atexit
import collections
from collections import Counter
import functools
import itertools
import shutil
import os
from typing import (
    Set,
    Dict,
    Optional,
    Tuple,
    List
)

import torch
import torch.nn.functional as F
from torch.utils._ordered_set import OrderedSet
from torch import _TorchCompileInductorWrapper
from torch import Tensor
from torch._dynamo.utils import set_current_node, UnsupportedFakeTensorException
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import dynamo_timed
from torch._dynamo.device_interface import register_interface_for_device
from torch._dynamo.backends import common 
from torch._dynamo.backends.common import AotAutograd
from torch._inductor import config
from torch._inductor.async_compile import shutdown_compile_workers
from torch._inductor.codegen.common import register_backend_for_device, register_device_op_overrides
from torch._inductor.virtualized import V
from torch._inductor import decomposition as inductor_decomp
from torch._inductor import scheduler
from torch._inductor.scheduler import (
    Dep,
    WeakDep,
    Scheduler, 
    SchedulerNode, 
    SchedulerBuffer,
    FusedSchedulerNode,
    BaseSchedulerNode,
    ForeachKernelSchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
    WhyNoFuse,
    MemoryDep
    )
from torch._decomp import decomposition_table
from torch.utils import _triton
from torch_npu.utils._dynamo_device import NpuInterface
try:
    from torch_npu.npu import device_count
except:
    from torch_npu.npu.utils import device_count

from ..npu.codegen.akg import AkgScheduling
from ..npu.codegen.mlir import NpuMlirScheduling
from ..npu.codegen.wrapper import NpuMlirWrapperCodeGen
from ..npu.npu_lowering import _register_npu_inductor_fallbacks
from ..npu.utils import (
    npu_optimize_fx_graph,
    run_once,
    logger,
)
from .. import config as anir_config
from . import npu_patch_deprecated
from .npu_meta import npu_patch_meta

_triton.has_triton = lambda: False
_triton.has_triton_package = lambda: False

# Fix Error: Exit earlier than child process.
atexit.register(shutdown_compile_workers)

# new npu meta registration.
npu_patch_meta()

dynamo_config.fake_tensor_cache_enabled = False

config.layout_optimization = False
config.size_asserts = False
config.fallback_random = True
config.optimize_scatter_upon_const_tensor = False

if anir_config.online_acc_comp:
    config.fx_graph_cache = False

aten = torch.ops.aten

## Override original dynamo device interface in torch_npu
if os.getenv('TORCHINDUCTOR_USE_AKG', '0') == '1':
    try:
        import akg
        import torch_mlir
        register_backend_for_device("npu", AkgScheduling, NpuMlirWrapperCodeGen)
    except:
        logger.warning(f"akg not found, fallback to torch-mlir for compilation.")
        register_backend_for_device("npu", NpuMlirScheduling, NpuMlirWrapperCodeGen)
else:
    register_backend_for_device("npu", NpuMlirScheduling, NpuMlirWrapperCodeGen)


class NewNpuInterface(NpuInterface):

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        # npu has no concept of cc. triton-npu compiler depends on subarch instead
        return torch.npu.get_device_name(device)

register_interface_for_device("npu", NewNpuInterface)

# recover from torch_npu._inductor patches to source code
def src_call(self, model_, inputs_):
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(model_, inputs_, config_patches=self.config)

_TorchCompileInductorWrapper.__call__ = src_call

## npu patch
from ..npu import npu_decomp
from torch._C import DispatchKey
from torch._prims_common.wrappers import out_wrapper

def disable_implicit_decomposition():
    '''
    Since torch official will implicitly decompose some aten ops,
    disable some ops here to avoid poor performance after decompose.
    '''
    disable_aten_ops = [
        'aten.upsample_nearest1d.vec', 'aten.upsample_nearest1d.default',
        'aten.upsample_nearest2d.vec', 'aten.upsample_nearest2d.default',
        'aten.upsample_nearest3d.vec', 'aten.upsample_nearest3d.default',
        'aten.upsample_bilinear2d.vec', 'aten.upsample_bilinear2d.default',
    ]
    for op_override in decomposition_table.keys():
        if str(op_override) in disable_aten_ops:
            if DispatchKey.Autograd in op_override.py_kernels:
                op_override.py_kernels.pop(DispatchKey.Autograd)
            if DispatchKey.CompositeImplicitAutograd in op_override.py_kernels:
                op_override.py_kernels.pop(DispatchKey.CompositeImplicitAutograd)


def _patch_run_node(tracer, node, args, kwargs, nnmodule):
    op = node.op

    with set_current_node(node):

        def make_error_message(e):
            return f"Failed running {op} {node.target}(*{args}, **{kwargs}):\n" + str(e)

        try:
            if op == "call_function":
                # patch start 
                if 'npu.npu_fusion_attention' in str(node.target):
                    if 'actual_seq_qlen' in kwargs:
                        kwargs['actual_seq_qlen'] = list(kwargs['actual_seq_qlen'])
                    if 'actual_seq_kvlen' in kwargs:
                        kwargs['actual_seq_kvlen'] = list(kwargs['actual_seq_kvlen'])
                # patch end
                return node.target(*args, **kwargs)
            elif op == "call_method":
                return getattr(args[0], node.target)(*args[1:], **kwargs)
            elif op == "call_module":
                if nnmodule is None:
                    raise RuntimeError(
                        f"Module {node.target} not found in the current module"
                    )
                return nnmodule(*args, **kwargs)
            elif op == "get_attr":
                return tracer.output_graph.get_submodule(node.target)
            elif op == "placeholder":
                if "example_value" not in node.meta:
                    raise RuntimeError(
                        f"placeholder {node.target} has no example value"
                    )
                return node.meta["example_value"]

        except (NotImplementedError, UnsupportedFakeTensorException) as e:
            # NB: mimic how wrap_fake_exception does it
            from torch._dynamo.exc import unimplemented

            unimplemented(make_error_message(e), from_exc=e)
        except Exception as e:
            raise RuntimeError(make_error_message(e)).with_traceback(
                e.__traceback__
            ) from e

    raise AssertionError(op)


def _register_npu_inductor_fallbacks_operation():
    from ..npu import inductor_patch

_register_npu_inductor_fallbacks_operation()
disable_implicit_decomposition()
torch._dynamo.utils.run_node = _patch_run_node

def wrap_compiler(fn):
    @functools.wraps(fn)
    def npu_compiler(gm: torch.fx.GraphModule, example_inputs, *args, **kwargs):
        npu_optimize_fx_graph(gm)
        return fn(gm, example_inputs, *args, **kwargs)
    return npu_compiler

def wrap_aot_autograd(fn):
    @functools.wraps(fn)
    def npu_aot_autograd(*args, **kwargs):
        _register_npu_inductor_fallbacks()
        def wrap_compiler_by_key(name):
            if name in kwargs:
                kwargs[name] = wrap_compiler(kwargs[name])
        wrap_compiler_by_key('fw_compiler')
        wrap_compiler_by_key('bw_compiler')
        wrap_compiler_by_key('inference_compiler')
        return fn(*args, **kwargs)
    return npu_aot_autograd

AotAutograd.__call__ = wrap_aot_autograd(AotAutograd.__call__)

# recompute last usage for inductor scheduler 

def used_or_aliased_buffer_names(node) -> Set[str]:
    used_names: OrderedSet[str] = OrderedSet()
    if isinstance(node, (SchedulerNode, FusedSchedulerNode)) and not isinstance(node, ForeachKernelSchedulerNode):
        snodes = [node] if isinstance(node, SchedulerNode) else node.snodes
        for snode in snodes:
            traced_graph = snode.node.data.traced_graph
            used_names = used_names.union(traced_graph.get_placeholder_names())
            used_names.add(snode.node.get_name())
    else:
        deps = [
            dep.name
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ]
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            if V.graph.name_to_buffer.get(dep):
                for alias in V.graph.name_to_buffer[dep].get_inputs_that_alias_output():
                    if alias not in used_names:
                        deps.append(alias)
    return used_names

def set_last_usage(
    node: BaseSchedulerNode, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]
):
    used_buffers = used_or_aliased_buffer_names(node)
    used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
    node.last_usage = used_buffers - future_used_buffers

def wrap_scheduler_codegen(fn):
    @functools.wraps(fn)
    def npu_sheduler_codegen(self, *args, **kwargs):
        future_used_buffers = set()
        for node_name in V.graph.get_output_names():
            future_used_buffers.add(node_name)
        for node in reversed(self.nodes):
            set_last_usage(node, future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)
        return fn(self, *args, **kwargs)
    return npu_sheduler_codegen

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
    name_to_dep_count: Counter[str] = collections.Counter()

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

def npu_can_fuse_vertical(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """
    Check if it is legal to fuse a consumer (node2) into a producer (node1).

    We can fuse them if all the reads of node2 either match
    corresponding writes in node1, or are written by nodes that can
    be scheduled before the fusion of node1 and node2.
    """
    node1_buf_names = node1.get_buffer_names()
    node1_op_names = node1.get_operation_names()
    computed_deps: OrderedSet[Dep] = OrderedSet()
    why = WhyNoFuse(node1, node2)

    for cd in node1.read_writes.writes:
        if not isinstance(cd, MemoryDep):
            continue
        for rd in node2.unmet_dependencies:
            if self.fusable_read_and_write(rd, cd):
                computed_deps.add(rd)

    for dep in node2.unmet_dependencies:
        if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
            computed_deps.add(dep)

    remaining_deps = OrderedSet(
        dep.name for dep in node2.unmet_dependencies - computed_deps
    )
    if remaining_deps & node1_buf_names:
        # MemoryDeps didn't match and read different locations of the same buffer.
        # Examples here include:
        #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
        #   - MemoryDep("foo", x) != StarDep("foo")
        why("memory deps did not match")
        return False
    for name in remaining_deps:
        if name not in self.name_to_buf:
            continue
        op_name = self.name_to_buf[name].defining_op.get_name()
        if node1_op_names & self.name_to_fused_node[op_name].ancestors:
            why("intermediate nodes between node1 & node2")
            return False

    return True

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

if anir_config.enable_graph_trace:
    Scheduler._codegen = wrap_scheduler_codegen(Scheduler._codegen)
    Scheduler.compute_ancestors = npu_compute_ancestors
    scheduler._prune_redundant_deps = _npu_prune_redundant_deps
    Scheduler.can_fuse_vertical = npu_can_fuse_vertical
    Scheduler._get_unmet_dep_nodes = _npu_get_unmet_dep_nodes

def wrap_avg_pool2d(fn):
    @functools.wraps(fn)
    def dynamo_avg_pool2d(input, *args, **kwargs):
        if input.dtype == torch.bfloat16:
            input = input.to(torch.float32)
            output = fn(input, *args, **kwargs)
            return output.to(torch.bfloat16)
        else:
            return fn(input, *args, **kwargs)
    return dynamo_avg_pool2d

F.avg_pool2d = wrap_avg_pool2d(F.avg_pool2d)

# patches for transfer_to_npu
def patch_transfer_to_npu():
    try:
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        from torch_npu.contrib.transfer_to_npu import (
            _replace_cuda_to_npu_in_list,
            device_kwargs_list,
            _replace_cuda_to_npu_in_kwargs,
        )

        def new_wrapper_cuda(module, method):    
            src_method = f"_src_{method}"
            if hasattr(getattr(module, method), '__wrapped__'):
                src_func = getattr(module, method).__wrapped__
            else:
                src_func = getattr(module, method)

            setattr(module, src_method, src_func)
            fn = getattr(module, src_method)
            
            def decorated(*args, **kwargs):
                replace_int = fn.__name__ in ['to', 'to_empty']
                if args:
                    args_new = list(args)
                    args = _replace_cuda_to_npu_in_list(args_new, replace_int)
                if kwargs:
                    for device_arg in device_kwargs_list:
                        device = kwargs.get(device_arg, None)
                        if device is not None:
                            _replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device)
                    device_ids = kwargs.get('device_ids', None)
                    if type(device_ids) == list:
                        device_ids = _replace_cuda_to_npu_in_list(device_ids, replace_int)
                return fn(*args, **kwargs)

            setattr(module, method, decorated)
            return decorated

        def new_device_wrapper(enter_fn, white_list):
            for fn_name in white_list:
                fn = getattr(enter_fn, fn_name, None)
                if fn:
                    new_wrapper_cuda(enter_fn, fn_name)

        transfer_to_npu._device_wrapper = new_device_wrapper
        transfer_to_npu._init()
    except:
        pass
