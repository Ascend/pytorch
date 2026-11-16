"""DVM initialization and shared Inductor patch installation.

The loader calls _initialize_dvm() once before importing DVM fusion. Importing
this module captures upstream fallback references; shared lowering/IR/scheduler
imports stay inside the initializer, after prerequisite hooks are installed.
"""

import atexit
import functools

import torch
from torch._C import DispatchKey
from torch._decomp import decomposition_table
from torch._dynamo.backends.common import AotAutograd
from torch._inductor.async_compile import shutdown_compile_workers
from torch._inductor import config
from torch._dynamo.device_interface import register_interface_for_device
from torch_npu.utils._dynamo import run_once
from torch_npu.utils._dynamo_device import NpuInterface
from torch.distributed import distributed_c10d
from torch.distributed.distributed_c10d import _world, timedelta
from torch.library import Library

from . import config as dvm_config
from .inductor_patch.fx_graph import npu_optimize_fx_graph, patch_dvm_mlir_post_grad_pass
from .npu_lowering import _register_npu_inductor_fallbacks
from .decomp import _register_dvm_decompositions


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


def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    if sparse:
        raise RuntimeError("the current NPU does not yet support sparse tensor, when sparse is set to True")
    return torch.ops.aten.embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq)


def py_contiguous(x, memory_format=torch.contiguous_format):
    return x.clone(memory_format=memory_format)


def _patch_add_ephemeral_timeout_for_all_pgs(timeout: timedelta) -> None:
    """
    This API adds an ephemeral timeout extension for all PGs locally
    on one rank. The timeout gets reset when the first collective issued
    after API called finished.
    NOTE: We only support to set timeout for cuda backends for now.
    NOTE: While this feature
    provides flexibility in specific scenarios, it introduces statefulness
    to timeout setting. Therefore, it is advisable to use this API sparingly
    and consider alternative approaches, such as directly setting the timeout
    or utilizing a barrier collective (one can set any timeout to the barrier),
    whenever feasible.

    Args:
        timeout (timedelta): The delta of timeout to extend.

    Returns:
        None.
    """
    for pg in _world.pg_map.keys():
        devices = pg._device_types
        if torch.device("npu") in devices:
            backend = pg._get_backend(torch.device("npu"))


# Keep the dispatcher Library alive after the explicit installation returns.
_python_dispatcher_lib = None


@run_once
def _initialize_dvm():
    """Install process-wide DVM hooks; called internally by the backend loader."""
    global _python_dispatcher_lib

    # Migrated from ascend_npu_ir/ascend_npu_ir/config.py.
    config.allow_buffer_reuse = False

    # Migrated from ascend_npu_ir/ascend_npu_ir/npu/npu_patch_deprecated.py.
    _python_dispatcher_lib = Library("aten", "IMPL", "PythonDispatcher")
    _python_dispatcher_lib.impl("embedding_backward", embedding_backward)
    _python_dispatcher_lib.impl("contiguous", py_contiguous)
    distributed_c10d._add_ephemeral_timeout_for_all_pgs = _patch_add_ephemeral_timeout_for_all_pgs

    # Remaining installation sequence migrated from
    # ascend_npu_ir/ascend_npu_ir/npu/npu_inductor_plugin.py.
    atexit.register(shutdown_compile_workers)
    config.layout_optimization = False
    config.size_asserts = False
    config.fallback_random = True
    config.optimize_scatter_upon_const_tensor = False
    if dvm_config.online_acc_comp:
        config.fx_graph_cache = False

    register_interface_for_device("npu", NpuInterface)
    _register_dvm_decompositions()
    disable_implicit_decomposition()
    AotAutograd.__call__ = wrap_aot_autograd(AotAutograd.__call__)
    patch_dvm_mlir_post_grad_pass(config)

    from torch._inductor import scheduler
    from torch._inductor.scheduler import Scheduler
    from .inductor_patch.scheduler import (
        wrap_scheduler_codegen,
        npu_compute_ancestors,
        _npu_prune_redundant_deps,
        npu_can_fuse_vertical,
        _npu_get_unmet_dep_nodes,
    )

    Scheduler._codegen = wrap_scheduler_codegen(Scheduler._codegen)
    Scheduler.compute_ancestors = npu_compute_ancestors
    scheduler._prune_redundant_deps = _npu_prune_redundant_deps
    Scheduler.can_fuse_vertical = npu_can_fuse_vertical
    Scheduler._get_unmet_dep_nodes = _npu_get_unmet_dep_nodes

    from torch_npu._inductor.lowering_patch import apply_lowering_patch

    from torch_npu._inductor.dvm.inductor_patch import lowering as shared_lowering
    from torch_npu._inductor.dvm.inductor_patch.ir import apply_ir_patch
    from torch_npu._inductor.dvm.inductor_patch.scheduler import _patch_scheduler

    apply_ir_patch()
    shared_lowering.sum_fallback_predicate = None
    apply_lowering_patch(shared_lowering)
    _patch_scheduler()
