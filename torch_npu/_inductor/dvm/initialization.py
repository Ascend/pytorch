"""DVM initialization and shared Inductor patch installation.

The loader calls _initialize_dvm() once before importing DVM fusion. Importing
this module captures upstream fallback references; shared lowering/IR/scheduler
imports stay inside the initializer, after prerequisite hooks are installed.
"""

import atexit
import functools

from torch._dynamo.backends.common import AotAutograd
from torch._inductor.async_compile import shutdown_compile_workers
from torch._inductor import config
from torch._dynamo.device_interface import register_interface_for_device
from torch_npu.utils._dynamo import run_once
from torch_npu.utils._dynamo_device import NpuInterface

from ..decomposition import disable_implicit_decomposition
from . import config as dvm_config
from .inductor_patch.fx_graph import patch_dvm_mlir_post_grad_pass
from .npu_lowering import _register_npu_inductor_fallbacks
from .decomp import _register_dvm_decompositions


def wrap_aot_autograd(fn):
    @functools.wraps(fn)
    def npu_aot_autograd(*args, **kwargs):
        _register_npu_inductor_fallbacks()
        return fn(*args, **kwargs)
    return npu_aot_autograd


@run_once
def _initialize_dvm():
    """Install process-wide DVM hooks; called internally by the backend loader."""
    # Migrated from ascend_npu_ir/ascend_npu_ir/config.py.
    config.allow_buffer_reuse = False

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
