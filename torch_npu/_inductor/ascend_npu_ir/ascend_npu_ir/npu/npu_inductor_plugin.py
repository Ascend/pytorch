import atexit
import functools
import os

import torch
import torch.nn.functional as F

from torch._dynamo.utils import set_current_node, UnsupportedFakeTensorException

from torch._inductor.async_compile import shutdown_compile_workers
from torch_npu._inductor.compile_options import register_npu_backend

from ..npu.codegen.akg import AkgScheduling
from ..npu.codegen.mlir import NpuMlirScheduling
from ..npu.codegen.wrapper import NpuMlirWrapperCodeGen
from ..npu.npu_lowering import _register_npu_inductor_fallbacks
from torch_npu._inductor.dvm.inductor_patch.fx_graph import (
    npu_optimize_fx_graph,
    patch_dvm_mlir_post_grad_pass,
)
from ..npu.utils import logger

from .. import config as anir_config

from torch._decomp import (
    decomposition_table,
)

from . import npu_patch_deprecated  # noqa: F401
from .npu_meta import npu_patch_meta

# Fix Error: Exit earlier than child process.
atexit.register(shutdown_compile_workers)

# new npu meta registration.
npu_patch_meta()

from torch._inductor import config
config.layout_optimization = False
config.size_asserts = False
config.fallback_random = True
config.optimize_scatter_upon_const_tensor = False

if anir_config.online_acc_comp:
    config.fx_graph_cache = False

aten = torch.ops.aten

# Override original dynamo device interface in torch_npu


def register_mlir_codegen_backend() -> None:
    """Register MLIR scheduling/wrapper; call on each mlir/dvm backend switch."""
    if os.getenv('TORCHINDUCTOR_USE_AKG', '0') == '1':
        try:
            import akg  # noqa: F401
            import torch_mlir  # noqa: F401
            register_npu_backend(AkgScheduling, NpuMlirWrapperCodeGen)
        except ImportError:
            logger.warning("AKG not found; falling back to torch-mlir for compilation.")
            register_npu_backend(NpuMlirScheduling, NpuMlirWrapperCodeGen)
    else:
        register_npu_backend(NpuMlirScheduling, NpuMlirWrapperCodeGen)

try:
    from torch_npu.npu import device_count
except ImportError:
    from torch_npu.npu.utils import device_count
from torch._dynamo.device_interface import register_interface_for_device
from torch_npu.utils._dynamo_device import NpuInterface

class NewNpuInterface(NpuInterface):

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        # npu has no concept of cc. triton-npu compiler depends on subarch instead
        return torch.npu.get_device_name(device)

register_interface_for_device("npu", NewNpuInterface)

# npu patch
from torch._C import DispatchKey
from torch_npu._inductor.decomposition import _register_mlir_dvm_decompositions

_register_mlir_dvm_decompositions()

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
                    kwargs = dict(kwargs)
                    for name in ('actual_seq_qlen', 'actual_seq_kvlen'):
                        if isinstance(kwargs.get(name), (tuple, list)):
                            kwargs[name] = list(kwargs[name])
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


disable_implicit_decomposition()
torch._dynamo.utils.run_node = _patch_run_node


from torch._dynamo.backends.common import AotAutograd

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

patch_dvm_mlir_post_grad_pass(config)

# recompute last usage for inductor scheduler

if anir_config.enable_graph_trace:
    from torch._inductor import scheduler
    from torch._inductor.scheduler import Scheduler
    from torch_npu._inductor.dvm.inductor_patch.scheduler import (
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
                    if isinstance(device_ids, list):
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
    # [wtd#11] Use except Exception instead of bare except to avoid catching KeyboardInterrupt and system exceptions
    except Exception:
        pass
