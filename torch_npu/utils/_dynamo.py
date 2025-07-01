import inspect
import sys
from typing import Dict, List

import torch
from torch._dynamo.utils import tensortype_to_dtype
from torch._dynamo.variables.torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import AutocastModeVariable
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch._dynamo.variables.functions import SkipFunctionVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._dynamo import optimize
from torch import _TorchCompileWrapper
import torch_npu
from torch_npu.dynamo import _get_global_npu_backend


class NPUTorchCtxManagerClassVariable(TorchCtxManagerClassVariable):
    def call_function(self, tx, args, kwargs):
        return NPUAutocastModeVariable.create(self.value, args, kwargs)      


class NPUAutocastModeVariable(AutocastModeVariable):
    @staticmethod
    def create(func, args, kwargs):
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()

        for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
            if key == "device_type" and func in [
                torch_npu.npu.amp.autocast,
            ]:
                arg = "npu" if func is torch_npu.npu.amp.autocast else "cpu"
            else:
                arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(arg.as_python_constant())
            else:
                target_values.append(arg)

        var = AutocastModeVariable(
            target_values, initial_values=None, **kwargs)
        return var


def UserDefinedClassVariable__new__(cls, value, **kwargs):
    if value in [
        torch.npu.amp.autocast,
        torch_npu.npu.amp.autocast,
        torch.npu.amp.autocast_mode.autocast,
        torch_npu.npu.amp.autocast_mode.autocast,
    ]:
        return NPUTorchCtxManagerClassVariable(value, **kwargs)
    elif value in [
        torch.npu.Stream,
        torch_npu.npu.Stream,
        torch.npu.streams.Stream,
        torch_npu.npu.streams.Stream,
        torch_npu.npu.BoolTensor,
        torch_npu.npu.ByteTensor,
        torch_npu.npu.CharTensor,
        torch_npu.npu.DoubleTensor,
        torch_npu.npu.FloatTensor,
        torch_npu.npu.HalfTensor,
        torch_npu.npu.IntTensor,
        torch_npu.npu.LongTensor,
        torch_npu.npu.ShortTensor,
        torch_npu.npu.BFloat16Tensor,
        torch.device,
    ]:
        return TorchInGraphFunctionVariable(value, **kwargs)
    return cls.__new__raw(cls)


def SkipFunctionVariable__new__(cls, value, reason=None, **kwargs):
    if value in [
        torch.npu.stream,
        torch_npu.npu.stream,
        torch_npu.npu.utils.stream,
    ]:
        return TorchInGraphFunctionVariable(value, **kwargs)
    return cls.__new__raw(cls)


def TensorVariable_call_method(self, tx, name, args, kwargs):
    if (
        name == 'type'
        and self.dtype is not None
        and len(args) == 0
        and isinstance(self.device, torch.device)
        and self.device.type == 'npu'
    ):
        tensortype = next(k for k, v in tensortype_to_dtype.items() if self.dtype in v)
        constant_result = ConstantVariable.create(f"torch.npu.{tensortype.__name__}")
        
        if len(args) == 1:
            return constant_result.getitem_const(args[0])
        elif args:
            return TupleVariable([constant_result.getitem_const(a) for a in args])
        return constant_result
    else:
        return TensorVariable.call_method_raw(self, tx, name, args, kwargs)


class _InductorNpuRegistry:
    _disabled_register = False
    _has_inited = False

    @classmethod
    def register_inductor_npu(cls):
        if cls.has_initialized() or cls._disabled_register:
            return
        from torch_npu import _inductor
        cls._has_inited = True
    
    @classmethod
    def disable_register(cls):
        cls._disabled_register = True

    @classmethod
    def enable_register(cls):
        cls._disabled_register = False
    
    @classmethod
    def has_initialized(cls):
        if cls._has_inited:
            return True
        # Maybe initialized by call `import torch_npu._inductor` manually.
        if 'torch_npu._inductor' in sys.modules:
            cls._has_inited = True
        return cls._has_inited


def is_inductor_npu_initialized():
    return _InductorNpuRegistry.has_initialized()


def disable_register_inductor_npu():
    _InductorNpuRegistry.disable_register()


def enable_register_inductor_npu():
    _InductorNpuRegistry.enable_register()


def register_inductor_npu():
    _InductorNpuRegistry.register_inductor_npu()


def patch_inductor_wrapper():
    from torch import _TorchCompileInductorWrapper
    src_call = _TorchCompileInductorWrapper.__call__
    
    def new_call(self, model_, inputs_):
        register_inductor_npu()
        return src_call(self, model_, inputs_)
    _TorchCompileInductorWrapper.__call__ = new_call


def patch_dynamo_optimize():
    src_optimize = optimize

    def npu_optimize(*args, **kwargs):
        backend = None
        if 'backend' in kwargs.keys():
            backend = kwargs['backend']
        elif len(args) == 1:
            backend = args[0]

        backend_name = None
        if isinstance(backend, str):
            backend_name = backend
        elif isinstance(backend, _TorchCompileWrapper):
            backend_name = backend.compiler_name

        if backend_name == 'npu':
            # Init torchair ahead of running model.
            _get_global_npu_backend()

        return src_optimize(*args, **kwargs)
    torch._dynamo.optimize = npu_optimize


def patch__aoti_compile_and_package_inner():
    from torch._inductor import _aoti_compile_and_package_inner
    src_fn = _aoti_compile_and_package_inner

    def wrap__aoti_compile_and_package_inner(*args, **kwargs):
        register_inductor_npu()
        return src_fn(*args, **kwargs)
    torch._inductor._aoti_compile_and_package_inner = wrap__aoti_compile_and_package_inner


def add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFunctionVariable.__new__raw = SkipFunctionVariable.__new__
    SkipFunctionVariable.__new__ = SkipFunctionVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
    patch_dynamo_optimize()
    patch__aoti_compile_and_package_inner()
    patch_inductor_wrapper()

