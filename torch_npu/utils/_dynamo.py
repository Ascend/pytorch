import sys
import inspect
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

use_jit_script = False


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
    from typing import Any, Optional, Literal
    from torch import _TorchCompileInductorWrapper
    from torch.utils._config_module import Config, ConfigModule, _ConfigEntry
    src_call = _TorchCompileInductorWrapper.__call__
    src_init = _TorchCompileInductorWrapper.__init__
    src_get_config_copy = ConfigModule.get_config_copy
    
    def new_call(self, model_, inputs_):
        register_inductor_npu()
        return src_call(self, model_, inputs_)
        
    def new_get_config_copy(self) -> Dict[str, Any]:
        ori_dict = src_get_config_copy(self)
        NpuBackendType = Literal["default", "mlir", "dvm"]
        if "npu_backend" not in ori_dict:
            ori_dict["npu_backend"] = "default"
            self._config["npu_backend"] = _ConfigEntry(
                    Config(default="default", value_type=NpuBackendType)
            )
        return ori_dict
    
    def new_init(self, mode, options, dynamic):
        src_init(self, mode, options, dynamic)
        if self.config.get("npu_backend") == "mlir" or torch._inductor.config.npu_backend == "mlir":
            import os	 
            os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
            device_id = torch_npu.npu.current_device()
            torch_npu._C._recovery_all_npu_stream(device_id)
            try:
                import torch_mlir
                from torch_mlir import ir
            except ImportError as e:
                raise ImportError("torch_mlir is not installed, install it first.") from e
            from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu import (
                npu_inductor_plugin,
            )
    
    _TorchCompileInductorWrapper.__call__ = new_call
    _TorchCompileInductorWrapper.__init__ = new_init
    ConfigModule.get_config_copy = new_get_config_copy
    torch._inductor.config.get_config_copy()


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
            _get_global_npu_backend(backend_name)
        return src_optimize(*args, **kwargs)
    torch._dynamo.optimize = npu_optimize


def patch_base_schedulernode():
    from torch._inductor.scheduler import BaseSchedulerNode
    from torch._inductor.scheduler import ExternKernelSchedulerNode 

    original_get_read_write_buffer_accesses = BaseSchedulerNode.get_read_write_buffer_accesses

    def new_get_read_write_buffer_accesses(
        self_instance, include_reads: bool, include_writes: bool
    ) -> dict[str, int]:
        if isinstance(self_instance, ExternKernelSchedulerNode):
            return {}
        return original_get_read_write_buffer_accesses(
            self_instance, include_reads, include_writes
        )

    BaseSchedulerNode.get_read_write_buffer_accesses = new_get_read_write_buffer_accesses


def patch_user_defined_class_variable():
    import functools
    original_method = UserDefinedClassVariable._in_graph_classes
    
    @staticmethod
    @functools.lru_cache(None)
    def patched_in_graph_classes():
        result = original_method()
        result.add(torch.npu.Event)  
        result.add(torch.npu.Stream) 
        return result
    UserDefinedClassVariable._in_graph_classes = patched_in_graph_classes

    
def fake_record_stream(self, s):
    """
    let dynamo trace Tensor.record_stream as this emtpy function,
    and you can replace it later in your compile backend to an actual function
    """
    if isinstance(self, torch._subclasses.fake_tensor.FakeTensor):
        return
    raise RuntimeError("tensor.record_stream is not supported on torch.compile! "
                       "You should write a pass to replace torch.npu.fake_record_stream to an actual function in FX graph "
                       "before aot_autograd.")


def patch_record_stream():
    torch.npu.fake_record_stream = fake_record_stream

    def method_record_stream(self, s):
        tx = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
        return torch._dynamo.variables.TorchInGraphFunctionVariable(
            torch.npu.fake_record_stream
        ).call_function(tx, [self, s], {})
    
    torch._dynamo.variables.tensor.TensorVariable.method_record_stream = method_record_stream


def patch_variable_builder():
    original_warp = torch._dynamo.variables.builder.VariableBuilder._wrap

    def _patch_wrapper(self, value):
        if isinstance(value, torch.npu.Event):
            self.install_guards(torch._dynamo.guards.GuardBuilder.ID_MATCH)
            torch._dynamo.utils.store_user_object_weakref(value)
            event_proxy = self.tx.output.create_proxy(
                "call_function",
                torch._dynamo.utils.get_user_object_from_id,
                (id(value),),
                {},
            )
            torch._dynamo.utils.set_example_value(event_proxy.node, value)
            out = torch._dynamo.variables.ctx_manager.EventVariable(
                event_proxy,
                value,
                source=self.source,
            )
            return out
        return original_warp(self, value)

    torch._dynamo.variables.builder.VariableBuilder._wrap = _patch_wrapper


def patch_builtin_variable():
    origin_call_id = torch._dynamo.variables.builtin.BuiltinVariable.call_id

    def _wrap_call_id(self, tx, *args):
        if torch._dynamo.variables.builtin.istype(args[0], torch._dynamo.variables.ctx_manager.EventVariable):
            return torch._dynamo.variables.ConstantVariable.create(id(args[0].value))
        return origin_call_id(self, tx, *args)

    torch._dynamo.variables.builtin.BuiltinVariable.call_id = _wrap_call_id


def add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFunctionVariable.__new__raw = SkipFunctionVariable.__new__
    SkipFunctionVariable.__new__ = SkipFunctionVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
    patch_dynamo_optimize()
    patch_inductor_wrapper()
    patch_base_schedulernode()
    patch_user_defined_class_variable()
    patch_record_stream()
    patch_variable_builder()
    patch_builtin_variable()
