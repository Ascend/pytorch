import inspect
from typing import Dict, List, Any
from functools import lru_cache
import torch
from torch._dynamo.utils import tensortype_to_dtype, proxy_args_kwargs
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
from torch._dynamo.variables.builder import wrap_fx_proxy
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
    ]:
        return TorchInGraphFunctionVariable(value, **kwargs)
    return cls.__new__raw(cls)


def SkipFunctionVariable__new__(cls, value, reason=None, **kwargs):
    if value in [
        torch.npu.stream,
        torch_npu.npu.stream,
        torch_npu.npu.utils.stream,
        torch.npu.current_stream,
        torch_npu.npu.current_stream,
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


def UserDefinedClassVariable_call_function(
    self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
    if self.value in [
        torch.npu.Stream,
        torch.npu.streams.Stream,
        torch.npu.streams.Event,
        torch_npu.npu.Stream,
        torch_npu.npu.streams.Stream,
        torch_npu.npu.streams.Event,
    ]:
        return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )
    else:
        return UserDefinedClassVariable.call_function_raw(self, tx, args, kwargs)


@lru_cache(None)
def patch_for_update_in_graph_functions():
    from torch._dynamo.trace_rules import load_object
    
    # In graph functions (including constant folding) that are C bindings
    torch_c_binding_in_graph_functions_npu = dict.fromkeys(
        [
            "torch_npu._C.is_autocast_enabled"
        ],
        TorchInGraphFunctionVariable,
    )
            
    # In graph functions (including constant folding) that are not C bindings
    torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
        [
            "torch.npu._utils._get_device_index",
            "torch_npu.npu._utils._get_device_index",
            "torch.npu.current_stream",
            "torch_npu.npu.current_stream",
            "torch.npu.is_available",
            "torch_npu.npu.is_available",
        ],
        TorchInGraphFunctionVariable,
    )

    torch_name_rule_map_npu = [
        torch_c_binding_in_graph_functions_npu,
        torch_non_c_binding_in_graph_functions_npu,
    ]
    
    constant_fold_functions_npu = dict.fromkeys(
        [
            "torch.npu.is_available",
            "torch_npu.npu.is_available",
            "torch_npu._C.is_autocast_enabled",
        ],
    )
    
    def load_object_npu(npu_dict):
        d: Dict[Any, VariableTracker] = dict()
        for k, v in npu_dict.items():
            obj = load_object(k)
            if obj is not None:
                if obj in d and d[obj] != v:
                    raise AssertionError(
                        f"Duplicate torch object {obj} with different rules: {v}, {d[obj]}"
                    )
                else:
                    d[obj] = v
        return d
    
    # update constant_fold_functions list for torch_npu function
    def update_constant_fold_functions_list():
        from torch._dynamo.variables.torch import constant_fold_functions
        
        d = list(load_object_npu(constant_fold_functions_npu).keys())
        constant_fold_functions.extend(d)
    
    # update get_torch_obj_rule_map dict for torch_npu function
    def update_get_torch_obj_rule_map_dict():
        from torch._dynamo.trace_rules import get_torch_obj_rule_map
        
        d: Dict[Any, VariableTracker] = dict()
        for m in torch_name_rule_map_npu:
            d.update(load_object_npu(m))
        get_torch_obj_rule_map().update(d)
        
    update_constant_fold_functions_list()
    update_get_torch_obj_rule_map_dict()
    

def add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFunctionVariable.__new__raw = SkipFunctionVariable.__new__
    SkipFunctionVariable.__new__ = SkipFunctionVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
    patch_dynamo_optimize()

    UserDefinedClassVariable.call_function_raw = UserDefinedClassVariable.call_function
    UserDefinedClassVariable.call_function = UserDefinedClassVariable_call_function
    patch_for_update_in_graph_functions()
