import inspect
from typing import Dict, List

import torch
from torch._dynamo.utils import tensortype_to_dtype
from torch._dynamo.utils import get_fake_value
from torch._dynamo.exc import unimplemented
from torch._dynamo.variables.torch import TorchVariable
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import AutocastModeVariable, CUDAStreamContextVariable, CUDAStreamVariable
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch._dynamo.variables.misc import SkipFilesVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.variables.lists import TupleVariable
import torch_npu


def wrap_fx_proxy_cls_stream(
    target_cls, tx, proxy, example_value=None, ignore_subclass=False, **options
):
    example_value = get_fake_value(proxy.node, tx)
    proxy.node.meta['example_value'] = example_value
    return NPUStreamVariable(proxy, example_value, **options)


class NPUStreamVariable(CUDAStreamVariable):
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("npu stream")


class NPUStreamContextVariable(CUDAStreamContextVariable):
    @staticmethod
    def create(tx, target_value, **kwargs):
        current_stream = wrap_fx_proxy_cls_stream(
            NPUStreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                torch_npu.npu.current_stream,
                (None,),
                {},
            ),
        )
        return NPUStreamContextVariable(
            target_values=[target_value],
            initial_values=[current_stream],
            **kwargs,
        )

    def enter(self, tx):
        # NPU stream generated inside of traced function
        if self.target_values[0].as_proxy() is not None:
            tx.output.create_proxy(
                "call_function",
                torch_npu.npu.set_stream,
                (self.target_values[0].as_proxy(),),
                {},
            )
        # NPU stream passed from outside of traced function
        else:
            stream = self.target_values[0].value
            tx.output.create_proxy(
                "call_function",
                torch._C._npu_setStream,
                (stream.stream_id, stream.device_index, stream.device_type),
                {},
            )
        torch_npu.npu.set_stream(self.target_values[0].value)

    def exit(self, tx, *args):
        tx.output.create_proxy(
            "call_function",
            torch_npu.npu.set_stream,
            (self.initial_values[0].as_proxy(),),
            {},
        )
        torch_npu.npu.set_stream(self.initial_values[0].value)

    def module_name(self):
        return "torch_npu.npu"


class NPUTorchVariable(TorchVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if self.value in [
            torch.npu.amp.autocast,
            torch_npu.npu.amp.autocast,
            torch.npu.amp.autocast_mode.autocast,
            torch_npu.npu.amp.autocast_mode.autocast,
        ]:
            return NPUAutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in [
            torch.npu.stream,
            torch_npu.npu.stream,
        ]:
            return NPUStreamContextVariable.create(tx, args[0], **options)
        elif self.value in [
            torch.npu.Stream,
            torch_npu.npu.Stream,
            torch.npu.streams.Stream,
            torch_npu.npu.streams.Stream,
        ]:
            return wrap_fx_proxy_cls_stream(
                NPUStreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch_npu.npu.streams.Stream,
                    (),
                    {},
                ),
                **options
            )
        else:
            return super().call_function(tx, args, kwargs)


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
        torch.npu.Stream,
        torch_npu.npu.Stream,
        torch.npu.streams.Stream,
        torch_npu.npu.streams.Stream,
        torch.device,
        torch_npu.npu.BoolTensor,
        torch_npu.npu.ByteTensor,
        torch_npu.npu.CharTensor,
        torch_npu.npu.DoubleTensor,
        torch_npu.npu.FloatTensor,
        torch_npu.npu.HalfTensor,
        torch_npu.npu.IntTensor,
        torch_npu.npu.LongTensor,
        torch_npu.npu.ShortTensor,
    ]:
        return NPUTorchVariable(value, **kwargs)
    return cls.__new__raw(cls)


def SkipFilesVariable__new__(cls, value, **kwargs):
    if value in [
        torch.npu.stream,
        torch_npu.npu.stream,
    ]:
        return NPUTorchVariable(value, **kwargs)
    return cls.__new__raw(cls)


def TensorVariable_call_method(self, tx, name, args, kwargs):
    if (
        name == 'type'
        and self.dtype is not None
        and len(args) == 0
        and isinstance(self.device, torch.device)
        and self.device.type == 'npu'
    ):
        tensortype = next(
            k for k, v in tensortype_to_dtype.items() if self.dtype in v)
        constant_result = ConstantVariable(f"torch.npu.{tensortype.__name__}")

        if len(args) == 1:
            return constant_result.getitem_const(args[0])
        elif args:
            return TupleVariable([constant_result.getitem_const(a) for a in args])
        return constant_result
    else:
        return TensorVariable.call_method_raw(self, tx, name, args, kwargs)


def add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFilesVariable.__new__raw = SkipFilesVariable.__new__
    SkipFilesVariable.__new__ = SkipFilesVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
