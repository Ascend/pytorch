from functools import wraps, partial
import sys
import torch

import torch_npu
from torch_npu._C.dvm import (
    NDObject,
    KernelType,
    KernelFlag,
    DataType,
    DynKernel,
    GraphSplitKernel,
    DynGraphSplitKernel,
    TorchKernel as Kernel,
)


debug_mode = False

bool_ = DataType.bool
float16 = DataType.float16
bfloat16 = DataType.bfloat16
float32 = DataType.float32
int32 = DataType.int32
int64 = DataType.int64
_FLAG_DYNAMIC = KernelFlag.kDynamic.value
_FLAG_UNIFY_WS = KernelFlag.kUnifyWS.value
_FLAG_SPECULATE = KernelFlag.kSpeculate.value

KERNEL_FACTORY = {
    ("mix", True): partial(DynKernel, KernelType.kMix, _FLAG_DYNAMIC),
    ("mix", False): partial(Kernel, KernelType.kMix, 0),
    ("split", True): DynGraphSplitKernel,
    ("split", False): GraphSplitKernel,
    ("spec", True): partial(
        DynKernel, KernelType.kVector, _FLAG_DYNAMIC | _FLAG_SPECULATE
    ),
    ("spec", False): partial(Kernel, KernelType.kVector, _FLAG_SPECULATE),
    ("vector", True): partial(DynKernel, KernelType.kVector, _FLAG_DYNAMIC),
    ("vector", False): partial(Kernel, KernelType.kVector, 0),
}


def kernel(
    ktype: str = "split",
    dyn_shape: bool = False,
):
    r"""kernel(ktype="split", dyn_shape=False)

    Return a decorator that builds and executes a DVM kernel.

    The decorated function ``builder(kobj)`` constructs the kernel object.
    The returned callable supports two execution styles:

    - ``fn(*args, **kwargs)``: takes input tensors only and returns output tensor(s).
    - ``fn.run(*args, **kwargs)``: takes input and output tensors, writes results
      into the provided output buffers, and returns ``None``.

    Args:
        ktype (str, optional): kernel type. Default: ``"split"``.
        dyn_shape (bool, optional): enable dynamic shapes. Default: ``False``.

    Returns:
        Callable: a callable kernel wrapper with a ``run`` method and ``kobj`` attribute.
    """

    def decorate(builder):
        kobj = KERNEL_FACTORY[(ktype, dyn_shape)]()
        kernel_name = getattr(builder, "__name__", "<unknown>")

        builder(kobj)
        kobj.setup()

        def _format_args(args):
            dump_parts = []
            arg_summaries = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    shape = tuple(a.shape)
                    dump_parts.append(str(a.shape))
                    arg_summaries.append(
                        f"Tensor(shape={shape}, dtype={a.dtype}, device={a.device})"
                    )
                elif isinstance(a, torch.SymInt) or isinstance(a, torch.SymFloat):
                    sym_name = type(a).__name__
                    dump_parts.append(sym_name)
                    arg_summaries.append(sym_name)
                else:
                    arg_summaries.append(type(a).__name__)
            return dump_parts, arg_summaries

        def _post_run(args):
            try:
                torch_npu.npu.synchronize()
            except Exception as exc:
                dump_text = kobj.dump()
                das_text = kobj.das()
                dump_parts, arg_summaries = _format_args(args)
                dump_id = ",".join(dump_parts)
                msg = [
                    "DVM debug sync failed.",
                    f"kernel_name={kernel_name}",
                    f"dump_id={dump_id}",
                    f"args={arg_summaries}",
                    f"dump={dump_text}",
                    f"das={das_text}",
                    f"error={type(exc).__name__}: {exc}",
                ]
                print("\n".join(msg), file=sys.stderr)
                raise

        @wraps(builder)
        def fn(*args, **kwargs):
            outputs = kobj(*args)
            if debug_mode:
                _post_run(args)
            return outputs

        def run(*args, **kwargs):
            kobj(*args)
            if debug_mode:
                _post_run(args)

        fn.run = run
        fn.kobj = kobj

        return fn

    return decorate


def _install_bf16_promote():
    unsupported_bf16_ops = (
        "sqrt",
        "abs",
        "log",
        "exp",
        "reciprocal",
        "logical_not",
        "round",
        "floor",
        "ceil",
        "trunc",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "logical_and",
        "logical_or",
        "select",
        "sum",
        "max",
        "min",
    )

    def _promote_bf16(op_fn):
        @wraps(op_fn)
        def wrapper(self, *args):
            need_cast_back = False

            def maybe_cast_arg(x):
                nonlocal need_cast_back
                if isinstance(x, NDObject) and x.dtype() == bfloat16:
                    need_cast_back = True
                    return self.cast(x, float32)
                return x

            new_args = tuple(maybe_cast_arg(a) for a in args)
            out = op_fn(self, *new_args)
            if need_cast_back:
                out = self.cast(out, bfloat16)
            return out

        return wrapper

    for name in unsupported_bf16_ops:
        op_fn = getattr(Kernel, name, None)
        if op_fn is not None:
            setattr(Kernel, name, _promote_bf16(op_fn))


_install_bf16_promote()
Kernel.set_deterministic(torch.are_deterministic_algorithms_enabled())
