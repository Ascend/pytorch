from functools import wraps, partial
import torch

from torch_npu._C.dvm import (
    GraphSplitKernel,
    Kernel,
    NDObject,
    DynGraphSplitKernel,
    KernelType,
    KernelFlag,
    DynKernel,
)



def promote_bf16(fn):
    @wraps(fn)
    def wrapper(self, *args):
        need_cast_back = False

        def maybe_cast_arg(x):
            nonlocal need_cast_back
            if isinstance(x, NDObject) and self.get_dtype(x) == torch.bfloat16:
                need_cast_back = True
                return self.cast(x, torch.float32)
            return x

        new_args = tuple(maybe_cast_arg(a) for a in args)
        out = fn(self, *new_args)
        if need_cast_back:
            out = self.cast(out, torch.bfloat16)
        return out

    return wrapper


unsupported_bf16_ops = [
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
]


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


def create_kernel(
    ktype: str = "split",
    dyn_shape: bool = False,
):
    factory = KERNEL_FACTORY[(ktype, dyn_shape)]
    return factory()


def kernel(
    ktype: str = "split",
    dyn_shape: bool = False,
    dump_to_file="",
):
    """
    Kernel decorator factory.

    The decorated function `builder(kobj)` is responsible for constructing
    the kernel object (IR / graph / configuration).

    The returned object `fn` supports:
      - fn(*args, **kwargs)
      - fn.run(*args, **kwargs)   # exactly the same as fn(*args, **kwargs)

    Additional attributes:
      - fn.kobj       : the underlying kernel object
      - fn._dump_set  : internal set for dump de-duplication
    """

    def decorate(builder):
        # 1) Create kernel object
        kobj = create_kernel(ktype, dyn_shape)

        # 2) Let the builder populate the kernel object
        builder(kobj)

        # 3) Finalize kernel construction
        kobj.setup()

        @wraps(builder)
        def fn(*args, **kwargs):
            outputs = kobj(*args, **kwargs)

            if dump_to_file:
                parts = []
                for a in args:
                    if isinstance(a, torch.Tensor):
                        parts.append(str(a.shape))
                    elif isinstance(a, torch.SymInt):
                        parts.append(type(a).__name__)
                dump_id = ",".join(parts)
                if dump_id not in fn._dump_set:
                    fn._dump_set.add(dump_id)
                    with open(dump_to_file, "a") as fd:
                        fd.write(f"{kobj.dump()}\n{kobj.das()}\n")

            return outputs

        def run(*args, **kwargs):
            return kobj(*args)

        # 6) Expose execution aliases and internal state
        fn.run = run  # `run` is an alias of `__call__`
        fn.kobj = kobj  # expose the underlying kernel object
        fn._dump_set = set()  # dump de-duplication set

        return fn

    return decorate


if Kernel:
    for name in unsupported_bf16_ops:
        if hasattr(Kernel, name):
            setattr(Kernel, name, promote_bf16(getattr(Kernel, name)))
    Kernel.set_deterministic(torch.are_deterministic_algorithms_enabled())
