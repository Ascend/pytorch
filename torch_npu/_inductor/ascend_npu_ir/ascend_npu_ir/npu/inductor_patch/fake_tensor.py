import torch
from torch._subclasses import fake_tensor
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
    Tensor,
    FakeTensor,
    FakeTensorMode,
    _StoragePointer,
    Sequence,
    PyTree,
    pytree,
    no_dispatch,
    is_sparse_any,
    Union,
    Set,
    T
)

def _npu_run_fallback_kernel(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    flat_args: Sequence[object],
    args_spec: PyTree,
    orig_not_implemented_exception: RuntimeError,
) -> FakeTensor:
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e: T) -> Union[T, Tensor]:
            if fake_mode.is_our_fake(e):
                out = torch.zeros(e.shape, dtype=e.dtype, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        r = func(*args, **kwargs)

    storages: Set[_StoragePointer] = set()

    for e in flat_args:
        if isinstance(e, Tensor):
            if not is_sparse_any(e):
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e: T) -> Union[T, FakeTensor]:
        if id(e) not in inp_impls and (
            isinstance(e, Tensor)
            and not is_sparse_any(e)
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    return pytree.tree_map(map_out, r)


def _patch_fake_tensor():
    fake_tensor.run_fallback_kernel = _npu_run_fallback_kernel