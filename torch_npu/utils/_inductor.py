from typing import Optional
import operator
from functools import reduce

import torch
from torch._prims_common import TensorLike
from torch._prims.rng_prims import register_rng_prim


aten = torch.ops.aten


def _max_unpoolnd_patch(
    self: TensorLike, indices: TensorLike, output_size: list[int], dim: int
):
    nc = reduce(operator.mul, self.shape[:-dim])
    hw = reduce(operator.mul, output_size)
    indices_nc_shape = [1] * self.ndim
    indices_nc_shape[:-dim] = self.shape[:-dim]
    indices_flat = (
        indices + aten.arange(nc, device=self.device).view(indices_nc_shape) * hw
    ).reshape(-1)

    output = self.new_zeros(list(self.shape[:-dim]) + list(output_size))
    return aten._unsafe_index_put(
        output.reshape(-1), [indices_flat], self.reshape(-1), accumulate=False
    ).view(output.shape)

torch._decomp.decompositions._max_unpoolnd = _max_unpoolnd_patch


def patch_philox_rand_offset():
    def get_philox_rand_offset_patch(shape):
        numel_scalar = 1
        for dim_size in shape:
            numel_scalar *= dim_size
        numel = torch.scalar_tensor(numel_scalar, dtype=torch.int64)

        return numel
    torch._prims.rng_prims.philox_rand_offset = get_philox_rand_offset_patch


def patch_register_philox_rand():
    rng_prims = torch._prims.rng_prims
    philox_rand_offset_meta = rng_prims.philox_rand_offset_meta
    philox_rand_offset = rng_prims.philox_rand_offset
    make_contiguous_strides_for = rng_prims.make_contiguous_strides_for
    _prims = torch._prims
    _device = rng_prims._device
    _dtype = rng_prims._dtype
    CUDARngStateHelper = rng_prims.CUDARngStateHelper


    def get_register_philox_rand_patch():
        name = "philox_rand"
        schema = "(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"  # noqa: B950
        
        
        def _philox_rand_meta(
            shape: torch.Size,
            seed: torch.Tensor,
            offset: torch.Tensor,
            stride: Optional[tuple[int, ...]],
            device: _device,
            dtype: _dtype,
        ):
            stride = make_contiguous_strides_for(shape)
            random_values = _prims.TensorMeta(
                shape=shape, strides=stride, dtype=dtype, device=device
            )
            offset = philox_rand_offset_meta(shape)
            return (random_values, offset)

        
        def _philox_rand(
            shape: torch.Size,
            seed: torch.Tensor,
            offset: torch.Tensor,
            stride: Optional[tuple[int, ...]],
            device: _device,
            dtype: _dtype,
        ):
            if device.type == "cpu":
                devices = []
            else:
                devices = [device]

            with torch.random.fork_rng(devices, device_type="npu"):                
                CUDARngStateHelper.set_torch_state_tensor(seed, offset)
                random_values = torch.rand(shape, device=device, dtype=dtype)

            return random_values, philox_rand_offset(shape)

        
        register_rng_prim(
            name=name,
            schema=schema,
            impl_aten=_philox_rand,
            impl_meta=_philox_rand_meta,
            doc="Philox based stateless rand operator",
            tags=(torch.Tag.nondeterministic_seeded,),
        )

    torch._prims.rng_prims.register_philox_rand = get_register_philox_rand_patch
    torch._prims.rng_prims.register_philox_rand()


def patch_register_run_and_save_rng_state_op():
    from torch._prims import rng_prims
    from torch._C import DispatchKey
    from torch._subclasses.fake_tensor import FakeTensorMode

    run_and_save_rng_state = getattr(
        rng_prims, "run_and_save_rng_state", None
    )

    if getattr(run_and_save_rng_state, "_npu_patched", False):
        return


    @run_and_save_rng_state.py_impl(DispatchKey.PrivateUse1)
    def impl_npu(op, *args, **kwargs):
        import torch_npu
        return torch_npu.npu.get_rng_state(), op(*args, **kwargs)


    backend_select_impl = run_and_save_rng_state.py_kernels.get(
        DispatchKey.BackendSelect, None
    )

    fake_tensor_mode_impl = run_and_save_rng_state.python_key_table.get(
        FakeTensorMode, None
    )


    def backend_select_with_npu(op, *args, **kwargs):
        from torch._prims.rng_prims import get_device

        device = get_device(args, kwargs)

        if device == "npu":
            return impl_npu(op, *args, **kwargs)

        return backend_select_impl(op, *args, **kwargs)


    def fake_tensor_mode_with_npu(mode, op, *args, **kwargs):
        from torch._prims.rng_prims import get_device

        device = get_device(args, kwargs)

        if device == "npu":
            with mode:
                return impl_npu(op, *args, **kwargs)

        return fake_tensor_mode_impl(mode, op, *args, **kwargs)

    run_and_save_rng_state.py_kernels[
        DispatchKey.BackendSelect
    ] = backend_select_with_npu
    run_and_save_rng_state.python_key_table[
        FakeTensorMode
    ] = fake_tensor_mode_with_npu


def patch_register_run_with_rng_state_op():
    from torch._prims import rng_prims
    from torch._C import DispatchKey
    from torch._subclasses.fake_tensor import FakeTensorMode

    run_with_rng_state = getattr(
        rng_prims, "run_with_rng_state", None
    )

    if getattr(run_with_rng_state, "_npu_patched", False):
        return


    @run_with_rng_state.py_impl(DispatchKey.PrivateUse1)
    def impl_npu(rng_state, op, *args, **kwargs):
        import torch_npu
        current_state = torch_npu.npu.get_rng_state()
        torch_npu.npu.set_rng_state(rng_state)
        try:
            out = op(*args, **kwargs)
        finally:
            torch_npu.npu.set_rng_state(current_state)
        return out


    backend_select_impl = run_with_rng_state.py_kernels.get(
        DispatchKey.BackendSelect, None
    )

    fake_tensor_mode_impl = run_with_rng_state.python_key_table.get(
        FakeTensorMode, None
    )


    def backend_select_with_npu(rng_state, op, *args, **kwargs):
        from torch._prims.rng_prims import get_device

        device = get_device(args, kwargs)

        if device == "npu":
            return impl_npu(rng_state, op, *args, **kwargs)

        return backend_select_impl(rng_state, op, *args, **kwargs)


    def fake_tensor_mode_with_npu(mode, rng_state, op, *args, **kwargs):
        from torch._prims.rng_prims import get_device

        device = get_device(args, kwargs)

        if device == "npu":
            with mode:
                return op(*args, **kwargs)

        return fake_tensor_mode_impl(mode, rng_state, op, *args, **kwargs)

    run_with_rng_state.py_kernels[
        DispatchKey.BackendSelect
    ] = backend_select_with_npu
    run_with_rng_state.python_key_table[
        FakeTensorMode
    ] = fake_tensor_mode_with_npu


def patch_rng_prims_device():
    from torch._prims import rng_prims
    src_get_device = rng_prims.get_device

    def new_patch_device(args, kwargs):
        device = src_get_device(args, kwargs)
        if device is None:
            devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
            if any(dev == "npu" for dev in devices):
                return "npu"
        return device
    rng_prims.get_device = new_patch_device


patch_register_run_and_save_rng_state_op()
patch_register_run_with_rng_state_op()
patch_philox_rand_offset()
patch_register_philox_rand()
patch_rng_prims_device()