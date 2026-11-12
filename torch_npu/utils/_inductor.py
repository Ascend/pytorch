import contextlib
import operator
from functools import reduce
from typing import Optional

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


def patch_rng_state_helper_for_npu():
    from torch._prims_common import CUDARngStateHelper

    helper = CUDARngStateHelper
    if getattr(helper, "_npu_patched", False):
        return

    original_get_torch_state_as_tuple = helper.get_torch_state_as_tuple
    original_set_torch_state_tensor = helper.set_torch_state_tensor
    original_set_new_offset = helper.set_new_offset

    def _should_use_npu_rng_state():
        return hasattr(torch, "npu") and torch.npu.is_available()

    def align_npu_offset(offset):
        return ((int(offset) + 3) // 4) * 4

    def get_torch_state_as_tuple(fake_mode=contextlib.nullcontext()):
        if not _should_use_npu_rng_state():
            return original_get_torch_state_as_tuple(fake_mode)

        with fake_mode:
            seed = torch.tensor(torch.npu.initial_seed())
            offset = torch.tensor(torch.npu._get_rng_state_offset())
            return seed, offset

    def set_torch_state_tensor(seed, offset):
        if not _should_use_npu_rng_state():
            return original_set_torch_state_tensor(seed, offset)

        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        torch.npu.set_rng_state(new_state)

    def set_new_offset(relative_offset):
        if not _should_use_npu_rng_state():
            return original_set_new_offset(relative_offset)

        torch.npu._set_rng_state_offset(align_npu_offset(relative_offset.item()))

    helper.get_torch_state_as_tuple = staticmethod(get_torch_state_as_tuple)
    helper.set_torch_state_tensor = staticmethod(set_torch_state_tensor)
    helper.set_new_offset = staticmethod(set_new_offset)
    helper._npu_patched = True


def patch_functionalized_rng_decomp_for_npu():
    from torch._decomp import decompositions_for_rng

    if getattr(decompositions_for_rng.throw_on_non_cuda, "_npu_patched", False):
        return

    original_throw_on_non_cuda = decompositions_for_rng.throw_on_non_cuda

    def throw_on_non_cuda(device):
        if device.type == "npu":
            return
        return original_throw_on_non_cuda(device)

    throw_on_non_cuda._npu_patched = True
    decompositions_for_rng.throw_on_non_cuda = throw_on_non_cuda


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


def patch_register_graphsafe_run_with_rng_state_op():
    from torch._prims import rng_prims
    from torch._functorch._aot_autograd.utils import (
        register_graphsafe_rng_device_type,
    )

    graphsafe_run_with_rng_state = getattr(
        rng_prims, "graphsafe_run_with_rng_state", None
    )
    if graphsafe_run_with_rng_state is None or getattr(
        graphsafe_run_with_rng_state, "_npu_patched", False
    ):
        return

    register_graphsafe_rng_device_type("npu")
    graphsafe_run_with_rng_state._npu_patched = True


def patch_functionalized_rng_wrapper_for_npu_graphsafe():
    from torch._functorch._aot_autograd import runtime_wrappers
    from torch._prims_common import CUDARngStateHelper
    from torch._functorch._aot_autograd.runtime_wrappers import config

    wrapper_cls = runtime_wrappers.FunctionalizedRngRuntimeWrapper
    if getattr(wrapper_cls, "_npu_graphsafe_rng_patched", False):
        return

    orig_pre_compile = wrapper_cls.pre_compile
    orig_post_compile = wrapper_cls.post_compile

    def has_npu_graphsafe_rng(metadata):
        device = getattr(metadata, "graphsafe_rng_device", None)
        return (
            getattr(metadata, "num_graphsafe_rng_states", 0) > 0
            and device is not None
            and device.type == "npu"
        )

    def pre_compile(self, fw_module, flat_args, aot_config, *, fw_metadata):
        if not config.functionalize_rng_ops or not has_npu_graphsafe_rng(fw_metadata):
            return orig_pre_compile(
                self, fw_module, flat_args, aot_config, fw_metadata=fw_metadata
            )

        fake_mode = runtime_wrappers.detect_fake_mode()
        if fake_mode is None:
            raise AssertionError(
                "fake_mode must not be None when functionalize_rng_ops is True"
            )
        seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        insert_at = len(flat_args) - fw_metadata.num_graphsafe_rng_states
        flat_args[insert_at:insert_at] = [seed, offset]

    def post_compile(self, compiled_fn, aot_config, *, runtime_metadata):
        if not (
            runtime_metadata.is_rng_op_functionalized
            and has_npu_graphsafe_rng(runtime_metadata)
        ):
            return orig_post_compile(
                self, compiled_fn, aot_config, runtime_metadata=runtime_metadata
            )

        num_graphsafe_rng_states = runtime_metadata.num_graphsafe_rng_states

        def compiled_fn_with_npu_rng_order(runtime_args):
            seed_offset = runtime_args[-2:]
            del runtime_args[-2:]
            insert_at = len(runtime_args) - num_graphsafe_rng_states
            runtime_args[insert_at:insert_at] = seed_offset
            return compiled_fn(runtime_args)

        compiled_fn_with_npu_rng_order._boxed_call = getattr(
            compiled_fn, "_boxed_call", True
        )
        return orig_post_compile(
            self,
            compiled_fn_with_npu_rng_order,
            aot_config,
            runtime_metadata=runtime_metadata,
        )

    wrapper_cls.pre_compile = pre_compile
    wrapper_cls.post_compile = post_compile
    wrapper_cls._npu_graphsafe_rng_patched = True

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
patch_register_graphsafe_run_with_rng_state_op()
patch_rng_state_helper_for_npu()
patch_functionalized_rng_decomp_for_npu()
patch_philox_rand_offset()
patch_register_philox_rand()
patch_rng_prims_device()
