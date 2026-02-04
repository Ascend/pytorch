import operator
from functools import reduce

import torch
from torch._prims_common import TensorLike
from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides


class NPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch_npu._C import _npu_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch_npu.npu.set_device({device_idx})"

    def synchronize(self):
        return "torch_npu.npu.synchronize()"

    def device_guard(self, device_idx):
        return f"torch_npu.npu._DeviceGuard({device_idx})"


def _inductor_register_device_op_overrides():
    register_device_op_overrides('npu', NPUDeviceOpOverrides())


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
