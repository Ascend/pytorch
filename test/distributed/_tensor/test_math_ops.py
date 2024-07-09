import itertools

import torch

from torch.distributed._tensor import distribute_tensor, DeviceMesh
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class DistMathOpsTest(DTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        tensor_to_sum = torch.randn(12, 8, 8).npu()
        mat1 = distribute_tensor(tensor_to_sum, device_mesh, shard_spec)
        keep_dim_or_not = [True, False, None]
        for dim in range(tensor_to_sum.ndim):
            for keep_dim in keep_dim_or_not:
                sum_args = (dim, keep_dim) if keep_dim is not None else (dim,)
                dim_sumed_tensor = tensor_to_sum.sum(*sum_args)
                dt_dim_sumed_tensor = mat1.sum(*sum_args).redistribute(
                    device_mesh, [Replicate()] * device_mesh.ndim
                )
                self.assertEqual(dt_dim_sumed_tensor.to_local(), dim_sumed_tensor)

        full_sumed_tensor = tensor_to_sum.sum()
        dt_sum = mat1.sum().redistribute(device_mesh, [Replicate()] * device_mesh.ndim)
        self.assertEqual(dt_sum.to_local(), full_sumed_tensor)


if __name__ == "__main__":
    run_tests()
