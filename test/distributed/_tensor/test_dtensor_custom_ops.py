import itertools
from typing import cast, List, Optional

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestDTensorCustomOps(DTensorTestBase):
    @property
    def world_size(self):
        # hard code world size to 4 as we need to test
        # at least with 2d mesh
        return 4

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dtensor_npu_bmmV2(self):
        npu_input1 = torch.randn(4, 12, 8).npu()
        npu_input2 = torch.randn(4, 8, 16).npu()
        output_sizes = []
        npu_input1.requires_grad = True
        npu_input2.requires_grad = True

        # local res
        local_res = torch_npu.npu_bmmV2(npu_input1, npu_input2, output_sizes)

        # distributed tensor
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()

        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        shard_spec_comb = list(itertools.product(placement_specs, placement_specs))
        for spec in shard_spec_comb:
            dt1 = distribute_tensor(npu_input1, device_mesh, [spec[0]])
            dt2 = distribute_tensor(npu_input2, device_mesh, [spec[0]])
            dist_res: DTensor = cast(DTensor, torch_npu.npu_bmmV2(dt1, dt2, output_sizes)).redistribute(
                device_mesh, [replica_spec]
            )
            self.assertEqual(dist_res.to_local(), local_res)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dtensor_fast_gelu(self):
        npu_input = torch.randn(4, 3).npu()
        local_res = torch_npu.fast_gelu(npu_input)

        # distributed tensor
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dist_tensor = distribute_tensor(npu_input, device_mesh, [Shard(0)])
        dist_res = torch_npu.fast_gelu(dist_tensor).redistribute(device_mesh, [Replicate()])

        self.assertEqual(npu_input.shape, dist_res.to_local().shape)
        self.assertEqual(local_res, dist_res.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dtensor_npu_fast_gelu(self):
        npu_input = torch.randn(4, 3).npu()
        local_res = torch_npu.npu_fast_gelu(npu_input)

        # distributed tensor
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dist_tensor = distribute_tensor(npu_input, device_mesh, [Shard(0)])
        dist_res = torch_npu.npu_fast_gelu(dist_tensor).redistribute(device_mesh, [Replicate()])
        self.assertEqual(local_res, dist_res.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dtensor_npu_dtype_cast(self):
        npu_input = torch.randn((2, 3), dtype=torch.float32).npu()
        dst_dtype = torch.float16
        local_result = torch_npu.npu_dtype_cast(npu_input, dst_dtype)

        # distributed tensor
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()

        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        for spec in placement_specs:
            dt_input = distribute_tensor(npu_input, device_mesh, [spec])
            dist_res: DTensor = cast(DTensor, torch_npu.npu_dtype_cast(dt_input, dst_dtype)).redistribute(
                device_mesh, [replica_spec]
            )
            self.assertEqual(dist_res.to_local().dtype, dst_dtype)
            self.assertEqual(dist_res.to_local(), local_result)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dtensor_npu_transpose(self):
        npu_input = torch.randn(5, 3, 6, 4).npu()
        perm = [1, 0, 2, 3]
        local_result = torch_npu.npu_transpose(npu_input, perm)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()

        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        for spec in placement_specs:
            dt_input = distribute_tensor(npu_input, device_mesh, [spec])
            dist_res: DTensor = cast(DTensor, torch_npu.npu_transpose(dt_input, perm)).redistribute(
                device_mesh, [replica_spec]
            )
            self.assertEqual(dist_res.to_local().shape, local_result.shape)


if __name__ == '__main__':
    run_tests()
