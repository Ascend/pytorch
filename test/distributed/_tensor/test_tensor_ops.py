import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import _Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorTestBase
)

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class DistTensorOpsTest(DTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_aten_contiguous(self):
        # this op not covered by dtensor_ops
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_inplace_op(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

        # test inplace op self and other dtensor with other specs
        # and make sure out spec not change
        shard_spec = [Shard(0)]
        partial_spec = [_Partial()]
        dt_to_inplace_add = distribute_tensor(input_tensor, mesh, shard_spec)
        partial_grad = DTensor.from_local(torch.randn(12, 3), mesh, partial_spec)
        res = dt_to_inplace_add.add_(partial_grad)
        self.assertTrue(res is dt_to_inplace_add)
        self.assertTrue(res.placements == tuple(shard_spec))

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_op_out_variant(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        sharded_dt_input = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = sharded_dt_input.clone() + 3
        sharded_dt_out = sharded_dt_input.clone()
        res = torch.add(sharded_dt_input, 3, out=sharded_dt_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is sharded_dt_out)
        self.assertEqual(sharded_dt_out.to_local(), expected_dt.to_local())

        # test op out variant with other spec and make sure out spec not change
        replica_spec = [Replicate()]
        replicate_out = distribute_tensor(input_tensor, mesh, replica_spec)
        expected_dt = replicate_out.clone() + 3
        res = torch.add(sharded_dt_input, 3, out=replicate_out)
        self.assertTrue(res is replicate_out)
        self.assertTrue(res.placements == tuple(replica_spec))
        self.assertEqual(replicate_out.to_local(), expected_dt.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_empty_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        empty_like_dt = torch.empty_like(dist_tensor)
        # empty is not deterministic, so we only check that the shard propagation worked
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_fill_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_full_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_ones_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_ones_like_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [_Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        self.assertEqual(dist_tensor.shape, (4, 8))

        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(dist_tensor.shape)
        self.assertEqual(
            ones_expected,
            ones_like_dt.redistribute(device_mesh, [Replicate()]).to_local(),
        )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_fill_inplace_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [_Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        self.assertEqual(dist_tensor.shape, (4, 8))

        torch.fill_(dist_tensor, 8)
        fill_expected = torch.full(
            dist_tensor.shape, 8 * self.world_size, dtype=input_tensor.dtype
        )
        self.assertEqual(fill_expected, dist_tensor.full_tensor())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_zeros_like_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [_Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        self.assertEqual(dist_tensor.shape, (4, 8))

        zeros_like_dt = torch.zeros_like(dist_tensor)
        zeros_expected = torch.zeros(dist_tensor.shape)
        self.assertEqual(
            zeros_expected,
            zeros_like_dt.redistribute(device_mesh, [Replicate()]).to_local(),
        )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_zero_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_zeros_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zeros_like(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_equal(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor_1 = torch.ones(4, 4)
        dist_tensor_1 = DTensor.from_local(input_tensor_1, device_mesh, shard_spec)

        # tensors are equal
        input_tensor_2 = torch.ones(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        self.assertTrue(eq_result)

        # tensors are different on some shards
        if self.rank == 0:
            input_tensor_2 = torch.ones(4, 4)
        else:
            input_tensor_2 = torch.randn(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # equal op all reduces each shard's local result
        self.assertFalse(eq_result)

    def _test_op(self, mesh, op_call, *args, **kwargs):
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            self.assertTrue(dtc.successful())
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(
                d_out.redistribute(mesh, [Replicate()] * mesh.ndim).to_local(),
                out,
            )


if __name__ == "__main__":
    run_tests()
