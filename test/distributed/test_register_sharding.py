import torch
from torch.distributed._tensor import distribute_tensor, Replicate
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms


class TestRegisterSharding(DTensorTestBase):

    def _run_matmul(self, shape1, shape2, device_mesh):
        x = torch.rand(shape1, device=self.device_type)
        dist_x = distribute_tensor(x, device_mesh, [Replicate()])
        y = torch.rand(shape2, device=self.device_type)
        dist_y = distribute_tensor(y, device_mesh, [Replicate()])

        local_out = torch.matmul(x, y)
        dist_out = torch.matmul(dist_x, dist_y)

        return local_out, dist_out

    def _run_matmul_backward(self, shape1, shape2, device_mesh):
        x = torch.rand(shape1, device=self.device_type, requires_grad=True)
        dist_x = distribute_tensor(x, device_mesh, [Replicate()])
        y = torch.rand(shape2, device=self.device_type, requires_grad=True)
        dist_y = distribute_tensor(y, device_mesh, [Replicate()])

        local_out = torch.matmul(x, y)
        dist_out = torch.matmul(dist_x, dist_y)

        local_out.backward(torch.ones_like(local_out))
        dist_out.backward(torch.ones_like(dist_out))

        return x, y, dist_x, dist_y

    def _run_npu_dtype_cast_backward(self, shape, device_mesh):
        x = torch.rand(
            shape, device=self.device_type, dtype=torch.float32, requires_grad=True
        )
        dist_x = distribute_tensor(x, device_mesh, [Replicate()])
        dst_dtype = torch.float16

        res = torch_npu.npu_dtype_cast(x, dst_dtype)
        res.sum().backward()

        dist_res = torch_npu.npu_dtype_cast(dist_x, dst_dtype)
        dist_res.sum().backward()

        self.assertEqual(res.dtype, dst_dtype)
        self.assertEqual(dist_res.dtype, dst_dtype)
        self.assertEqual(dist_x.shape, dist_x.grad.shape)

        return x, dist_x

    @with_comms
    def test_matmul_unshardable(self):
        device_mesh = self.build_device_mesh()

        # both unshardable
        local_out, dist_out = self._run_matmul((1, 2, 3), 3, device_mesh)
        self.assertTrue(dist_out.placements[0].is_replicate())
        self.assertEqual(dist_out.full_tensor(), local_out)

        # x is unshardable
        local_out, dist_out = self._run_matmul(3, (4, 3, 2), device_mesh)
        self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # y is unshardable
        local_out, dist_out = self._run_matmul((4, 3, 2), 2, device_mesh)
        self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        self.assertEqual(dist_out.full_tensor(), local_out)

    @with_comms
    def test_matmul_shardable(self):
        device_mesh = self.build_device_mesh()

        # n@n=1
        local_out, dist_out = self._run_matmul(4, 4, device_mesh)  # output_shape=(1)
        self.assertTrue(dist_out.placements[0].is_replicate())
        self.assertEqual(dist_out.full_tensor(), local_out)

        # n@...nm=...m
        # case2:
        local_out, dist_out = self._run_matmul(
            4, (3, 4, 2), device_mesh
        )  # output_shape=(3,2)
        self.assertTrue(dist_out.placements[0].is_partial())
        self.assertEqual(dist_out.full_tensor(), local_out)

        # case3:
        local_out, dist_out = self._run_matmul(
            4, (3, 4, 8), device_mesh
        )  # output_shape=(3,8)
        self.assertTrue(dist_out.placements[0].is_shard(dim=-1))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # case4:
        local_out, dist_out = self._run_matmul(
            4, (8, 4, 3), device_mesh
        )  # output_shape=(8,3)
        self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # ...nm@m=...n
        # case5:
        local_out, dist_out = self._run_matmul(
            (8, 2, 4), 4, device_mesh
        )  # output_shape=(8,2)
        self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # case6:
        local_out, dist_out = self._run_matmul(
            (2, 4, 8), 8, device_mesh
        )  # output_shape=(2,4)
        self.assertTrue(dist_out.placements[0].is_partial())
        self.assertEqual(dist_out.full_tensor(), local_out)

        #  ...nm@...mk=...nk(braodcast)
        # case7: max_size_in_1 and not max_dim1_index==len(shape1)-1:
        local_out, dist_out = self._run_matmul(
            (2, 8, 4), (2, 2, 4, 2), device_mesh
        )  # output_shape=(2,2,8,2)
        self.assertTrue(dist_out.placements[0].is_shard(dim=2))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # case8: max_size_in_2 and not max_dim1_index==len(shape2)-2:
        local_out, dist_out = self._run_matmul(
            (2, 4), (8, 2, 4, 2), device_mesh
        )  # output_shape=(8,2,2,2)
        self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        self.assertEqual(dist_out.full_tensor(), local_out)

        # case9: sharding the core dimension
        local_out, dist_out = self._run_matmul(
            (2, 2, 4), (2, 2, 4, 2), device_mesh
        )  # output_shape=(2,2,2,2)
        self.assertTrue(dist_out.placements[0].is_partial())
        self.assertEqual(dist_out.full_tensor(), local_out)

    @with_comms
    def test_matmul_backward(self):
        device_mesh = self.build_device_mesh()

        # unshardable
        x, y, dist_x, dist_y = self._run_matmul_backward((1, 2, 3), 3, device_mesh)
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        self.assertEqual(dist_y.grad.full_tensor(), y.grad)

        # n@...nm=...m
        x, y, dist_x, dist_y = self._run_matmul_backward(4, (3, 4, 2), device_mesh)
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        self.assertEqual(dist_y.grad.full_tensor(), y.grad)

        # ...nm@m=...n
        x, y, dist_x, dist_y = self._run_matmul_backward((8, 2, 4), 4, device_mesh)
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        self.assertEqual(dist_y.grad.full_tensor(), y.grad)

        #  ...nm@...mk
        x, y, dist_x, dist_y = self._run_matmul_backward(
            (2, 4), (8, 2, 4, 2), device_mesh
        )
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
        self.assertEqual(dist_y.grad.full_tensor(), y.grad)

    @with_comms
    def test_npu_dtype_cast_backward(self):
        device_mesh = self.build_device_mesh()

        # shard case
        x, dist_x = self._run_npu_dtype_cast_backward((1, 8, 3), device_mesh)

        self.assertTrue(dist_x.grad.placements[0].is_shard(dim=1))
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)

        # replicate case
        x, dist_x = self._run_npu_dtype_cast_backward((1, 2, 3), device_mesh)

        self.assertTrue(dist_x.grad.placements[0].is_replicate())
        self.assertEqual(dist_x.grad.full_tensor(), x.grad)


if __name__ == "__main__":
    run_tests()
