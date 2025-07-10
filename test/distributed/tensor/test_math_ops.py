import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestMathOps(DTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_rms_norm_forward(self):
        device_mesh = self.build_device_mesh()

        x = torch.randn((1, 128, 64), dtype=torch.float32).npu()
        gamma = torch.randn(64, dtype=torch.float32).npu()

        y, rstd = torch_npu.npu_rms_norm(x, gamma)

        dist_x = distribute_tensor(x, device_mesh, [Shard(1)])
        dist_gamma = distribute_tensor(gamma, device_mesh, [Replicate()])

        dist_y, dist_rstd = torch_npu.npu_rms_norm(dist_x, dist_gamma)

        self.assertEqual(dist_y.full_tensor(), y)
        self.assertEqual(dist_gamma.full_tensor(), gamma)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_rms_norm_backward(self):
        device_mesh = self.build_device_mesh()

        x = torch.randn((1, 128, 64), dtype=torch.float32).npu()
        gamma = torch.randn(64, dtype=torch.float32).npu()
        grad_y = torch.randn((1, 128, 64), dtype=torch.float32).npu()

        x = x.npu()
        gamma = gamma.npu()
        grad_y = grad_y.npu()
        x.requires_grad = True
        gamma.requires_grad = True

        y, rstd = torch_npu.npu_rms_norm(x, gamma, epsilon=1e-06)
        y.backward(grad_y)
        dx = x.grad
        dw = gamma.grad

        dist_x = distribute_tensor(x, device_mesh, [Shard(2)])
        dist_gamma = distribute_tensor(gamma, device_mesh, [Replicate()])

        dist_y, dist_rsts = torch_npu.npu_rms_norm(dist_x, dist_gamma, epsilon=1e-06)
        dist_grad_y = distribute_tensor(grad_y, device_mesh, dist_y.placements)
        dist_y.backward(dist_grad_y)
        dist_dx = dist_x.grad
        dist_dw = dist_gamma.grad

        self.assertEqual(dist_y.full_tensor(), y)
        self.assertEqual(dist_gamma.full_tensor(), gamma)

        self.assertEqual(dist_dx.full_tensor(), dx)
        self.assertEqual(dist_dw.full_tensor(), dw)


if __name__ == "__main__":
    run_tests()
