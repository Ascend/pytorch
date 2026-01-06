import itertools

import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import SupportedDevices


def get_shape_from_layout(batch: int, num_head: int, seq_length: int, dimension: int, layout: str):
    layout_map = {
        "B": batch,
        "N": num_head,
        "S": seq_length,
        "D": dimension,
        "1": 1,
    }
    shape = []
    for dim in layout:
        if dim in layout_map:
            shape.append(layout_map[dim])
        else:
            raise ValueError(f"Invalid layout character: {dim}")

    return tuple(shape)


class TestMathOps(DTensorTestBase):
    @property
    def world_size(self):
        device_count = torch.npu.device_count()
        device_num = 4
        if device_count > 1:
            device_num = min(device_num, device_count)
        return device_num

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
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

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
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

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_add_rms_norm_forward(self):
        device_mesh = self.build_device_mesh()

        x1 = torch.randn((1, 128, 64), dtype=torch.float32).npu()
        x2 = torch.randn((1, 128, 64), dtype=torch.float32).npu()
        gamma = torch.randn(64, dtype=torch.float32).npu()

        y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma)

        def test_placement_comb(placements1, placements2):
            dist_x1 = distribute_tensor(x1, device_mesh, placements1)
            dist_x2 = distribute_tensor(x2, device_mesh, placements2)
            dist_gamma = distribute_tensor(gamma, device_mesh, [Replicate()])
            dist_y, dist_rstd, dist_x = torch_npu.npu_add_rms_norm(dist_x1, dist_x2, dist_gamma)
            self.assertEqual(dist_y.full_tensor(), y)
            self.assertEqual(dist_rstd.full_tensor(), rstd)
            self.assertEqual(dist_x.full_tensor(), x)

        placement = [Shard(0), Shard(1), Shard(2), Replicate()]
        placement_combs = itertools.product(placement, placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "rotary_mode,input_layout,sin_cos_layout",
        [
            ("half", "BNSD", "11SD"),
            ("half", "BNSD", "B1SD"),
            ("half", "BNSD", "BNSD"),
            ("half", "BSND", "1S1D"),
            ("half", "BSND", "BS1D"),
            ("half", "BSND", "BSND"),
            ("half", "SBND", "S11D"),
            ("half", "SBND", "SB1D"),
            ("half", "SBND", "SBND"),
            ("interleave", "BNSD", "11SD"),
            ("interleave", "BSND", "1S1D"),
            ("interleave", "SBND", "S11D"),
        ]
    )
    def test_npu_rotary_mul_forward(self, rotary_mode, input_layout, sin_cos_layout):
        device_mesh = self.build_device_mesh()

        B = 8
        N = 8
        S = 64
        D = 32
        x_shape = get_shape_from_layout(B, N, S, D, input_layout)
        x = torch.randn(x_shape, dtype=torch.float32, device="npu")
        sin_cos_shape = get_shape_from_layout(B, N, S, D, sin_cos_layout)
        sin = torch.randn(sin_cos_shape, dtype=torch.float32, device="npu") * 2 - 1
        cos = torch.randn(sin_cos_shape, dtype=torch.float32, device="npu") * 2 - 1

        y = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode=rotary_mode)

        def test_placement_comb(x_placements, sin_placements, cos_placements):
            dist_x = distribute_tensor(x, device_mesh, x_placements)
            dist_sin = distribute_tensor(sin, device_mesh, sin_placements)
            dist_cos = distribute_tensor(cos, device_mesh, cos_placements)
            dist_y = torch_npu.npu_rotary_mul(dist_x, dist_cos, dist_sin, rotary_mode=rotary_mode)
            self.assertEqual(dist_y.full_tensor(), y)

        placements = [Shard(0), Shard(1), Shard(2), Replicate()]
        for placement in placements:
            if isinstance(placement, Shard) and sin_cos_shape[placement.dim] == 1:
                test_placement_comb([placement], [Replicate()], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "rotary_mode,input_layout,sin_cos_layout",
        [
            ("half", "BNSD", "11SD"),
            ("half", "BNSD", "B1SD"),
            ("half", "BNSD", "BNSD"),
            ("half", "BSND", "1S1D"),
            ("half", "BSND", "BS1D"),
            ("half", "BSND", "BSND"),
            ("half", "SBND", "S11D"),
            ("half", "SBND", "SB1D"),
            ("half", "SBND", "SBND"),
            ("interleave", "BNSD", "11SD"),
            ("interleave", "BSND", "1S1D"),
            ("interleave", "SBND", "S11D"),
        ]
    )
    def test_npu_rotary_mul_backward(self, rotary_mode, input_layout, sin_cos_layout):
        device_mesh = self.build_device_mesh()

        B = 8
        N = 8
        S = 64
        D = 32
        x_shape = get_shape_from_layout(B, N, S, D, input_layout)
        x = torch.randn(x_shape, dtype=torch.float32, device="npu", requires_grad=True)
        sin_cos_shape = get_shape_from_layout(B, N, S, D, sin_cos_layout)
        sin = torch.randn(sin_cos_shape, dtype=torch.float32, device="npu") * 2 - 1
        cos = torch.randn(sin_cos_shape, dtype=torch.float32, device="npu") * 2 - 1
        sin.requires_grad = True
        cos.requires_grad = True


        y = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode=rotary_mode)
        grad_y = torch.ones_like(y, dtype=torch.float32, device="npu")
        y.backward(grad_y)

        def test_placement_comb(x_placements, sin_placements, cos_placements):
            dist_x = distribute_tensor(x, device_mesh, x_placements)
            dist_sin = distribute_tensor(sin, device_mesh, sin_placements)
            dist_cos = distribute_tensor(cos, device_mesh, cos_placements)
            dist_y = torch_npu.npu_rotary_mul(dist_x, dist_cos, dist_sin, rotary_mode=rotary_mode)
            dist_grad_y = distribute_tensor(grad_y, device_mesh, dist_y.placements)
            dist_y.backward(dist_grad_y)
            self.assertEqual(dist_y.full_tensor(), y)
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)
            self.assertEqual(dist_sin.grad.full_tensor(), sin.grad)
            self.assertEqual(dist_cos.grad.full_tensor(), cos.grad)

        placements = [Shard(0), Shard(1), Shard(2), Replicate()]
        for placement in placements:
            if isinstance(placement, Shard) and sin_cos_shape[placement.dim] == 1:
                test_placement_comb([placement], [Replicate()], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement])


instantiate_parametrized_tests(TestMathOps)


if __name__ == "__main__":
    run_tests()
