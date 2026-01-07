import itertools

import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_npu
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
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


class TestMathOps(NPUDTensorTestBase):
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


class TestConv2d(NPUDTensorTestBase):
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_replicate(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(3, 3, 224, 224, device="npu", requires_grad=True)
        weight_tensor = torch.randn(64, 3, 3, 3, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(0)])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = torch.randn(64, device="npu", requires_grad=True)
        d_bias = distribute_tensor(bias, mesh, [Replicate()])


        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, d_bias, stride, padding, dilation, groups)
        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)

        self.assertEqual(output_dtensor.full_tensor(), output_tensor)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_weight_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(3, 3, 224, 224, device="npu", requires_grad=True)
        weight_tensor = torch.randn(64, 3, 3, 3, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Shard(0)])

        bias = torch.randn(64, device="npu", requires_grad=True)
        d_bias = distribute_tensor(bias, mesh, [Shard(0)])


        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, d_bias, stride, padding, dilation, groups)
        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)

        self.assertEqual(output_dtensor.full_tensor(), output_tensor)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_input_shard1(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(8, 4, 224, 224, device="npu", requires_grad=True)
        weight_tensor = torch.randn(64, 4, 3, 3, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(1)])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Shard(1)])

        bias = torch.randn(64, device="npu", requires_grad=True)
        d_bias = distribute_tensor(bias, mesh, [Replicate()])

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, d_bias, stride, padding, dilation, groups)
        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)

        self.assertEqual(output_dtensor.full_tensor(), output_tensor)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_bias_is_None_replicate(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = None

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, bias, stride, padding, dilation, groups)
        self.assertEqual(output_dtensor.full_tensor(), output_tensor)
    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_bias_is_None_input_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(0)])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = None

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, bias, stride, padding, dilation, groups)
        self.assertEqual(output_dtensor.full_tensor(), output_tensor)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_bias_is_None_weight_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Shard(0)])

        bias = None

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1

        output_tensor = torch_npu.npu_conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        output_dtensor = torch_npu.npu_conv2d(input_dtensor, weight_dtensor, bias, stride, padding, dilation, groups)
        self.assertEqual(output_dtensor.full_tensor(), output_tensor)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_replicate(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = torch.randn(4, device="npu", requires_grad=True)
        d_bias = distribute_tensor(bias, mesh, [Replicate()])

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, True]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Replicate()])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)
        self.assertEqual(bias_dgrad.full_tensor(), bias_grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_bias_is_None_replicate(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = None

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, False]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Replicate()])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)
  
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_input_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(0)])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = torch.randn(4, device="npu", requires_grad=True)

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, True]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Shard(0)])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)
        self.assertEqual(bias_dgrad.full_tensor(), bias_grad)
  
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_bias_is_None_input_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 3, 3, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Shard(0)])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Replicate()])

        bias = None

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, False]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Shard(0)])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)
    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_weight_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Shard(0)])

        bias = torch.randn(4, device="npu", requires_grad=True)

        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, True]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Shard(1)])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)
        self.assertEqual(bias_dgrad.full_tensor(), bias_grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_conv2d_backward_bias_is_None_weight_shard0(self):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(4, 3, 28, 28, device="npu", requires_grad=True)
        weight_tensor = torch.randn(4, 3, 4, 4, device="npu", requires_grad=True)

        input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight_tensor, mesh, [Shard(0)])

        bias = None
        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        groups = 1
        output_mask = [True, True, False]

        output_tensor = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
        grad_output = torch.ones_like(output_tensor, device="npu")
        grad_output_dtensor = distribute_tensor(grad_output, mesh, [Shard(1)])

        input_grad, weight_grad, bias_grad = torch_npu.npu_conv2d_backward(input_tensor, grad_output, weight_tensor, stride, padding, dilation, groups, output_mask)
        input_dgrad, weight_dgrad, bias_dgrad = torch_npu.npu_conv2d_backward(input_dtensor, grad_output_dtensor, weight_dtensor, stride, padding, dilation, groups, output_mask)
        self.assertEqual(input_dgrad.full_tensor(), input_grad)
        self.assertEqual(weight_dgrad.full_tensor(), weight_grad)


class TestGroupedMatmulAdd(NPUDTensorTestBase):
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_grouped_matmul_add__replicate(self):
        mesh = self.build_device_mesh()

        x = torch.randn(8, 8, dtype=torch.float16, device="npu")
        weight = torch.randn(8, 4, dtype=torch.float16, device="npu")
        y = torch.randn(32, 4, dtype=torch.float, device="npu")
        group_list = torch.tensor([2, 4, 6, 8]).to(torch.int64).npu()
        x_dtensor = distribute_tensor(x, mesh, [Replicate()])
        weight_dtensor = distribute_tensor(weight, mesh, [Replicate()])
        y_dtensor = distribute_tensor(y, mesh, [Replicate()])
        group_list_dtensor = distribute_tensor(group_list, mesh, [Replicate()])
        transpose_x = True
        transpose_weight = False
        group_type = 2

        torch_npu.npu_grouped_matmul_add_(y, x, weight, group_list, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        torch_npu.npu_grouped_matmul_add_(y_dtensor, x_dtensor, weight_dtensor, group_list_dtensor, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        self.assertEqual(y_dtensor.full_tensor(), y)
    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_grouped_matmul_add__shard_D_weight(self):
        mesh = self.build_device_mesh()

        x = torch.randn(8, 8, dtype=torch.float16, device="npu")
        weight = torch.randn(8, 4, dtype=torch.float16, device="npu")
        y = torch.randn(32, 4, dtype=torch.float, device="npu")
        group_list = torch.tensor([2, 4, 6, 8]).to(torch.int64).npu()
        x_dtensor = distribute_tensor(x, mesh, [Shard(1)])
        weight_dtensor = distribute_tensor(weight, mesh, [Shard(1)])
        y_dtensor = distribute_tensor(y, mesh, [Shard(1)])
        group_list_dtensor = distribute_tensor(group_list, mesh, [Replicate()])
        transpose_x = True
        transpose_weight = False
        group_type = 2

        torch_npu.npu_grouped_matmul_add_(y, x, weight, group_list, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        torch_npu.npu_grouped_matmul_add_(y_dtensor, x_dtensor, weight_dtensor, group_list_dtensor, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        self.assertEqual(y_dtensor.full_tensor(), y)


class TestCrossEntropyLoss(NPUDTensorTestBase):
    def generate_data_cross_entropy_loss(self, N, C, input_strategy, target_strategy, weight_strategy=None):
        mesh = self.build_device_mesh()

        x = torch.randn(N, C, device="npu", requires_grad=True)
        target = torch.arange(0, N, device="npu")
        input_dtensor = distribute_tensor(x, mesh, input_strategy)
        target_dtensor = distribute_tensor(target, mesh, target_strategy)

        if weight_strategy:
            weight = torch.rand(C, device="npu")
            weight_dtensor = distribute_tensor(weight, mesh, weight_strategy)

            input_tuple = (x, target, weight, input_dtensor, target_dtensor, weight_dtensor, mesh)

            return input_tuple
        else:
            input_tuple = (x, target, input_dtensor, target_dtensor, mesh)
            
            return input_tuple


    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_replicate(self):
        x, target, input_dtensor, target_dtensor, _ = self.generate_data_cross_entropy_loss(8, 8, [Replicate()], [Replicate()])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="none")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="none")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_not_evenly_shardable(self):
        x, target, input_dtensor, target_dtensor, _ = self.generate_data_cross_entropy_loss(7, 8, [Shard(0)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_evenly_shardable(self):
        x, target, input_dtensor, target_dtensor, _ = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)


    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_evenly_shardable_weight(self):
        reductions = ["none", "sum"]
        x, target, weight, input_dtensor, target_dtensor, weight_dtensor, _ = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)], [Replicate()])

        for re in reductions:
            loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, weight, re)
            loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, weight_dtensor, re)

            self.assertEqual(loss_dtensor.full_tensor(), loss)
            self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_replicate_reduction_is_mean(self):
        x, target, input_dtensor, target_dtensor, _ = self.generate_data_cross_entropy_loss(8, 8, [Replicate()], [Replicate()])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        loss.backward()
        loss_dtensor.backward()
        self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)

    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_input_shard0_reduction_is_none(self):
        reductions = ["none", "sum", "mean"]
        x, target, input_dtensor, target_dtensor, mesh = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)])
        
        for re in reductions:
            loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction=re)
            loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction=re)
            if re == "none":
                grad = torch.randn(loss.size(), device="npu")
                grad_dtensor = distribute_tensor(grad, mesh, [Shard(0)])

                loss.backward(grad)
                loss_dtensor.backward(grad_dtensor)
                self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)
            else:
                loss.backward()
                loss_dtensor.backward()
                self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)
    
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_input_shard1_reduction_is_sum(self):
        x, target, input_dtensor, target_dtensor, _ = self.generate_data_cross_entropy_loss(8, 8, [Shard(1)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="sum")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="sum")
        
        loss.backward()
        loss_dtensor.backward()
        self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)


class TestRepeatInterleaveSelfInt(NPUDTensorTestBase):
    def generate_data_repeat_interleave_self_int(self, size, repeats_value, input_strategy):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(size, device="npu", requires_grad=True)
        input_dtensor = distribute_tensor(input_tensor, mesh, input_strategy)

        result = (input_tensor, repeats_value, input_dtensor, mesh)

        return result

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_replicate(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((5, 5), 3, [Replicate()])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard1(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((8, 8), 3, [Shard(1)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard0(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((8, 8), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_dim_is_None_shard0_is_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((8, 5), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
        output = torch.repeat_interleave(input_tensor, repeats_value)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard0_dim1_is_not_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((5, 5), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard1_dim1_is_not_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor, _ = self.generate_data_repeat_interleave_self_int((5, 8), 3, [Shard(1)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_shard0_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard1_replicate_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_dim_None(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
            output = torch.repeat_interleave(input_tensor, repeats_value)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard00_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(0)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard01_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(0)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Shard(1)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard10_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard11_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor, mesh = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, mesh, [Shard(1)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)


instantiate_parametrized_tests(TestMathOps)


if __name__ == "__main__":
    run_tests()
