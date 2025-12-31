import numpy as np
import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase


class TestRegisterSharding(NPUDTensorTestBase):
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

        self.assertEqual(dist_x.grad.full_tensor(), x.grad)

        # replicate case
        x, dist_x = self._run_npu_dtype_cast_backward((1, 2, 3), device_mesh)

        self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    def test_npu_fussion_attention_forward(self):
        scale = 0.08838
        query = self.trans_BNSD2BSH(torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32))
        key = self.trans_BNSD2BSH(torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32))
        value = self.trans_BNSD2BSH(torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32))
        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention(
            query, key, value, head_num=32, input_layout="BSH", scale=scale)

        device_mesh = self.build_device_mesh()
        dist_query = distribute_tensor(query, device_mesh, [Replicate()])
        dist_key = distribute_tensor(key, device_mesh, [Replicate()])
        dist_value = distribute_tensor(value, device_mesh, [Replicate()])
        dist_attention_score, dist_softmax_max, dist_softmax_sum, dist_softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention(
            dist_query, dist_key, dist_value, head_num=32, input_layout="BSH", scale=scale)

        self.assertEqual(dist_attention_score.full_tensor(), attention_score)
        self.assertEqual(dist_softmax_max.full_tensor(), softmax_max)
        self.assertEqual(dist_softmax_sum.full_tensor(), softmax_sum)
        self.assertEqual(dist_softmax_out.full_tensor(), softmax_out)

    @with_comms
    def test_npu_fussion_attention_grad(self):
        scale = 0.08838
        query = torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32)
        key = torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32)
        value = torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32)
        dy = torch.randn(1, 32, 128, 128, device=self.device_type, dtype=torch.float32)

        # get attention_in
        query = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res, x_max, x_sum = self.tsoftmax(query.to(torch.float32))
        attention_in = torch.matmul(softmax_res, value)

        query = self.trans_BNSD2BSH(query)
        key = self.trans_BNSD2BSH(key)
        value = self.trans_BNSD2BSH(value)
        dy = self.trans_BNSD2BSH(dy)

        x_max = x_max.expand(1, 32, 128, 8).npu()
        x_sum = x_sum.expand(1, 32, 128, 8).npu()
        out = self.trans_BNSD2BSH(attention_in)

        dq, dk, dv, dpse = torch_npu.npu_fusion_attention_grad(
            query, key, value, dy, head_num=32, input_layout="BSH",
            softmax_max=x_max, softmax_sum=x_sum, attention_in=attention_in, scale_value=scale)

        device_mesh = self.build_device_mesh()
        dist_query = distribute_tensor(query, device_mesh, [Replicate()])
        dist_key = distribute_tensor(key, device_mesh, [Replicate()])
        dist_value = distribute_tensor(value, device_mesh, [Replicate()])
        dist_dy = distribute_tensor(dy, device_mesh, [Replicate()])
        dist_xmax = distribute_tensor(x_max, device_mesh, [Replicate()])
        dist_xsum = distribute_tensor(x_sum, device_mesh, [Replicate()])
        dist_attention_in = distribute_tensor(out, device_mesh, [Replicate()])
        dist_dq, dist_dk, dist_dv, dist_dpse = torch_npu.npu_fusion_attention_grad(
            dist_query, dist_key, dist_value, dist_dy, head_num=32, input_layout="BSH",
            softmax_max=dist_xmax, softmax_sum=dist_xsum, attention_in=dist_attention_in, scale_value=scale)

        self.assertEqual(dist_dq.full_tensor(), dq)
        self.assertEqual(dist_dk.full_tensor(), dk)
        self.assertEqual(dist_dv.full_tensor(), dv)
        if dist_dpse is not None:
            self.assertEqual(dist_dpse.full_tensor(), dpse)
        else:
            self.assertEqual(dist_dpse, dpse)

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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
    
    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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
  
    @skipIfUnsupportMultiNPU(4)
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
  
    @skipIfUnsupportMultiNPU(4)
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
    
    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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
   
    @skipIfUnsupportMultiNPU(4)
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
        group_type = 1

        torch_npu.npu_grouped_matmul_add_(y, x, weight, group_list, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        torch_npu.npu_grouped_matmul_add_(y_dtensor, x_dtensor, weight_dtensor, group_list_dtensor, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        self.assertEqual(y_dtensor.full_tensor(), y)
    
    @skipIfUnsupportMultiNPU(4)
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
        group_type = 1

        torch_npu.npu_grouped_matmul_add_(y, x, weight, group_list, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        torch_npu.npu_grouped_matmul_add_(y_dtensor, x_dtensor, weight_dtensor, group_list_dtensor, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)
        self.assertEqual(y_dtensor.full_tensor(), y)
   
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_grouped_matmul_add__shard_D_x(self):
        mesh = self.build_device_mesh()

        x = torch.randn(8, 8, dtype=torch.float16, device="npu")
        weight = torch.randn(8, 4, dtype=torch.float16, device="npu")
        y = torch.randn(32, 4, dtype=torch.float, device="npu")
        group_list = torch.tensor([2, 4, 6, 8]).to(torch.int64).npu()
        x_dtensor = distribute_tensor(x, mesh, [Shard(0)])
        weight_dtensor = distribute_tensor(weight, mesh, [Shard(1)])
        y_dtensor = distribute_tensor(y, mesh, [Shard(0)])
        group_list_dtensor = distribute_tensor(group_list, mesh, [Replicate()])
        transpose_x = True
        transpose_weight = False
        group_type = 1

        with self.assertRaises(RuntimeError) as cm:
            torch_npu.npu_grouped_matmul_add_(y_dtensor, x_dtensor, weight_dtensor, group_list_dtensor, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)

        err = cm.exception
        self.assertIn("Sharding propagation failed for Op", str(err))

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_replicate(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Replicate()])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Replicate()])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Replicate()])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Replicate()])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard00(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(0)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(0)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(0)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(0)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard01(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(0)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(0)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(0)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(1)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard10(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(1)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(1)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(1)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(0)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard11(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(1)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(1)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(1)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(1)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @with_comms
    def generate_data_cross_entropy_loss(self, N, C, input_strategy, target_strategy, weight_strategy=None):
        mesh = self.build_device_mesh()

        x = torch.randn(N, C, device="npu", requires_grad=True)
        target = torch.arange(0, N, device="npu")
        input_dtensor = distribute_tensor(x, mesh, input_strategy)
        target_dtensor = distribute_tensor(target, mesh, target_strategy)

        if weight_strategy:
            weight = torch.rand(C, device="npu")
            weight_dtensor = distribute_tensor(weight, mesh, weight_strategy)

            input_tuple = (x, target, weight, input_dtensor, target_dtensor, weight_dtensor)

            return input_tuple
        else:
            input_tuple = (x, target, input_dtensor, target_dtensor)
            
            return input_tuple


    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_replicate(self):
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Replicate()], [Replicate()])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="none")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="none")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_not_evenly_shardable(self):
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(7, 8, [Shard(0)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_evenly_shardable(self):
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        self.assertEqual(loss_dtensor.full_tensor(), loss)
        self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)


    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_input_shard0_evenly_shardable_weight(self):
        reductions = ["none", "sum"]
        x, target, weight, input_dtensor, target_dtensor, weight_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)], [Replicate()])

        for re in reductions:
            loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, weight, re)
            loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, weight_dtensor, re)

            self.assertEqual(loss_dtensor.full_tensor(), loss)
            self.assertEqual(log_prob_dtensor.full_tensor(), log_prob)

    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_replicate_reduction_is_mean(self):
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Replicate()], [Replicate()])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="mean")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="mean")

        loss.backward()
        loss_dtensor.backward()
        self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)

    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_input_shard0_reduction_is_none(self):
        reductions = ["none", "sum", "mean"]
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Shard(0)], [Shard(0)])
        
        for re in reductions:
            loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction=re)
            loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction=re)
            if re == "none":
                loss.backward()
                loss_dtensor.backward()
                self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)
            else:
                grad = torch.randn(loss.size(), device="npu")
                grad_dtensor = distribute_tensor(grad, input_dtensor.mesh, [Shard(0)])

                loss.backward(grad)
                loss_dtensor.backward(grad_dtensor)
                self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)
    
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_npu_npu_cross_entropy_loss_backward_input_shard1_reduction_is_sum(self):
        x, target, input_dtensor, target_dtensor = self.generate_data_cross_entropy_loss(8, 8, [Shard(1)], [Shard(0)])

        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, target, reduction="sum")
        loss_dtensor, log_prob_dtensor, _, _ = torch_npu.npu_cross_entropy_loss(input_dtensor, target_dtensor, reduction="sum")
        
        loss.backward()
        loss_dtensor.backward()
        self.assertEqual(input_dtensor.grad.full_tensor(), x.grad)
    
    @with_comms
    def generate_data_repeat_interleave_self_int(self, size, repeats_value, input_strategy):
        mesh = self.build_device_mesh()

        input_tensor = torch.randn(size, device="npu", requires_grad=True)
        input_dtensor = distribute_tensor(input_tensor, mesh, input_strategy)

        return input_tensor, repeats_value, input_dtensor

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_replicate(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((5, 5), 3, [Replicate()])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard1(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((8, 8), 3, [Shard(1)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard0(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((8, 8), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_dim_is_None_shard0(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((5, 8), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
        output = torch.repeat_interleave(input_tensor, repeats_value)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_dim_is_None_shard1(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((5, 8), 3, [Shard(1)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
        output = torch.repeat_interleave(input_tensor, repeats_value)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_dim_is_None_shard0_is_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((8, 5), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
        output = torch.repeat_interleave(input_tensor, repeats_value)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard0_dim1_is_not_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((5, 5), 3, [Shard(0)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_self_int_shard1_dim1_is_not_evenly_shardable(self):
        input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int((5, 8), 3, [Shard(1)])

        output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
        output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)
        
        self.assertEqual(output_dtensor.full_tensor(), output)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_shard0_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard1_replicate_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_replicate_dim_None(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Replicate()])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value)
            output = torch.repeat_interleave(input_tensor, repeats_value)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Replicate()])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard00_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(0)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard01_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(0)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Shard(1)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard10_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Shard(0)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_torch_repeat_interleave_backward_self_int_shard11_dim1(self):
        sizes = [(2, 2), (5, 5), (5, 8), (8, 5), (8, 8)]

        for size in sizes:
            input_tensor, repeats_value, input_dtensor = self.generate_data_repeat_interleave_self_int(size, 3, [Shard(1)])

            output_dtensor = torch.repeat_interleave(input_dtensor, repeats_value, dim=1)
            output = torch.repeat_interleave(input_tensor, repeats_value, dim=1)

            grad_tensor = torch.randn(output.size(), device="npu")
            grad_dtensor = distribute_tensor(grad_tensor, input_dtensor.mesh, [Shard(1)])

            output_dtensor.backward(grad_dtensor)
            output.backward(grad_tensor)
            self.assertEqual(input_dtensor.grad.full_tensor(), input_tensor.grad)


if __name__ == "__main__":
    run_tests()
