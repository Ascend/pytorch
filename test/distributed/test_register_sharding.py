import numpy as np
import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase


class TestRegisterSharding(NPUDTensorTestBase):
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


if __name__ == "__main__":
    run_tests()
