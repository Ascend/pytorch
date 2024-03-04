# Copyright (c) 2022-2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestScaledMaskedSoftmax(TestCase):
    @unittest.skip("skip test_scaled_masked_softmax_shape_format now")
    def test_scaled_masked_softmax_shape_format(self):
        shape_format = [
            [[np.float16, 29, (16, 6, 128, 128)], [np.float16, 29, (16, 6, 128, 128)]],
            [[np.float16, 2, (16, 6, 128, 512)], [np.float16, 2, (16, 1, 128, 512)]],
            [[np.float16, 0, (16, 6, 512, 512)], [np.float16, 0, (16, 1, 512, 512)]],
            [[np.float16, 0, (4, 4, 2048, 2048)], [np.float16, 0, (4, 1, 2048, 2048)]],
            [[np.float32, 29, (16, 6, 128, 128)], [np.float32, 29, (16, 6, 128, 128)]],
            [[np.float32, 2, (16, 6, 128, 512)], [np.float32, 2, (16, 1, 128, 512)]],
            [[np.float32, 0, (16, 6, 512, 512)], [np.float32, 0, (16, 1, 512, 512)]],
        ]

        # forward ut test
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -5, 5)
            cpu_mask, npu_mask = create_common_tensor(item[1], -1, 1)
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_output = cpu_op_exec_forward(cpu_input, cpu_mask,
                                             scale, fixed_triu_mask)
            npu_output = npu_op_exec_forward(npu_input, npu_mask,
                                             scale, fixed_triu_mask)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

        # backward ut test
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -5, 5)
            cpu_y_grad, npu_y_grad = create_common_tensor(item[0], -5, 5)
            cpu_mask, npu_mask = create_common_tensor(item[1], -1, 1)
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            if cpu_y_grad.dtype == torch.float16:
                cpu_y_grad = cpu_y_grad.to(torch.float32)
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_x_grad = cpu_op_exec_backward(cpu_input, cpu_y_grad,
                                              cpu_mask, scale, fixed_triu_mask)
            npu_x_grad = npu_op_exec_backward(npu_input, npu_y_grad,
                                              npu_mask, scale, fixed_triu_mask)
            cpu_x_grad = cpu_x_grad.astype(npu_x_grad.dtype)
            self.assertRtolEqual(cpu_x_grad, npu_x_grad)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', 'dtype `bf16` is only supported on 910B, skip this ut!')
    def test_scaled_masked_softmax_bf16(self):
        shape_format = [
            [[torch.bfloat16, 29, (16, 6, 128, 128)], [torch.float16, 29, (16, 6, 128, 128)]],
            [[torch.bfloat16, 2, (16, 6, 128, 512)], [torch.float16, 2, (16, 1, 128, 512)]],
            [[torch.bfloat16, 0, (16, 6, 512, 512)], [torch.float16, 0, (16, 1, 512, 512)]],
            [[torch.bfloat16, 0, (4, 4, 2048, 2048)], [torch.float16, 0, (4, 1, 2048, 2048)]],
        ]

        # forward ut test
        for item in shape_format:
            cpu_input, npu_input = gen_data_bf16(item[0])
            cpu_mask, npu_mask = gen_data_bf16(item[1])
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_output = cpu_op_exec_forward(cpu_input, cpu_mask,
                                             scale, fixed_triu_mask)
            npu_output = npu_op_exec_forward(npu_input, npu_mask,
                                             scale, fixed_triu_mask)
            self.assertRtolEqual(cpu_output, npu_output)

        # backward ut test
        for item in shape_format:
            cpu_input, npu_input = gen_data_bf16(item[0])
            cpu_y_grad, npu_y_grad = gen_data_bf16(item[0])
            cpu_mask, npu_mask = gen_data_bf16(item[1])
            cpu_mask = cpu_mask > 0
            npu_mask = npu_mask > 0
            scale = random.uniform(-1, 1)
            fixed_triu_mask = False
            cpu_x_grad = cpu_op_exec_backward(cpu_input, cpu_y_grad,
                                              cpu_mask, scale, fixed_triu_mask)
            npu_x_grad = npu_op_exec_backward(npu_input, npu_y_grad,
                                              npu_mask, scale, fixed_triu_mask)
            self.assertRtolEqual(cpu_x_grad, npu_x_grad)


def cpu_op_exec_forward(x, mask, scale, fixed_triu_mask):
    x = x.float()
    if fixed_triu_mask:
        mask_tri = torch.triu(torch.ones(mask.shape, device=mask.device), diagonal=1).bool()
        output = F.softmax((x * scale).masked_fill(mask_tri, value=-10000), dim=-1).half()
    else:
        output = F.softmax((x * scale).masked_fill(mask, value=-10000), dim=-1).half()
    return output.detach().half().numpy()


def npu_op_exec_forward(x, mask, scale, fixed_triu_mask):
    output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
    return output.cpu().detach().half().numpy()


def cpu_op_exec_backward(x, y_grad, mask, scale, fixed_triu_mask):
    x.requires_grad_(True)
    x_fp32 = x.float()
    y_grad = y_grad.float()
    if fixed_triu_mask:
        mask_tri = torch.triu(torch.ones(mask.shape, device=mask.device), diagonal=1).bool()
        output = F.softmax((x_fp32 * scale).masked_fill(mask_tri, value=-10000), dim=-1).half()
        output.backward(y_grad)
    else:
        output = F.softmax((x_fp32 * scale).masked_fill(mask, value=-10000), dim=-1).half()
        output.backward(y_grad)
    x_grad = x.grad
    return x_grad.detach().half().numpy()


def npu_op_exec_backward(x, y_grad, mask, scale, fixed_triu_mask):
    x.requires_grad_(True)
    output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
    output.backward(y_grad)
    x_grad = x.grad
    return x_grad.half().cpu().detach().numpy()


def gen_data_bf16(item):
    dtype, npu_format, shape = item
    cpu_input = torch.randn(shape, dtype=dtype)
    npu_input = cpu_input.npu()
    if npu_format != -1:
        npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
    return cpu_input, npu_input


if __name__ == "__main__":
    run_tests()
