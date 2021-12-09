# Copyright (c) 2021, Huawei Technologies.All rights reserved.
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

import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import graph_mode
import torch.nn as nn

class TestContiguous(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output


    @graph_mode
    def test_uncontiguous_mul_reshape(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1.reshape([2,6])
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 1, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_mul_transpose(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1.transpose(2,1)
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 1, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_multi_expand(self, device):
        def run_noncontig_case(input1, input2):
            add0 = input1.expand(2,2,3).sum()
            add1 = input1.expand(3,2,1,3).sum()
            add2 = input1.expand(4,2,4,3).sum()
            add3 = input2.expand(1,2,1).sum()
            add4 = input2.expand(5,4,2).sum()
            output = add0 + add1 + add2 + add3 + add4
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, 0, (2, 1, 3)], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, ()], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, cpu_input2)
        npu_output = npu_uncontiguous_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_mul_slice(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1[:, :, 0:2]
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 1, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_mul_slice_mul(self, device):
        def run_noncontig_case(input1, input2):
            input1 = torch.mul(input1, input2)
            input1 = input1[1:3, :, :]
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 1, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_mul_select(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1[:, 2, :]
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (1, 4, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_uncontiguous_z_mul_index(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1[0:4:2, :, :]
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 1, 3)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_d2dcopy_view_mul(self, device):
        def run_noncontig_case(input1, input2):
            input1.copy_(input2)
            view1_res = input2.view(2,4)
            mul_res = view1_res * 2.0
            view2_res = mul_res.view(4,1,1,2)
            add1_res = input1 + view2_res

            slice_res = input2[1:2, :, :, :]
            mul2_res = slice_res * 3.0
            expand_res = mul2_res.expand(4,1,1,2)
            add2_res = add1_res + expand_res

            return add2_res
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, 0, (4, 1, 1, 2)], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, (4, 1, 1, 2)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, cpu_input2)
        npu_output = npu_uncontiguous_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_a_uncontiguous_mul_combine(self, device):
        def run_noncontig_case(input1, input2):
            input1 = input1[:, 1:, :, :]
            input1 = input1.view(4,1,15)
            output = torch.mul(input1, input2)
            return output
        def cpu_uncontiguous_op_exec(input1, input2):
            return run_noncontig_case(input1, input2).numpy()
        def npu_uncontiguous_op_exec(input1, input2):
            output = run_noncontig_case(input1, input2)
            return output.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (4, 2, 3, 5)], 1, 100)
        cpu_output = cpu_uncontiguous_op_exec(cpu_input1, 2.0)
        npu_output = npu_uncontiguous_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)

    
    def op_exec_inner_format_cpu(self, input_data, weight, in_channels, out_channels, kernel_size, padding=0, stride=1,
                                 dilation=1, bias=True):
        input1 = input_data
        weight1 = weight
        bias1 = False
        if bias != None:
            bias1 = True
        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=1)
        m1.weight.data = weight1
        Output = m1(input1)
        a = Output.view(9,6)/1000
        b = Output.view(6,9)
        mm = torch.mm(b, a)
        mm = mm.view(2,2,9)
        '''
        add1 = mm.transpose(2,1)
        add2 = mm[:, 0:1, 1:3]
        Output = add2 + add1
        '''

        return Output

    def op_exec_inner_format_npu(self, input_data, weight, in_channels, out_channels, kernel_size, padding=0, stride=1,
                                 dilation=1, bias=True):
        input1 = input_data
        weight1 = weight
        bias1 = False
        if bias != None:
            bias1 = True
        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=1)
        m1.weight.data = weight1
        m1 = m1.to("npu")
        Output = m1(input1)
        a = Output.view(9,6)/1000
        b = Output.view(6,9)
        mm = torch.mm(b, a)
        mm = mm.view(2,2,9)
        '''
        add1 = mm.transpose(2,1)
        add2 = mm[:, 0:1, 1:3]
        Output = add2 + add1
        '''
        return Output

    @graph_mode
    def test_conv2d_backward_shape_format(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float16, 0, [3, 3, 7, 7]], [np.float16, 0, [2, 3, 2, 2]], 0, [2, 2], 1, None],
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_inner_format_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0],
                                                       kernel_size=kernel_size, padding=item[2], stride=item[3],
                                                       dilation=item[4], bias=item[5])
            npu_output = self.op_exec_inner_format_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0],
                                                       kernel_size=kernel_size, padding=item[2], stride=item[3],
                                                       dilation=item[4], bias=item[5])
            torch.npu.disable_graph_mode()
            npu_output = npu_output.to("cpu")
            torch.npu.enable_graph_mode()
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())


instantiate_device_type_tests(TestContiguous, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
