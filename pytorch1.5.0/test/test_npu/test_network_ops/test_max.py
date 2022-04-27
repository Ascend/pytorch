# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestMax(TestCase):
    def cpu_op_exec(self, input1):
        '''
        调用算子  torch.max(input) → Tensor
        '''
        output = torch.max(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        '''
        调用适配算子函数  Tensor max_npu(const Tensor& self)
        '''
        output = torch.max(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_other_exec(self, input1, input2):
        '''
        调用算子  torch.max(input, other, out=None) → Tensor
        '''
        output = torch.max(input1, input2)
        output = output.numpy()
        return output

    def npu_op_other_exec(self, input1, input2):
        '''
        适配算子函数  Tensor max_npu(const Tensor& self, const Tensor& other)
        '''
        output = torch.max(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.max(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_dim_exec(self, input1, dim, keepdim):
        '''
        调用算子  torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
        '''
        output1, output2 = torch.max(input1, dim, keepdim)
        output1 = output1.numpy()
        # 这里需要将索引从64位转32位 便于拿去与npu的对比
        output2 = output2.int().numpy()  
        return output1, output2

    def npu_op_dim_exec(self, input1, dim, keepdim):
        '''
        适配算子函数  tuple<Tensor, Tensor> max_npu(const Tensor& self, int64_t dim, bool keepdim)
        '''
        output1, output2 = torch.max(input1, dim, keepdim)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2
    
    def cpu_max_values_exec(self, input):
        output = input.max()
        output = output.numpy()
        return output
        
    def npu_max_values_exec(self, input):
        output = input.max()
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def _cpu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch._max(input1, dim, keepdim)
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def _npu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch._max(input1, dim, keepdim)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def cpu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype)
        indices = torch.tensor(0).to(torch.long)
        torch.max(input1, dim=dim, keepdim=keepdim, out=(out,indices))
        out = out.numpy()
        indices = indices.numpy()
        return out,indices

    def npu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype).npu() 
        indices = torch.tensor(0).to(torch.long).npu()
        torch.max(input1, dim=dim, keepdim=keepdim, out=(out,indices))
        out = out.to("cpu").numpy()
        indices = indices.to("cpu").numpy()
        return out,indices

    def max_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)

            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def max_result_dim(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(npu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)

            self.assertRtolEqual(cpu_output_dim, npu_output_dim)
    
    def _max_result_dim(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self._cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self._npu_op_dim_exec(npu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)

            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def max_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output_other = self.cpu_op_other_exec(cpu_input1, cpu_input2)
            npu_output_other = self.npu_op_other_exec(npu_input1, npu_input2)

            cpu_output_other = cpu_output_other.astype(npu_output_other.dtype)
            self.assertRtolEqual(cpu_output_other, npu_output_other)

    def max_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], -100, 100)
            cpu_input4, npu_input4 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_other_exec(cpu_input1, cpu_input2)
            npu_output_out1 = self.npu_op_other_exec(npu_input1, npu_input2)
            npu_output_out2 = self.npu_op_exec_out(npu_input1, npu_input2, npu_input4)
            cpu_output = cpu_output.astype(npu_output_out1.dtype)

            self.assertRtolEqual(cpu_output, npu_output_out1)
            self.assertRtolEqual(cpu_output, npu_output_out2)

            cpu_out_dim, cpu_out_indices = self.cpu_op_dim_exec_out(cpu_input1, dim=0, keepdim=True)
            npu_out_dim, npu_out_indices = self.npu_op_dim_exec_out(npu_input1, dim=0, keepdim=True)
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(npu_input1, dim=0, keepdim=True)
            cpu_out_dim = cpu_out_dim.astype(npu_out_dim.dtype)
            if npu_out_dim.dtype != np.float16:
                self.assertRtolEqual(npu_out_dim, cpu_out_dim)
                self.assertRtolEqual(npu_out_indices, cpu_out_indices)
            else:
                self.assertRtolEqual(npu_out_dim, npu_output_dim)
                self.assertRtolEqual(npu_out_indices, npu_output_indices)
    
    # Npu and cpu have different logic to find the maximum value index. 
    # The existence of two maximum values will cause the second output to be different. 
    def max_name_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(cpu_input1, item[1], item[2])

            if npu_output_dim.dtype != np.float16:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim)
                #self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))
            else:
                self.assertRtolEqual( npu_output_dim, cpu_output_dim.astype(np.float16))
                #self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))

    # Npu and cpu have different logic to find the maximum value index. 
    # The existence of two maximum values will cause the second output to be different. 
    def max_name_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec_out(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec_out(npu_input1, item[1], item[2])

            if npu_output_dim.dtype != np.float16:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim)
                #self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))
            else:
                self.assertRtolEqual( npu_output_dim, cpu_output_dim.astype(np.float16))
                #self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))
                
    def max_values_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_max_values_exec(cpu_input1)
            npu_output = self.npu_max_values_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)            

    def test_max_out_result(self, device):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 58, 28, 28]],  [np.float16, 0, [58, 58, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [256, 128, 7, 7]],   [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [256, 3, 224, 224]], [np.float32, 0, [3, 3, 7, 7]]],
            [[np.float32, 0, [2, 3, 3, 3]],       [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]],   [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.max_out_result_other(shape_format)

    def test_max_shape_format_fp16_1d(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_1d(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_2d(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_2d(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_3d(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    # ---------------------------------------dim
    def test_max_dim_shape_format_fp16_1d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_1d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_2d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_2d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_3d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_1d_(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self._max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_1d_(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self._max_result_dim(shape_format)
    
    #One-dimensional NZ to ND result is wrong, CCB has given a conclusion 
    def test_max_dim_shape_format_fp16_2d_(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self._max_result_dim(shape_format)
    
    #One-dimensional NZ to ND result is wrong, CCB has given a conclusion 
    def test_max_dim_shape_format_fp32_2d_(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self._max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_3d_(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self._max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_3d_(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self._max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_4d_(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self._max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_4d_(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self._max_result_dim(shape_format)

    # -----------------------------other

    def test_max_other_shape_format_fp16_1d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_1d(self, device):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_2d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_2d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_3d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34]], np.random.randint(0, 4), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)
    
    def test_max_dimname_shape_format(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34], ('N', 'C', 'H', 'W')],
         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_name_result_other(shape_format)
    
    def test_max_dimname_shape_format_fp16(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34], ('N', 'C', 'H', 'W')],
         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_name_result_other(shape_format)
    
    def test_max_dimname_out_shape_format(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34], ('N', 'C', 'H', 'W')],
         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_name_out_result_other(shape_format)
    
    def test_max_dimname_out_shape_format_fp16(self, device):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34], ('N', 'C', 'H', 'W')],
         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.max_name_out_result_other(shape_format)

    def test_max_values_shape_format(self, device):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]  
        self.max_values_result(shape_format)

    def test_max_out_5d_ncdhw(self, device):
        a_cpu = torch.rand(2, 2, 2, 2, 2)
        b_cpu = torch.rand(2, 2, 2, 2, 2)
        a_npu = a_cpu.clone().npu()
        b_npu = b_cpu.clone().npu()
        cpu_output = self.cpu_op_other_exec(a_cpu, b_cpu)
        npu_output_out2 = self.npu_op_exec_out(a_npu, b_npu, a_npu)
        self.assertRtolEqual(cpu_output, npu_output_out2)

instantiate_device_type_tests(TestMax, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
