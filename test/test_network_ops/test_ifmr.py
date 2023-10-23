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

from functools import reduce

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIFMR(TestCase):
    def cpu_op_exec(self,
                    input_data,
                    with_offset,
                    bins_num=128,
                    min_percentile=0.999999,
                    max_percentile=0.999999,
                    search_range1=0.7,
                    search_range2=1.3,
                    search_step=0.01):
        pre_mode = np.float32
        input_data = input_data.numpy().astype(pre_mode)
        data_min = input_data.min()
        data_max = input_data.max()
        data_num = reduce(lambda x, y: x * y, input_data.shape)
        data_num = np.array(data_num, pre_mode)

        bins, threshold = np.histogram(input_data, bins_num)
        cumsum = np.cumsum(bins).astype(np.int32)

        bins_num = np.array(bins_num, pre_mode)
        cdf = cumsum.astype(pre_mode) / data_num
        max_index = np.where(cdf >= np.array(max_percentile, pre_mode), 0,
                             1).sum().astype(pre_mode)
        min_index = np.where(cdf >= np.array(1 - min_percentile, pre_mode), 0,
                             1).sum().astype(pre_mode)
        max_init = max_index / bins_num * (data_max - data_min) + data_min
        min_init = min_index / bins_num * (data_max - data_min) + data_min

        step = np.arange(search_range1,
                         search_range2,
                         search_step,
                         dtype=pre_mode)
        if with_offset:
            if max_init < 0:
                max_init = np.array(0, pre_mode)
            if min_init > 0:
                min_init = np.array(0, pre_mode)
            min_list = min_init * np.ones(step.shape, dtype=pre_mode)
        else:
            max_init = np.max([np.abs(max_init), np.abs(min_init)])
        max_list = max_init * step

        if with_offset:
            scale = (max_list - min_list) / 255
            scale = np.where(scale < 1.192092896e-07, 1, scale)
            offset = np.round(min_list / scale)
            offset = -(offset + 128)
        else:
            scale = max_list / 127
            offset = np.round(scale * 0)

        loss_list = np.zeros(step.shape, dtype=pre_mode)
        for i in range(step.size):
            quant_data = np.round(input_data / scale[i]) + offset[i]
            np.clip(quant_data, -128, 127, out=quant_data)
            quant_data = (quant_data - offset[i]) * scale[i]
            loss_list[i] = np.sum(np.square(quant_data - input_data))
        index = np.argmin(loss_list)
        return scale[index], offset[index]

    def npu_op_exec(self, input_data, with_offset):
        min_value = torch.min(input_data)
        max_value = torch.max(input_data)
        min_value = torch.reshape(min_value, (1,))
        max_value = torch.reshape(max_value, (1,))
        hist = torch.histc(input_data.to('cpu'),
                           bins=128,
                           min=min_value[0].to('cpu'),
                           max=max_value[0].to('cpu'))
        cdf = torch.cumsum(hist, dim=0).int()

        cdf = cdf.to('npu')
        scale, offset = torch_npu.npu_ifmr(input_data,
                                           min_value,
                                           max_value,
                                           cdf,
                                           min_percentile=0.999999,
                                           max_percentile=0.999999,
                                           search_start=0.7,
                                           search_end=1.3,
                                           search_step=0.01,
                                           with_offset=with_offset)

        return scale, offset

    def test_ifrm_with_offset(self, device="npu"):
        format_list = [0, 3]
        shape_list = [(2, 2, 3, 4), (5, 5)]
        shape_format = [[np.float32, i, j] for i in format_list
                        for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            scale_cpu, offset_cpu = self.cpu_op_exec(cpu_input,
                                                     with_offset=True)
            scale_npu, offset_npu = self.npu_op_exec(npu_input,
                                                     with_offset=True)
            self.assertTrue((scale_cpu - scale_npu[0]) / scale_cpu < 0.0001)
            self.assertEqual(offset_cpu, offset_npu[0])

    def test_ifrm_without_offset(self, device="npu"):
        format_list = [0, 3]
        shape_list = [(2, 2, 3, 4), (5, 5)]
        shape_format = [[np.float32, i, j] for i in format_list
                        for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            scale_cpu, offset_cpu = self.cpu_op_exec(cpu_input,
                                                     with_offset=False)
            scale_npu, offset_npu = self.npu_op_exec(npu_input,
                                                     with_offset=False)
            self.assertTrue((scale_cpu - scale_npu[0]) / scale_cpu < 0.0001)
            self.assertEqual(offset_cpu, offset_npu[0])


if __name__ == "__main__":
    run_tests()
