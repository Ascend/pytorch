import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestIFMR(TestCase):

    def supported_op_exec(self, input1, data_min, data_max, cumsum, bins_num, min_percentile,
                          max_percentile, search_start, search_end, search_step, with_offset):
        cdf = cumsum / torch.numel(input1)
        max_index = torch.sum(torch.where(cdf >= max_percentile, 0, 1))
        min_index = torch.sum(torch.where(cdf >= (1 - min_percentile), 0, 1))
        max_init = max_index / torch.tensor([bins_num]).npu() * (data_max - data_min) + data_min
        min_init = min_index / torch.tensor([bins_num]).npu() * (data_max - data_min) + data_min

        step = torch.arange(search_start, search_end, search_step)
        step = step.to('npu')

        if with_offset:
            if max_init < 0:
                max_init = torch.tensor([0]).npu()
            if min_init > 0:
                min_init = torch.tensor([0]).npu()
            min_list = min_init * torch.ones(step.shape).npu()
        else:
            max_init = torch.max(torch.abs(max_init), torch.abs(min_init))
        max_list = max_init * step

        if with_offset:
            scale = (max_list - min_list) / torch.tensor([255]).npu()
            scale = torch.where(scale < 1.192092896e-07, torch.ones([1]).npu(), scale)
            offset = torch.round(min_list / scale)
            offset = -(offset + 128)
        else:
            scale = max_list / torch.tensor([127]).npu()
            offset = torch.round(scale * 0)

        loss_list = torch.zeros(step.shape)
        for i in range(torch.numel(step)):
            quant_data = torch.round(input1 / scale[i]) + offset[i]
            quant_data = quant_data.clip(-128, 127)
            quant_data = (quant_data - offset[i]) * scale[i]
            loss_list[i] = torch.sum(torch.square(quant_data - input1))
        index = torch.argmin(loss_list)

        return scale[index].cpu().detach(), offset[index].cpu().detach()

    def custom_op_exec(self, input1, data_min, data_max, cumsum, min_percentile, max_percentile, search_start,
                       search_end, search_step, with_offset):
        scale, offset = torch_npu.npu_ifmr(input1, data_min, data_max, cumsum, min_percentile, max_percentile,
                                           search_start, search_end, search_step, with_offset)
        return scale.cpu().detach(), offset.cpu().detach()

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_npu_ifmr(self, device="npu"):
        item = [np.float32, 0, (2, 2, 4, 4)]
        _, npu_input = create_common_tensor(item, -1, 1)
        bins_num = 128
        min_percentile = 0.999999
        max_percentile = 0.999999
        search_start = 0.7
        search_end = 1.3
        search_step = 0.01
        with_offset = False

        data_min = torch.min(npu_input)
        data_max = torch.max(npu_input)
        data_min = torch.reshape(data_min, (1,))
        data_max = torch.reshape(data_max, (1,))

        hist = torch.histc(npu_input.to('cpu'),
                           bins=bins_num,
                           min=data_min[0].to('cpu'),
                           max=data_max[0].to('cpu'))
        cumsum = torch.cumsum(hist, dim=0).int()
        cumsum = cumsum.to('npu')

        supported_scale, supported_offset = self.supported_op_exec(npu_input, data_min, data_max, cumsum, bins_num,
                                                                   min_percentile, max_percentile, search_start,
                                                                   search_end, search_step, with_offset)
        custom_scale, custom_offset = self.custom_op_exec(npu_input, data_min, data_max, cumsum, min_percentile,
                                                          max_percentile, search_start, search_end, search_step,
                                                          with_offset)
        self.assertRtolEqual(supported_scale, custom_scale[0])
        self.assertRtolEqual(supported_offset, custom_offset[0])


if __name__ == "__main__":
    run_tests()
