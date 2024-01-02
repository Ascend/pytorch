import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestUnique2(TestCase):

    def test_unique2(self):
        shape_format = [
            [[np.uint8, (2, 3)], True, True, True],
            [[np.int8, (2, 3)], True, True, True],
            [[np.int16, (2, 3)], True, True, True],
            [[np.int32, (2, 3)], True, True, True],
            [[np.int64, (2, 3)], True, True, False],
            [[np.int64, (5, 3)], True, False, True],
            [[np.int64, (2, 3, 4)], True, False, False],
            [[np.int64, (3, 3)], True, True, True],
            [[np.int64, (2, 3)], True, False, False],
            [[np.float32, (2, 3)], True, False, False],
            [[np.bool_, (2, 3)], True, True, True],
            [[np.float16, (2, 3)], True, True, True],
            [[np.float16, (208, 3136, 19, 5)], True, False, True]
        ]

        for item in shape_format:
            input1 = np.random.uniform(-10, 10, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            if item[0][0] == np.float16:
                cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            npu_input1 = torch.from_numpy(input1).npu()

            cpu_output_y, cpu_yInverse, cpu_yCounts = torch._unique2(cpu_input1, item[1], item[2], item[3])
            npu_output_y, npu_yInverse, npu_yCounts = torch._unique2(npu_input1, item[1], item[2], item[3])

            cpu_output_y = cpu_output_y.numpy()
            if item[0][0] == np.float16:
                cpu_output_y = cpu_output_y.astype(np.float16)
            self.assertRtolEqual(cpu_output_y, npu_output_y.cpu().numpy())
            self.assertRtolEqual(cpu_yInverse.numpy(), npu_yInverse.cpu().numpy())
            self.assertRtolEqual(cpu_yCounts.numpy(), npu_yCounts.cpu().numpy())


if __name__ == "__main__":
    run_tests()
