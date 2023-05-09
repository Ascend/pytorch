import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestImageNormalize(TestCase):
    def result_error(self, npu_input, npu_output):
        if npu_output.shape != npu_input.shape:
            self.fail("shape error")
        if npu_output.dtype != npu_input.dtype:
            self.fail("dtype error")

    def test_image_normalize(self):
        cpu_input = np.random.uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)
        npu_input = torch.from_numpy(cpu_input).npu()

        mean = [0.485, 0.456, 0.406]
        variance = [0.229, 0.224, 0.225]

        npu_output = torch_npu.image_normalize(npu_input, mean, variance, dtype=0)

        self.result_error(npu_input, npu_output)


if __name__ == "__main__":
    if torch_npu.npu.get_device_name(0)[:10] == 'Ascend910B':
        run_tests()
