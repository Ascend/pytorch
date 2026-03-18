import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestWithDevice(TestCase):

    @skipIfUnsupportMultiNPU(2)
    def test_with_device(self):
        torch.npu.set_device(1)
        # -float -> _get_device_index -> return current_device
        # -int -> ignore, return -1
        # future exchangeDevice:
        #     if < std::numeric_limits<c10::DeviceIndex>::min(), raise error
        #     else, ignore, return -1 
        for i in [-258, -200.8, -128, -128.8, -127.99, -7, -7.88, -1, -0.2]:
            s = torch.npu.Stream(i)
            self.assertEqual(s.device_index, 1)
            device = torch.npu.current_device()
            self.assertEqual(device, 1)


if __name__ == "__main__":
    run_tests()
