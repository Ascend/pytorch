import os
import torch

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests

os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:False'


class TestNpu(TestCase):

    @skipIfUnsupportMultiNPU(2)
    def test_creat_tensor(self):
        device_number = torch.npu.device_count()
        for device_idx in range(device_number):
            t = torch.randn(2, 255, 255, device=f"npu:{device_idx}")
            self.assertTrue(str(t.device) == f"npu:{device_idx}")

    @skipIfUnsupportMultiNPU(2)
    def test_creat_stream(self):
        device_number = torch.npu.device_count()
        stream_instance = set()
        for i in range(device_number):
            torch.npu.set_device(i)
            default_stream = torch.npu.default_stream()
            current_stream = torch.npu.current_stream()
            stream_instance.add(current_stream)
        self.assertTrue(len(stream_instance) == device_number)

    def _test_host_to_device(self, t_cpu):
        t_device_0 = t_cpu.to("npu:0")
        self.assertTrue(str(t_device_0.device) == "npu:0")
        t_device_1 = t_cpu.to("npu:1")
        self.assertTrue(str(t_device_1.device) == "npu:1")
        self.assertRtolEqual(t_cpu.numpy(), t_device_0.cpu().numpy())
        self.assertRtolEqual(t_cpu.numpy(), t_device_1.cpu().numpy())

    def _test_device_to_device(self, t_cpu):
        t_device_0 = t_cpu.to("npu:0")
        self.assertTrue(str(t_device_0.device) == "npu:0")
        t_device_1 = t_device_0.to("npu:1")
        self.assertTrue(str(t_device_1.device) == "npu:1")
        self.assertRtolEqual(t_device_1.cpu(
        ).numpy(), t_device_0.cpu().numpy())

    def _test_device_copy(self):
        t_cpu = torch.rand(2, 255, 255)
        self._test_host_to_device(t_cpu)
        self._test_device_to_device(t_cpu)

    def _test_module(self):

        class MyModule(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                if (x.device.type == 'cpu'):
                    x = self.fc1(x)
                    return self.fc2(x)
                else:
                    self.fc1 = self.fc1.to("npu:0")
                    self.fc2 = self.fc2.to("npu:1")
                    x = self.fc1(x)
                    return self.fc2(x.to("npu:1"))

        module = MyModule()
        module.eval()
        input_t = torch.rand(8, 2)
        output_cpu = module(input_t)
        output_npu = module(input_t.to("npu:0"))
        self.assertTrue(str(output_npu.device) == "npu:1")
        self.assertRtolEqual(output_cpu.detach().numpy(), output_npu.detach().cpu().numpy(), prec=1.e-3)

    @skipIfUnsupportMultiNPU(2)
    def test_aclop_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self._test_device_copy()
        self._test_module()

    @skipIfUnsupportMultiNPU(2)
    def test_opapi_with_multi_device(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self._test_device_copy()
        self._test_module()


if __name__ == '__main__':
    run_tests()
