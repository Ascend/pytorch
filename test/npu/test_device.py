import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDevice(TestCase):
    def device_monitor(func):
        def wrapper(self, *args, **kwargs):
            device_id = 0
            torch.npu.set_device(device_id)
            npu_device = torch.randn(2).npu(device_id).device
            device_types = [
                "npu",
                "npu:" + str(device_id),
                torch.device("npu:" + str(device_id)),
                torch.device("npu:" + str(device_id)).type,
                npu_device
            ]
            for device_type in device_types:
                kwargs["device"] = device_type
                npu_tensor = func(self, *args, **kwargs)
                self.assertEqual(npu_tensor.device.type, "npu")
                self.assertEqual(npu_tensor.device.index, device_id)
            kwargs["device"] = None
            func(self, *args, **kwargs)

        return wrapper

    @device_monitor
    def test_torch_tensor_to_device(self, device=None):
        cpu_tensor = torch.randn(2, 3)
        return cpu_tensor.to(device, torch.int64)

    @device_monitor
    def test_torch_tensor_new_empty_with_device_input(self, device=None):
        npu_tensor = torch.ones(2, 3).to(device)
        return npu_tensor.new_empty((2, 3), dtype=torch.float16, device=device)

    @device_monitor
    def test_torch_func_arange_with_device_input(self, device=None):
        return torch.arange(5, dtype=torch.float32, device=device)

    @device_monitor
    def test_torch_func_zeros_with_device_input(self, device=None):
        return torch.zeros((2, 3), dtype=torch.int8, device=device)

    @device_monitor
    def test_tensor_method_npu_with_device_input(self, device=None):
        if isinstance(device, str):
            device = torch.device(device)
        cpu_input = torch.randn(2, 3)
        return cpu_input.npu(device)

    @device_monitor
    def test_torch_func_tensor_with_device_input(self, device=None):
        return torch.tensor((2, 3), device=device)

    def test_device_argument_as_input(self):
        device_str = "npu:0"

        torch.npu.set_device(device_str)
        device = torch.device(device_str)
        assert isinstance(device, torch.device)

        torch.npu.set_device(device)
        tensor = torch.rand(2, 3).npu()
        assert isinstance(tensor.device, torch.device)
        assert tensor.device.type == "npu"
        assert tensor.device.index == 0

        new_device = torch.device(device)
        assert isinstance(new_device, torch.device)
        assert new_device.type == "npu"
        assert new_device.index == 0

        new_device = torch.device(device=device)
        assert isinstance(new_device, torch.device)
        assert new_device.type == "npu"
        assert new_device.index == 0

        new_device = torch.device(device=device_str)
        assert isinstance(new_device, torch.device)
        assert new_device.type == "npu"
        assert new_device.index == 0

        new_device = torch.device(type="npu", index=0)
        assert isinstance(new_device, torch.device)
        assert new_device.type == "npu"
        assert new_device.index == 0

    def test_torch_npu_device(self):
        device = torch.device(0)
        assert device.type == "npu"
        device = torch.device(device=0)
        assert device.type == "npu"
        assert isinstance(device, torch._C.device)
        assert isinstance(device, torch.device)

    def test_multithread_device(self):
        import threading

        def _worker(result):
            try:
                cur = torch_npu.npu.current_device()
                self.assertEqual(cur, 0)
            except Exception:
                result[0] = 1

        result = [0]
        torch.npu.set_device("npu:0")
        thread = threading.Thread(target=_worker, args=(result,))
        thread.start()
        thread.join()
        self.assertEqual(result[0], 0)


if __name__ == '__main__':
    run_tests()
