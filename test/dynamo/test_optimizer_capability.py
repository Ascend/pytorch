import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch
import torch.optim.optimizer as optimizer_module
from torch._dynamo.testing import CompileCounter
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices

import torch_npu
from torch_npu.utils import _optim


class TestForeachCapability(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_supported_devices_preserve_upstream_entries(self):
        npu_backend = torch._C._get_privateuse1_backend_name()
        upstream = _get_foreach_kernels_supported_devices()
        other_devices = [device for device in upstream if device != npu_backend]
        for name, supported in (
            ("Ascend910B3", True),
            ("Ascend910_9391", True),
            ("Ascend950DT_9582", True),
            ("Ascend310P", False),
        ):
            with (
                self.subTest(name=name),
                mock.patch.object(torch_npu.npu, "current_device", return_value=0),
                mock.patch.object(
                    torch_npu.npu,
                    "get_device_properties",
                    return_value=SimpleNamespace(name=name),
                ),
            ):
                devices = _optim.patch_supported_devices()
                self.assertEqual(
                    [device for device in devices if device != npu_backend],
                    other_devices,
                )
                self.assertEqual(npu_backend in devices, supported)

    def test_supported_devices_does_not_mutate_upstream_list(self):
        npu_backend = torch._C._get_privateuse1_backend_name()
        upstream = ["cuda", "xpu", "mtia", npu_backend]
        original = list(upstream)
        with (
            mock.patch.object(
                _optim, "_get_foreach_kernels_supported_devices", return_value=upstream
            ),
            mock.patch.object(torch_npu.npu, "current_device", return_value=0),
            mock.patch.object(
                torch_npu.npu,
                "get_device_properties",
                return_value=SimpleNamespace(name="Ascend310P"),
            ),
        ):
            self.assertEqual(_optim.patch_supported_devices(), original[:-1])
            self.assertEqual(upstream, original)

    @unittest.skipUnless(torch_npu.npu.is_available(), "NPU is required")
    def test_capability_query_fullgraph(self):
        npu_backend = torch._C._get_privateuse1_backend_name()

        def fn(x):
            supported = optimizer_module._get_foreach_kernels_supported_devices()
            return x + (1 if npu_backend in supported else -1)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        torch.testing.assert_close(compiled(x), fn(x))
        torch.testing.assert_close(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 1)

    @unittest.skipUnless(torch_npu.npu.is_available(), "NPU is required")
    def test_capability_query_current_device_guard(self):
        torch_npu.npu.init()
        npu_backend = torch._C._get_privateuse1_backend_name()
        current = [0]
        properties = torch_npu.npu.get_device_properties(0)
        names = ["Ascend910B3", "Ascend310P"]

        def fn(x):
            supported = optimizer_module._get_foreach_kernels_supported_devices()
            return x + (1 if npu_backend in supported else -1)

        counter = CompileCounter()
        # Simulate two device capabilities on CPU; do not access a second NPU.
        with (
            mock.patch.object(
                torch_npu._C, "_npu_getDevice", side_effect=lambda: current[0]
            ),
            mock.patch.object(
                torch_npu._C, "_npu_getDeviceProperties", return_value=properties
            ),
            mock.patch.object(
                type(properties), "name", new=property(lambda prop: names[current[0]])
            ),
            mock.patch.object(torch_npu.npu, "device_count", return_value=2),
        ):
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            x = torch.ones(2)
            for index in (0, 0, 1, 1, 0):
                current[0] = index
                expected = x + (1 if index == 0 else -1)
                torch.testing.assert_close(compiled(x), expected)
            self.assertEqual(counter.frame_count, 2)
            torch._dynamo.reset()

    def _check_optimizer_lifetime(self, optimizer_class):
        def run_optimizer():
            parameter = torch.nn.Parameter(torch.ones(10, 10, device="npu"))
            expected = torch.nn.Parameter(torch.ones(10, 10))
            optimizer = optimizer_class([parameter], lr=0.01)
            reference = optimizer_class([expected], lr=0.01, foreach=False)

            def step():
                optimizer.step()

            compiled = torch.compile(step, backend="eager")
            for _ in range(2):
                parameter.grad = torch.full_like(parameter, 0.25)
                expected.grad = torch.full_like(expected, 0.25)
                with torch.no_grad():
                    compiled()
                    reference.step()
                torch.testing.assert_close(parameter.cpu(), expected)
            parameter_ref = weakref.ref(parameter)
            self.assertIsNotNone(parameter_ref())
            return parameter_ref

        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            parameter_ref = run_optimizer()
            self.assertIsNone(parameter_ref())
        finally:
            if gc_enabled:
                gc.enable()

    @unittest.skipUnless(torch_npu.npu.is_available(), "NPU is required")
    def test_adam_static_address_finalizer(self):
        self._check_optimizer_lifetime(torch.optim.Adam)

    @unittest.skipUnless(torch_npu.npu.is_available(), "NPU is required")
    def test_adamw_static_address_finalizer(self):
        self._check_optimizer_lifetime(torch.optim.AdamW)

    @unittest.skipUnless(torch_npu.npu.is_available(), "NPU is required")
    def test_sgd_static_address_finalizer(self):
        self._check_optimizer_lifetime(torch.optim.SGD)


if __name__ == "__main__":
    unittest.main()
