
import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor.utils import run_and_get_code
import torch_npu
import torch_npu.testing


class TestSynchronizeSkip(TestCase):
    def test_synchronize_not_in_compiled_graph(self):

        def func_with_synchronize(x):
            y = x + 1.0
            torch_npu.npu.utils.synchronize()
            return y * 2.0

        x = torch.randn(32, 16, device="npu", dtype=torch.float32)
        expected = (x + 1.0) * 2.0
        compiled_func = torch.compile(func_with_synchronize, backend="inductor", dynamic=False)
        result, inductor_code_list = run_and_get_code(compiled_func, x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        full_code = "\n".join(inductor_code_list)
        self.assertNotIn("synchronize", full_code)
        self.assertIn("async_compile.triton", full_code)


class TestSynchronizeInterfaces(TestCase):
    def test_inductor_device_op_overrides(self):
        from torch_npu.utils._inductor import NPUDeviceOpOverrides
        overrides = NPUDeviceOpOverrides()
        self.assertTrue(overrides.synchronize().endswith(".synchronize()"))

    def test_dynamo_device_interface(self):
        from torch_npu.utils._dynamo_device import NpuInterface
        self.assertTrue(NpuInterface.synchronize is torch_npu.npu.utils.synchronize or NpuInterface.synchronize == torch_npu.npu.utils.synchronize)

    def test_set_compile_mode_calls_synchronize(self):
        with unittest.mock.patch.object(torch_npu.npu, "is_initialized", return_value=True, create=True), \
             unittest.mock.patch.object(torch_npu.npu, "synchronize") as mock_sync:
            torch_npu.npu.set_compile_mode(jit_compile=False)
            mock_sync.assert_called()

    def test_profiler_disable(self):
        import torch_npu.profiler.profiler as npu_profiler

        with unittest.mock.patch.object(torch_npu.npu, "synchronize") as mock_sync, \
             unittest.mock.patch.object(npu_profiler, "_disable_profiler_in_child_thread"):
            from torch_npu.profiler.profiler import profile
            profile.disable_profiler_in_child_thread()
            mock_sync.assert_called()

    def test_dataloader_init_calls_synchronize(self):
        try:
            import torch_npu.utils._module as npu_module
        except ImportError:
            self.skipTest("Required module is unavailable")

        with unittest.mock.patch.object(torch_npu.npu, "synchronize") as mock_sync, \
             unittest.mock.patch.object(npu_module, "origin_mpdl_iter_init") as mock_origin_init:
            mock_self = unittest.mock.MagicMock()
            npu_module._mpdl_iter_init(mock_self, loader=unittest.mock.MagicMock())

            mock_sync.assert_called()
            mock_origin_init.assert_called()

    def test_make_graphed_callables_calls_synchronize(self):
        import torch_npu.npu.graphs as npu_graphs
        
        def func(x):
            return x
        args = (torch.randn(1),)
        
        with unittest.mock.patch.object(torch_npu.npu, "synchronize") as mock_sync, \
             unittest.mock.patch.object(torch_npu.npu, "NPUGraph"), \
             unittest.mock.patch.object(torch_npu.npu, "Stream"), \
             unittest.mock.patch.object(npu_graphs, "graph_pool_handle", return_value=(0, 0)), \
             unittest.mock.patch.object(torch_npu.npu, "stream", side_effect=RuntimeError("stop")):
            try:
                npu_graphs.make_graphed_callables(func, args)
            except Exception:
                # Execution fails due to mocks, but sync should have happened already
                if not mock_sync.called:
                    raise
            mock_sync.assert_called()

if __name__ == "__main__":
    run_tests()
