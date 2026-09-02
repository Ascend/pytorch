import torch
import torch_npu

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclgraphRngState(TestCase):
    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_get_set_rng_state_during_capture(self):
        torch.npu.set_device(0)

        torch_npu.npu.manual_seed(123)
        eager_state = torch_npu.npu.get_rng_state()
        eager_first = torch.rand(8, device="npu")
        torch_npu.npu.set_rng_state(eager_state)
        eager_second = torch.rand(8, device="npu")
        eager_next = torch.rand(8, device="npu")

        torch_npu.npu.manual_seed(123)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_state = torch_npu.npu.get_rng_state()
            graph_first = torch.rand(8, device="npu")
            torch_npu.npu.set_rng_state(graph_state)
            graph_second = torch.rand(8, device="npu")

        torch_npu.npu.manual_seed(123)
        graph.replay()
        torch.npu.synchronize()
        graph_next = torch.rand(8, device="npu")

        self.assertEqual(eager_first, eager_second)
        self.assertEqual(graph_first, graph_second)
        self.assertEqual(graph_first, eager_first)
        self.assertEqual(graph_next, eager_next)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_set_rng_state_with_nonzero_offset_during_capture(self):
        torch.npu.set_device(0)

        torch_npu.npu.manual_seed(123)
        eager_pre = torch.rand(8, device="npu")
        eager_state = torch_npu.npu.get_rng_state()
        eager_first = torch.rand(8, device="npu")
        torch_npu.npu.set_rng_state(eager_state)
        eager_second = torch.rand(8, device="npu")
        eager_next = torch.rand(8, device="npu")

        torch_npu.npu.manual_seed(123)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_pre = torch.rand(8, device="npu")
            graph_state = torch_npu.npu.get_rng_state()
            graph_first = torch.rand(8, device="npu")
            torch_npu.npu.set_rng_state(graph_state)
            graph_second = torch.rand(8, device="npu")

        torch_npu.npu.manual_seed(123)
        graph.replay()
        torch.npu.synchronize()
        graph_next = torch.rand(8, device="npu")

        self.assertEqual(graph_pre, eager_pre)
        self.assertEqual(eager_first, eager_second)
        self.assertEqual(graph_first, graph_second)
        self.assertEqual(graph_first, eager_first)
        self.assertEqual(graph_next, eager_next)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_set_rng_state_seed_mismatch_raises(self):
        torch.npu.set_device(0)

        torch_npu.npu.manual_seed(0)
        torch.rand(1, device="npu")
        torch_npu.npu.manual_seed(1)
        mismatched_state = torch_npu.npu.get_rng_state()
        torch_npu.npu.manual_seed(0)

        error = (
            "NPUGeneratorImpl::set_current_seed can be called during stream "
            "capture only if new seed is the same as the original seed."
        )
        graph = torch.npu.NPUGraph()
        with self.assertRaisesRegex(RuntimeError, error):
            with torch.npu.graph(graph):
                torch_npu.npu.set_rng_state(mismatched_state)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_checkpoint_preserve_rng_state(self):
        torch.npu.set_device(0)
        torch.npu.manual_seed(42)

        def fn(x):
            return x * torch.sigmoid(torch.randn(1, device="npu"))

        # Warm up the random op before capture.
        fn(torch.ones(1, device="npu"))

        torch.npu.manual_seed(42)
        eager_in = torch.ones(1, device="npu", requires_grad=True)
        eager_out = torch.utils.checkpoint.checkpoint(
            fn, eager_in, use_reentrant=False, preserve_rng_state=True
        )
        (eager_in_grad,) = torch.autograd.grad(eager_out, eager_in)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_in = torch.ones(1, device="npu", requires_grad=True)
            graph_out = torch.utils.checkpoint.checkpoint(
                fn, graph_in, use_reentrant=False, preserve_rng_state=True
            )
            (graph_in_grad,) = torch.autograd.grad(graph_out, graph_in)

        torch.npu.manual_seed(42)
        graph.replay()
        torch.npu.synchronize()

        self.assertEqual(eager_in_grad, graph_in_grad, prec=0.0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_manual_seed_mismatch_raises(self):
        torch.npu.set_device(0)
        torch.npu.manual_seed(0)

        error = (
            "NPUGeneratorImpl::set_current_seed can be called during stream "
            "capture only if new seed is the same as the original seed."
        )
        graph = torch.npu.NPUGraph()
        with self.assertRaisesRegex(RuntimeError, error):
            with torch.npu.graph(graph):
                torch.npu.manual_seed(1)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_register_generator_state_under_inference_mode(self):
        torch.npu.set_device(0)

        generator = torch.Generator(device="npu")
        generator.manual_seed(0)

        graph = torch.npu.NPUGraph()
        with torch.inference_mode():
            graph.register_generator_state(generator)

        with torch.npu.graph(graph):
            graph_out = torch.rand(8, device="npu", generator=generator)

        eager_generator = torch.Generator(device="npu")
        eager_generator.manual_seed(0)
        eager_ref = torch.rand(8, device="npu", generator=eager_generator)

        generator.manual_seed(0)
        graph.replay()
        torch.npu.synchronize()

        self.assertEqual(graph_out, eager_ref)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_rng_after_failed_capture(self):
        """Test RNG capture after an operation is rejected during capture."""
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        x = torch.ones(1, device="npu")

        with torch.npu.stream(stream):
            graph.capture_begin()
            with self.assertRaises(RuntimeError):
                (x + 1).item()
            # NPU rejects the stream synchronization without invalidating capture.
            graph.capture_end()

        torch.npu.current_stream().wait_stream(stream)

        result = torch.randn(4, device="npu")
        self.assertEqual(result.shape, (4,))

        # Capture again on the same stream to verify recovery.
        new_graph = torch.npu.NPUGraph()
        buf = torch.empty(4, device="npu")
        with torch.npu.stream(stream):
            new_graph.capture_begin()
            buf.copy_(torch.randn_like(buf))
            new_graph.capture_end()
        torch.npu.current_stream().wait_stream(stream)
        buf.zero_()
        new_graph.replay()
        torch.npu.synchronize()
        self.assertFalse(torch.allclose(buf, torch.zeros_like(buf)))

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_rng_concurrent_replay_on_different_streams(self):
        torch.npu.set_device(0)
        seed = 1234
        shape = (64,)

        torch.npu.manual_seed(seed)
        ref0 = torch.randn(shape, device="npu")
        ref1 = torch.randn(shape, device="npu")

        torch.npu.manual_seed(seed)
        g0 = torch.npu.NPUGraph()
        g1 = torch.npu.NPUGraph()
        s_cap = torch.npu.Stream()
        buf0 = torch.empty(shape, device="npu")
        buf1 = torch.empty(shape, device="npu")

        with torch.npu.stream(s_cap):
            g0.capture_begin()
            buf0.copy_(torch.randn_like(buf0))
            g0.capture_end()

        torch.npu.current_stream().wait_stream(s_cap)

        with torch.npu.stream(s_cap):
            g1.capture_begin()
            buf1.copy_(torch.randn_like(buf1))
            g1.capture_end()

        torch.npu.current_stream().wait_stream(s_cap)

        s0 = torch.npu.Stream()
        s1 = torch.npu.Stream()

        buf0.zero_()
        buf1.zero_()

        s0.wait_stream(torch.npu.current_stream())
        s1.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(s0):
            g0.replay()
        with torch.npu.stream(s1):
            g1.replay()

        torch.npu.synchronize()

        self.assertEqual(buf0, ref0)
        self.assertEqual(buf1, ref1)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_graph_rng_reset_recapture(self):
        torch.npu.set_device(0)
        seed = 4321
        shape = (8,)

        torch.npu.manual_seed(seed)
        ref = torch.randn(shape, device="npu")

        torch.npu.manual_seed(seed)
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        buf = torch.empty(shape, device="npu")

        with torch.npu.stream(stream):
            graph.capture_begin()
            buf.copy_(torch.randn_like(buf))
            graph.capture_end()
        torch.npu.current_stream().wait_stream(stream)

        graph.reset()

        torch.npu.manual_seed(seed)
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            graph.capture_begin()
            buf.copy_(torch.randn_like(buf))
            graph.capture_end()
        torch.npu.current_stream().wait_stream(stream)

        buf.zero_()
        graph.replay()
        torch.npu.synchronize()

        self.assertEqual(buf, ref)


if __name__ == "__main__":
    run_tests()
