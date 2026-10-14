# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._inductor.scheduler import FusedSchedulerNode, Scheduler
from torch._inductor.test_case import TestCase

import torch_npu
import torch_npu._inductor  # noqa: F401 - registers the NPU Inductor backend
from torch_npu.utils._dynamo import has_triton
from torch.testing._internal.common_utils import run_tests


@unittest.skipUnless(torch.npu.is_available() and has_triton(), "requires NPU Triton")
class NpuStreamFusionTests(TestCase):
    def compile_and_check_streams(self, fn, x):
        observed = []
        original = Scheduler.generate_stream_ctx_switching

        def check_mapping(scheduler, node):
            self.assertIn(node, scheduler.node_to_stream)
            if isinstance(node, FusedSchedulerNode):
                stream = scheduler.node_to_stream[node]
                self.assertTrue(all(scheduler.node_to_stream[leaf] == stream
                                    for leaf in node.get_nodes()))
                observed.append(stream)
            return original(scheduler, node)

        torch.npu.synchronize()
        with mock.patch.object(Scheduler, "generate_stream_ctx_switching", check_mapping):
            compiled = torch.compile(fn, fullgraph=True)
            actual = compiled(x)
        self.assertEqual(actual, fn(x))
        self.assertTrue(observed, "the regression must exercise fused scheduler nodes")
        return observed

    def test_side_stream_fusion_preserves_assignment(self):
        side = torch.npu.Stream()

        def fn(x):
            with side:
                intermediate = x + 1
                result = intermediate * 2
            side.synchronize()
            return intermediate, result

        streams = self.compile_and_check_streams(fn, torch.randn(32, device="npu"))
        self.assertTrue(all(stream > 0 for stream in streams))

    def test_two_stream_fusion_keeps_assignments_distinct(self):
        first = torch.npu.Stream()
        second = torch.npu.Stream()

        def fn(x):
            with first:
                intermediate_a = x + 1
                a = intermediate_a * 2
            with second:
                intermediate_b = x - 1
                b = intermediate_b * 3
            first.synchronize()
            second.synchronize()
            return intermediate_a, a, intermediate_b, b

        streams = self.compile_and_check_streams(fn, torch.randn(32, device="npu"))
        self.assertEqual(len(set(streams)), 2)
        self.assertTrue(all(stream > 0 for stream in streams))

    def test_fusion_does_not_cross_default_and_side_streams(self):
        side = torch.npu.Stream()

        def fn(x):
            default_intermediate = x + 1
            default_result = default_intermediate * 2
            with side:
                side_intermediate = x - 1
                side_result = side_intermediate * 3
            side.synchronize()
            return default_intermediate, default_result, side_intermediate, side_result

        streams = self.compile_and_check_streams(fn, torch.randn(32, device="npu"))
        self.assertIn(0, streams)
        self.assertTrue(any(stream > 0 for stream in streams))


if __name__ == "__main__":
    run_tests()
