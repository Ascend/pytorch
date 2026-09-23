# Owner(s): ["module: dynamo"]

"""Regression tests for NPU Stream/Event routing through Dynamo."""

import torch
import torch._dynamo.test_case
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch.testing._internal.common_utils import run_tests


class TraceStreamEventTests(torch._dynamo.test_case.TestCase):
    def test_npu_stream_event_in_graph_classes(self):
        # Trigger torch_npu's lazy Dynamo setup before reading the cached set.
        torch.compile(lambda x: x + 1, backend="eager", fullgraph=True)(
            torch.ones(1)
        )
        in_graph_classes = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(torch.npu.Stream, in_graph_classes)
        self.assertIn(torch.npu.Event, in_graph_classes)

    def test_dynamo_trace_stream_event(self):
        def my_backend(gm, example_inputs):
            node_names = (node.name for node in gm.graph.nodes)
            self.assertIn("current_stream", node_names)
            self.assertIn("set_stream", node_names)
            self.assertIn("record_stream", node_names)
            return gm

        @torch.compile(backend=my_backend, fullgraph=True)
        def test_stream_in_graph(a):
            s = torch.npu.Stream()
            event = torch.npu.Event()
            r = torch.add(a, 2)
            event.record()
            with torch.npu.stream(s):
                event.wait()
                r = torch.add(r, 1)
                r.record_stream(s)
            r = torch.add(r, 1)
            return r

        i = torch.randn([3, 3], device="npu:0")
        r = test_stream_in_graph(i)
        self.assertEqual(r, i + 4)


if __name__ == "__main__":
    run_tests()
