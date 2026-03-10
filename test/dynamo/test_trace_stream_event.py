import unittest
import torch
import torch_npu


class TraceStreamEventTests(unittest.TestCase):
    def test_dynamo_trace_stream_event(self):

        def my_backend(gm, example_inputs):
            print(gm.graph)
            node_names = (node.name for node in gm.graph.nodes)
            self.assertIn("current_stream", node_names)
            self.assertIn("set_stream", node_names)
            self.assertIn("fake_record_stream", node_names)
            return gm        

        @torch.compile(backend=my_backend)
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
        with self.assertRaises(RuntimeError) as context:            
            r = test_stream_in_graph(i)
            return r
        self.assertIn("tensor.record_stream is not supported on torch.compile", str(context.exception))


if __name__ == '__main__':
    unittest.main()