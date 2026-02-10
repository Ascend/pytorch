import unittest
import torch
import torch_npu


class TraceStreamEventTests(unittest.TestCase):
    def test_dynamo_trace_stream_event(self):

        def cus_func(t):
            s = torch.npu.Stream()
            tmp = torch.add(t, 2)
            event = torch.npu.Event()        
            event.record()
            with torch.npu.stream(s):
                event.wait(s)
                r = torch.relu(tmp)
                r.record_stream(s)
            return r

        def my_backend(gm, example_inputs):
            graph = gm.graph
            print(graph)
            fx_target_list = (node.target for node in graph.nodes)
            assert_target_list = (torch_npu.npu.streams.Stream, 
                                torch_npu.npu.streams.Event, 
                                "record",
                                "wait",
                                torch_npu.utils._dynamo.fake_record_stream)
            for target in assert_target_list:
                self.assertIn(target, fx_target_list)
            return gm   
             
        opt_m = torch.compile(cus_func, backend=my_backend, fullgraph=True, dynamic=False)
        i = torch.randn([3, 3], device="npu:0")
        with self.assertRaises(RuntimeError) as context:
            r = opt_m(i)
        self.assertIn("tensor.record_stream is not supported on torch.compile", str(context.exception))


if __name__ == '__main__':
    unittest.main()