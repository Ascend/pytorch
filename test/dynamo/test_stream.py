# Owner(s): ["module: dynamo"]
import functools
import unittest
import torch
import torch._dynamo.test_case
import torch_npu

requires_npu = functools.partial(unittest.skipIf, not torch.npu.is_available(), "requires npu")


class StreamintoDynamoTests(torch._dynamo.test_case.TestCase):

    @requires_npu()
    def test_stream(self):
        def model_1(x):
            a = x * x
            s = torch.npu.Stream()
            s.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(s):
                b = x + a
            return b
        inp = torch.randn(2, 8).npu()
        m = torch.compile(model_1, backend="aot_eager", fullgraph=True)
        output = m(inp)
        output1 = model_1(inp)
        torch.allclose(output, output1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
