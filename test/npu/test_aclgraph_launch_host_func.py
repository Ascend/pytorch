import unittest
from itertools import chain

import torch
from torch import nn
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

callback_stream = torch.npu.Stream()


def callback_add(params):
    global callback_stream
    with torch.npu.stream(callback_stream):
        x, y, result = params
        result.copy_(x + y)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.result = torch.rand([5, 5]).npu()

    def forward(self, graph, x, y):
        call_params = [torch.matmul(x, y), torch.matmul(x, y), self.result]
        for _ in range(10000):
            torch_npu.npu._launch_host_func(torch.npu.current_stream(), callback_add, call_params)
        return self.result


class TestAclgraphLaunchHostFunc(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_launch_host_func(self):
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.set_device(0)

        self.capture_stream = torch_npu.npu.Stream()
        self.graph = torch_npu.npu.NPUGraph()
        
        torch_npu.npu._subscribe_report(self.capture_stream)
        a = torch.randn([5, 5]).npu()
        b = torch.randn([5, 5]).npu()
        model = MyModel()
        with torch_npu.npu.stream(self.capture_stream):
            with torch_npu.npu.graph(self.graph, stream=self.capture_stream):
                self.res = model.forward(self.graph, a, b)

            torch.npu.synchronize()
            for _ in range(5):
                self.graph.replay()
                torch.npu.synchronize()
        real = torch.matmul(a, b) + torch.matmul(a, b)
        self.assertEqual(self.res.cpu(), real.cpu())
        torch_npu.npu._unsubscribe_report(self.capture_stream)


if __name__ == '__main__':
    run_tests()