# Owner(s): ["module: dynamo"]
import functools
import unittest

import torch

import torch_npu
import torch_npu.testing
import torchair

import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.backends.debugging import ExplainWithBackend
from torch._dynamo.backends.onnxrt import has_onnxruntime
from torch._dynamo.backends.tvm import has_tvm
from torch._dynamo.testing import same
from torch.testing._internal.inductor_utils import HAS_CUDA

RUN_NPU = torch.npu.is_available()
requires_npu = functools.partial(unittest.skipIf, not RUN_NPU, "requires npu")


class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TestOptimizations(torch._dynamo.test_case.TestCase):
    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertEqual(r1.size(), r2.size())
        self.assertEqual(r1.stride(), r2.stride())
        self.assertEqual(r1.dtype, r2.dtype)

        self.assertEqual(r1.size(), r3.size())
        self.assertEqual(r1.stride(), r3.stride())
        self.assertEqual(r1.dtype, r3.dtype)

    def test_example_inputs_runtime_use(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            def fwd(*args):
                nonlocal r1
                r = graph.forward(*args)
                r1 = r[0]
                return r

            return fwd

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def _check_backend_works(self, backend):
        model = Seq().eval()
        ipt = torch.randn(2, 10)
        r1 = model(ipt)
        r2 = torch.compile(model, backend=backend)(ipt)
        self.assertTrue(same(r1, r2.float(), tol=0.01))

    def test_eager(self):
        self._check_backend_works("eager")

    def test_torchscript(self):
        self._check_backend_works("ts")

    def test_aot_eager(self):
        self._check_backend_works("aot_eager")

    def test_aot_eager_decomp_partition(self):
        self._check_backend_works("aot_eager_decomp_partition")

    def test_aot_ts(self):
        self._check_backend_works("aot_ts")

    @requires_npu()
    def test_aot_cudagraphs(self):
        npu_backend = torchair.get_npu_backend()
        model = Seq().eval()
        ipt = torch.randn(2, 10)
        ipt_npu = torch.randn(2, 10).npu()
        r1 = model(ipt)
        r2 = torch.compile(model, backend=npu_backend)(ipt_npu)
        self.assertTrue(same(r1, r2.float(), tol=0.01))

    @unittest.skipIf(not has_onnxruntime(), "requires onnxruntime")
    def test_onnxrt(self):
        self._check_backend_works("onnxrt")

    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm(self):
        self._check_backend_works("tvm")

    def test_list_backends(self):
        self.assertIn("inductor", torch._dynamo.list_backends())
        self.assertIn("inductor", torch._dynamo.list_backends(exclude_tags=None))
        self.assertNotIn("eager", torch._dynamo.list_backends())
        self.assertNotIn("eager", torch._dynamo.list_backends(exclude_tags=["debug"]))
        self.assertIn("eager", torch._dynamo.list_backends(exclude_tags=[]))


class NormalizeIRTests(torch._dynamo.test_case.TestCase):
    def test_inplace_normalize(self):
        def fn(a, b):
            x = torch.cos(a)
            x += b
            return torch.sin(x)

        a = torch.randn(10)
        b = torch.randn(10).to(torch.float64)

        ref = fn(a, b)

        optimized_fn = torch._dynamo.optimize("aot_eager")(fn)
        res = optimized_fn(a, b)
        self.assertTrue(same(ref, res))


class MPSNotSupportedTest(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
    def test_mps_not_supported(self):
        model = Seq().to("mps")
        example_input = torch.randn(1, 10).to("mps")
        self.assertRaises(
            RuntimeError,
            lambda: torch.compile(model, backend="inductor")(example_input),
        )


class TestExplainWithBackend(torch._dynamo.test_case.TestCase):
    def test_explain_with_backend(self):
        def fn3(x):
            x = torch.sin(x)
            torch._dynamo.graph_break()
            x = torch.sin(x)
            return x

        def fn2(x):
            x = torch.cos(x)
            x = fn3(x)
            x = torch.cos(x)
            return x

        def fn1(x):
            x = torch.tan(x)
            x = fn2(x)
            x = torch.tan(x)
            return x

        def fn(x):
            x = torch.sigmoid(x)
            x = fn1(x)
            x = torch.sigmoid(x)
            return x

        # Wrap TorchInductor with explain backend
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        input_tensor = torch.randn(5)
        result = optimized_fn(input_tensor)

        # Check that fn still produces the same output when wrapped by ExplainWithBackend
        self.assertTrue(torch.allclose(result, fn(input_tensor)))

        # Verify ExplainOutput object contents, output might change but make sure these fields are present
        explain_output = eb.output()
        explain_str = str(explain_output)
        self.assertIn("Graph Count", explain_str)
        self.assertIn("Graph Break Count", explain_str)
        self.assertIn("Op Count", explain_str)
        self.assertIn("Break Reasons", explain_str)

        # Verify that for the given functions above, we report the correct number of graphs, graph breaks, and ops
        self.assertEqual(8, explain_output.graph_count)
        self.assertEqual(7, explain_output.graph_break_count)
        self.assertEqual(8, explain_output.op_count)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
