# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch_npu
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm, same

from torch.nn import functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION


class CutomizedCtxManager:
    def __init__(self, mode):
        self.prev = torch.is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_value, traceback):
        torch._C._set_grad_enabled(self.prev)


class CtxManagerTests(torch._dynamo.test_case.TestCase):
    def test_no_grad(self):
        def fn1(a, b):
            x = a + 1
            # redundant no_grad should get ignored
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with torch.no_grad():
            torch._dynamo.testing.standard_test(
                self, fn=fn1, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn2, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(
                self, fn=fn3, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn4, nargs=2, expected_ops=3
            )  # coalesced noop

    def test_grad_mode_guard(self):
        def fn(a, b):
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            a = a + 1
            a.tolist()  # graph break
            ret = a + b
            torch.set_grad_enabled(prev_grad)
            return ret

        a = torch.randn([3, 4])
        b = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for _ in range(10):
            opt_fn(a, b)
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        def fn(x):
            before = torch.is_grad_enabled()
            with torch.set_grad_enabled(False):
                torch._dynamo.graph_break()
                with torch.set_grad_enabled(True):
                    x = torch.mul(x, 5)
                    torch._dynamo.graph_break()
                    x = torch.sqrt(x)
                    assert torch.is_grad_enabled()
                assert not torch.is_grad_enabled()
            assert torch.is_grad_enabled() == before
            return x

        a = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        for _ in range(10):
            opt_fn(a)
        self.assertEqual(cnts.frame_count, 2)

    def test_torch_profiler(self):
        # wrap torch.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.profiler.profile():
                y = y + 2
                with torch.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # wrap torch.autograd.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.autograd.profiler.profile():
                y = y + 2
                with torch.autograd.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_npu_stream_context_manager1(self):
        def fn(x):
            s = torch.npu.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            with torch.npu.stream(s):
                x = torch.relu(x)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="npu:0")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_npu_stream_context_manager2(self):
        def fn(x, s):
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            with torch.npu.stream(s):
                x = torch.relu(x)

            s1 = torch.npu.current_stream()
            with torch.npu.stream(s1):
                x = torch.relu(x)

            s2 = torch.npu.Stream()
            with torch.npu.stream(s2):
                x = torch.relu(x)

            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="npu:0")
        s = torch.npu.Stream()
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_npu_stream_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            new_stream = torch.npu.Stream()
            with torch.npu.stream(new_stream):
                x = torch.sin(x)
                x = torch.add(x, 3)

            cur_stream = torch.npu.current_stream()
            cur_stream.wait_stream(new_stream)

            x = torch.add(x, 4)
            is_idle = cur_stream.query()
            cur_stream.synchronize()

            with torch.npu.stream(new_stream):
                x = torch.add(x, 5)
            new_stream.synchronize()

            is_equal = cur_stream == new_stream

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="npu:0")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 20)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu:0")
    def test_npu_event_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.npu.current_stream()
            new_stream = torch.npu.Stream()

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            is_idle = event.query()

            new_stream.wait_event(event)
            with torch.npu.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.npu.Event()
            new_event.record(new_stream)

            x = torch.add(x, 5)
            new_event.wait(cur_stream)

            # use new event to sync
            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="npu:0")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 19)

    def test_autograd_profiler_enabled(self):
        def fn(x):
            if torch.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        x = torch.randn((2, 2), requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        if torch.autograd._profiler_enabled():
            torch.autograd._disable_profiler()
        assert not torch.autograd._profiler_enabled()
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

        with torch.autograd.profiler.profile():
            assert torch.autograd._profiler_enabled()
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_autocast(self):
        if not torch.npu.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="npu")
                b_float32 = torch.rand((8, 8), device="npu")
                d_float32 = torch.rand((8, 8), device="npu")

                with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "npu")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.bfloat16)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_npu_amp_autocast(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="npu")
                b_float32 = torch.rand((8, 8), device="npu")

                with torch.npu.amp.autocast(dtype=torch.torch.float64):
                    c_float64 = torch.mm(a_float32, b_float32)
                return c_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "npu")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    def test_is_autocast_cpu_enabled(self):
        def fn(a_float32, b_float32):
            with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                c_float16 = torch.mm(a_float32, b_float32)
                if torch.is_autocast_cpu_enabled():
                    c_float16 = c_float16 + 1
            return c_float16

        a = torch.rand((8, 8))
        b = torch.rand((8, 8))
        ref = fn(a, b)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                with torch.autocast("cpu"):
                    with torch.autocast("npu", dtype=torch.float32):
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.0, True
                        )
                return out

        dtype = torch.float32
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device="npu", dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device="npu", dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device="npu", dtype=dtype, requires_grad=True
        )

        module = MyModule()
        real = module(query, key, value)
        real_device = real.device
        real_dtype = real.dtype

        opt_mod = torch._dynamo.optimize("inductor")(module)
        compiled = opt_mod(query, key, value)

        self.assertEqual(compiled.device, real_device)
        self.assertEqual(compiled.dtype, real_dtype)

        self.assertEqual(compiled.device.type, "npu")
        self.assertEqual(compiled.device.index, 0)
        self.assertEqual(compiled.dtype, torch.float32)

    def test_autocast_cpu(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cpu")
        self.assertEqual(exported.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                torch._dynamo.graph_break()
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    torch._dynamo.graph_break()
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        opt = torch._dynamo.optimize("eager")(module)
        res = opt(torch.tensor([0.5]))
        self.assertEqual(res.device, real_device)
        self.assertEqual(res.dtype, real_dtype)

        self.assertEqual(res.device.type, "cpu")
        self.assertEqual(res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # Regression for: See pytorch/pytorch/issues/93890
        def fn(x):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                x = torch.mm(x, x)
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x

        x = torch.rand([4, 4])
        self.assertEqual(x.dtype, torch.float32)
        res = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_res = opt_fn(x)
        self.assertTrue(torch.allclose(res, opt_res))
        self.assertEqual(res.dtype, torch.bfloat16)
        self.assertEqual(opt_res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_inner_fn(self):
        class MyModule(torch.nn.Module):
            @staticmethod
            def mm_breaks(x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y)

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    torch._dynamo.graph_break()
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        torch._dynamo.graph_break()
                        g_float32 = torch.mm(a_float32, b_float32)
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            # Check that nested with non-inlineable function with graph break
                            torch._dynamo.graph_break()
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # We remember to exit the inner autocast correctly to outer
                    # even after graph breaks
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    assert f_float16.dtype == f_float16_1.dtype
                return f_float16, g_float32

        module = MyModule()
        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    def test_autocast_graph_break_method(self):
        class MyModule(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def mm_not_break(self, x, y):
                return torch.mm(x, y) + self.bias

            def mm_breaks(self, x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y) + self.bias

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        g_float32 = torch.mm(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    assert (
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                return f_float16, g_float32

        module = MyModule(bias=torch.rand((8, 8), device="cpu", dtype=torch.bfloat16))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # Autocast doesn't work on addition, so we need the bias to be `bfloat16`
            res = torch.rand((8, 8), device="cpu", dtype=torch.float32) + torch.rand(
                (8, 8), device="cpu", dtype=torch.bfloat16
            )
            self.assertEqual(res.dtype, torch.float32)

        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_autocast_float64(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="npu")
                b_float32 = torch.rand((8, 8), device="npu")
                d_float32 = torch.rand((8, 8), device="npu")

                with torch.autocast(device_type="npu", dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_autocast_device(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="npu")
                b_float32 = torch.rand((8, 8), device="npu")
                d_float32 = torch.rand((8, 8), device="npu")

                with torch.autocast("npu"):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.torch.float16)

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_autocast_arguments_binding(self):
        def f1(x):
            with torch.npu.amp.autocast(False):
                x = torch.sin(x + 1)
            return x

        def f2(x):
            with torch.cpu.amp.autocast(False):
                x = torch.cos(x + 1)
            return x

        x = torch.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)
        opt_f1 = torch.compile(backend="eager")(f1)
        opt_f2 = torch.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))

    @unittest.skipIf(not torch.npu.is_available(), "requires npu")
    def test_autocast_decorator(self):
        def autocast_func(orig_func):
            @torch.amp.autocast(device_type="npu", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_npu(orig_func):
            @torch.npu.amp.autocast(dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cpu(orig_func):
            @torch.cpu.amp.autocast(dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def mm(a, b):
            return torch.mm(a, b)

        mm_float16 = autocast_func(mm)
        mm_float16_npu = autocast_func_npu(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        def fn(a, b):
            return mm_float16(a, b), mm_float16_npu(a, b), mm_float16_cpu(a, b)

        a_float32 = torch.rand((8, 8), device="npu")
        b_float32 = torch.rand((8, 8), device="npu")

        ref = fn(a_float32, b_float32)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(a_float32, b_float32)
        self.assertTrue(same(ref, res))
        self.assertTrue(res[0].dtype == torch.float16)
        self.assertTrue(res[1].dtype == torch.float16)

    def test_generic_context_manager(self):
        def fn(x):
            with CutomizedCtxManager(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                x = torch.relu(x)
            return x - 1

        with torch.no_grad():
            torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=6)

        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=6)

    def test_nested_generic_context_manager(self):
        def fn(x):
            with CutomizedCtxManager(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with CutomizedCtxManager(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        with torch.no_grad():
            torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=9)

        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=9)

    def test_generic_context_manager_with_graph_break(self):
        def fn(x):
            with CutomizedCtxManager(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 2)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 4)
            self.assertEqual(cnts.op_count, 4)

    def test_nested_generic_context_manager_with_graph_break(self):
        def fn(x):
            with CutomizedCtxManager(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with CutomizedCtxManager(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    torch._dynamo.graph_break()
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 4)
            self.assertEqual(cnts.op_count, 4)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 4)
            self.assertEqual(cnts.op_count, 4)

    def test_graph_break_inlining_grad(self):
        def gn(z):
            with torch.no_grad():
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y, z):
            a = torch.mm(x, y)
            z = gn(z)
            return a

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4)
        opt_fn(x, y, z).sum().backward()

        self.assertEqual(cnts.frame_count, 2)

    def _graph_break_inlining_autocast_test_helper(self, device):
        def gn(x, y):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z = torch.mm(x, y)
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y):
            z = torch.mm(x, y)
            z = z + gn(x, y)
            return z

        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_graph_break_inlining_autocast(self):
        for device in ["npu", "cpu"]:
            if device == "npu" and not (
                torch.npu.is_available() and torch.npu.is_bf16_supported()
            ):
                continue
            self._graph_break_inlining_autocast_test_helper(device)

    def test_disable_saved_tensors_hooks(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        x = torch.ones(1)

        y = torch.zeros(1)

        add = x + y;  x = y = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (add,)
"""
        self.assertExpectedInline(actual, expected)

    def test_disable_saved_tensors_hooks_prev_disabled(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        x = torch.ones(1)

        y = torch.zeros(1)

        add = x + y;  x = y = None

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message')
        return (add,)
"""
        self.assertExpectedInline(actual, expected)

    def test_disable_saved_tensors_hooks_prev_disabled_nested(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                @torch.autograd.graph.disable_saved_tensors_hooks(
                    "This is not supported inner"
                )
                def inner_fn(x, y):
                    return x + y

                return inner_fn(x, y) + x

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        x = torch.ones(1)

        y = torch.zeros(1)

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported inner')

        add = x + y;  y = None

        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        add_1 = add + x;  add = x = None

        _saved_tensors_hooks_disable_3 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message')
        return (add_1,)
"""
        self.assertExpectedInline(actual, expected)

    def test_disable_saved_tensors_hooks_graph_break(self):
        def fn(x):
            with torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ):
                y = x + 1
                torch._dynamo.graph_break()
                return y * 2

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=False)(torch.randn(()))

        def check_graph(actual, expected):
            self.assertExpectedInline(actual, expected)

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        y = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (y,)
"""
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        check_graph(actual, expected)

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_ : torch.Tensor):
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported')

        mul = l_y_ * 2;  l_y_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (mul,)
"""
        graph = eager.graphs[1]
        actual = normalize_gm(graph.print_readable(False))
        check_graph(actual, expected)

    def test_context_wrapping_grad_mode_decorator(self):
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]
        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, mode = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    def inner_func(x):
                        return x.sin()

                    with ctx_wrapper_inverse():
                        if call:
                            inner_func = ctx_wrapper()(inner_func)
                        else:
                            inner_func = ctx_wrapper(inner_func)

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_grad_mode_nested_function_decorator(self):
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]

        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, mode = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    with ctx_wrapper_inverse():
                        if call:

                            @ctx_wrapper()
                            def inner_func(x):
                                return x.sin()

                        else:

                            @ctx_wrapper
                            def inner_func(x):
                                return x.sin()

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_set_grad_enabled_nested_function(self):
        modes = [True, False]
        for decorator in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                mode = modes[i]
                mode_inverse = modes[(i + 1) % 2]

                def fn(x):
                    with torch.set_grad_enabled(mode_inverse):
                        if decorator:

                            @torch.set_grad_enabled(mode)
                            def inner_func(x):
                                return x.sin()

                        else:

                            def inner_func(x):
                                return x.sin()

                            inner_func = torch.set_grad_enabled(mode)(inner_func)

                        # Consuming set_grad_enabled by calling it on a function
                        # should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with torch.set_grad_enabled(mode_inverse):
                        return inner_func(x)

            x = torch.zeros(10, requires_grad=True)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))
            self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
