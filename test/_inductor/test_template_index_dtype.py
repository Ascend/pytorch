import contextlib
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest import mock

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.common import ArgName, SizeArg
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V

from torch_npu._inductor import select_algorithm as sa


class TestTemplateIndexDtype(unittest.TestCase):
    def make_template(self, name="mm", manual_output_buffer=None):
        template = object.__new__(sa.NPUTritonTemplate)
        template.name = name
        template.manual_output_buffer = manual_output_buffer
        template.compile_options = sa.NPUTemplateCompileOption()
        template.grid = lambda *args: (1, 1, 1)
        template.template = object()
        template.debug = False
        return template

    def test_automatic_and_explicit_selection(self):
        for name in ("mm", "bmm", "flex_attention"):
            for safe in (True, False):
                for override in (None, "tl.int32", "tl.int64"):
                    with self.subTest(name=name, safe=safe, override=override):
                        defines = StringIO()
                        expected = override or ("tl.int32" if safe else "tl.int64")
                        with mock.patch.object(
                            sa.TritonScheduling, "can_use_32bit_indexing",
                            return_value=safe,
                        ):
                            dtype = self.make_template(name)._write_index_dtype_define(
                                defines, sympy.Symbol("s0", positive=True), [], override
                            )
                        self.assertEqual(dtype, expected)
                        self.assertEqual(
                            defines.getvalue(),
                            f"INDEX_DTYPE : tl.constexpr = {expected}\n",
                        )

    def test_invalid_override(self):
        with self.assertRaisesRegex(ValueError, "Unsupported index dtype"):
            self.make_template()._write_index_dtype_define(
                StringIO(), sympy.Integer(1), [], "tl.float32"
            )

    def test_real_index_range_selection(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        for numel, stride, expected in (
            (16, 1, "tl.int32"),
            (2**31, 1, "tl.int64"),
            (16, 2**31, "tl.int64"),
        ):
            with self.subTest(numel=numel, stride=stride), V.set_graph_handler(graph):
                layout = ir.FixedLayout(
                    torch.device("cpu"), torch.float32,
                    [sympy.Integer(numel)], [sympy.Integer(stride)],
                )
                dtype = self.make_template()._write_index_dtype_define(
                    StringIO(), sympy.Integer(numel),
                    [ir.Buffer(name="input", layout=layout)],
                )
                self.assertEqual(dtype, expected)

    def test_generate_and_final_renderer(self):
        layout = ir.FixedLayout(torch.device("cpu"), torch.float32, [16])
        large = ir.Buffer(name="large", layout=layout)
        for manual in (None, "OUT"):
            for dtype in ("tl.int32", "tl.int64"):
                with self.subTest(manual=manual, dtype=dtype):
                    template = self.make_template(manual_output_buffer=manual)
                    graph = mock.Mock()
                    graph.set_current_device.return_value = contextlib.nullcontext()
                    graph.sizevars.optimization_hints.return_value = []
                    kernel = mock.MagicMock()
                    kernel.__enter__.return_value = kernel
                    kernel.args.input_buffers = {}
                    kernel.args.output_buffers = {}
                    kernel.args.sizevars = {}
                    kernel.render.return_value.finalize_all.return_value = "code"
                    module = SimpleNamespace(__file__="unused.py", key="key")
                    with (
                        V.set_graph_handler(graph),
                        mock.patch.object(sa, "NPUTritonTemplateKernel", return_value=kernel) as ctor,
                        mock.patch.object(sa.PyCodeCache, "load", return_value=module),
                        mock.patch.object(sa, "TritonCPUBenchmarkRequest"),
                        mock.patch.object(sa.TensorMeta, "from_irnodes"),
                        mock.patch.object(sa, "TritonTemplateCaller") as caller,
                    ):
                        template.generate(
                            [], layout, 1, 4, call_sizes=[32],
                            index_dtype_override=dtype,
                        )
                        caller.call_args.args[3](ir.Buffer(name="final_out", layout=layout))
                    self.assertEqual(ctor.call_count, 2)
                    for call in ctor.call_args_list:
                        self.assertEqual(call.kwargs["index_dtype_override"], dtype)
                        self.assertIn(f"INDEX_DTYPE : tl.constexpr = {dtype}", call.kwargs["defines"])
                        self.assertNotIn("index_dtype_override", call.kwargs["meta"])

        # Large auxiliary inputs remain excluded from the automatic range check;
        # a manually managed output uses call_sizes instead of layout.size.
        for manual in (None, "OUT"):
            template = self.make_template(manual_output_buffer=manual)
            checked = []

            def select(numel, buffers):
                checked.append((numel, list(buffers)))
                return False

            with (
                mock.patch.object(sa.TritonScheduling, "can_use_32bit_indexing", side_effect=select),
                V.set_graph_handler(mock.MagicMock()),
                mock.patch.object(sa, "NPUTritonTemplateKernel", side_effect=RuntimeError("stop before rendering")),
                self.assertRaisesRegex(RuntimeError, "stop before rendering"),
            ):
                template.generate(
                    [large], layout, 1, 4, call_sizes=[32], large_input_buffers=[large]
                )
            self.assertEqual(checked[0][0], 32 if manual else 16)
            self.assertEqual(len(checked[0][1]), 0 if manual else 1)
            self.assertNotIn(large, checked[0][1])

    def test_runtime_renderer_and_symbolic_signature(self):
        layout = ir.FixedLayout(torch.device("cpu"), torch.float32, [16])
        for name in ("mm", "flex_attention"):
            for safe in (True, False):
                for override in (None, "tl.int32", "tl.int64"):
                    with self.subTest(name=name, safe=safe, override=override):
                        dtype = override or ("tl.int32" if safe else "tl.int64")
                        with (
                            mock.patch.object(sa.TritonScheduling, "can_use_32bit_indexing", return_value=safe),
                            mock.patch.object(sa, "NPUTritonTemplateKernel") as ctor,
                        ):
                            factory = self.make_template(name).make_runtime_renderer_factory(
                                input_nodes=[], runtime_args=[], layout=layout,
                                num_stages=1, num_warps=4, index_dtype_override=override,
                            )
                            factory(ir.Buffer(name="out", layout=layout))
                        options = ctor.call_args.kwargs
                        self.assertEqual(options["index_dtype_override"], dtype)
                        self.assertIn(f"INDEX_DTYPE : tl.constexpr = {dtype}", options["defines"])
                        self.assertNotIn("index_dtype_override", options["meta"])

                        # Exercise the inherited property and real signature code
                        # with two symbolic sizes, without a device or a launch.
                        kernel = object.__new__(sa.NPUTritonTemplateKernel)
                        kernel._index_dtype_override = options["index_dtype_override"]
                        signature = [SizeArg(f"ks{i}", sympy.Symbol(f"s{i}", positive=True)) for i in range(2)]
                        kernel.args = mock.Mock()
                        kernel.args.python_argdefs.return_value = (
                            [ArgName(f"ks{i}") for i in range(2)], [], signature, []
                        )
                        kernel.use_jit = False
                        kernel.output_node = ir.Buffer(name="out", layout=layout)
                        kernel.meta = {}
                        kernel.compile_option_keys = frozenset()
                        kernel.reset_to_zero_arg_names = None
                        kernel.num_stages = 1
                        kernel.num_warps = 4
                        with (
                            V.set_graph_handler(mock.Mock()),
                            sa.config.patch({"profile_bandwidth": False, "benchmark_kernel": False}),
                            mock.patch.object(DeviceProperties, "create", return_value=None),
                            mock.patch.object(TritonKernel, "inductor_meta_common", return_value={}),
                            mock.patch("torch._inductor.codegen.triton_utils.config_of", return_value={}),
                        ):
                            kernel.jit_lines()
                        meta = kernel.triton_meta["signature"]
                        self.assertEqual(set(meta.values()), {"i32" if dtype == "tl.int32" else "i64"})


if __name__ == "__main__":
    unittest.main()
