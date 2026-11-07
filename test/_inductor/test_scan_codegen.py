import unittest
from types import SimpleNamespace

import sympy

import torch
from torch._inductor.codegen.common import CSE
from torch._inductor.codegen.triton import HelperFunctions, TritonCSEVariable
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges

from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel
from torch_npu._inductor.codegen.triton_utils import NPUKernelType


class TestScanCodegen(TestCase):
    @staticmethod
    def make_kernel(names, persistent, length=197, split=False):
        kernel = object.__new__(NPUIndexTritonKernel)
        kernel.inside_reduction = True
        kernel.cooperative_reduction = False
        kernel.persistent_reduction = persistent
        kernel.optimize_mask = True
        kernel.npu_kernel_type = NPUKernelType.SIMD_SIMT_MIX
        kernel.current_subblock_axis = set()
        kernel.current_node = None
        kernel._load_mask = None
        kernel.cse = CSE()
        kernel.compute = IndentedBuffer()
        kernel.body = IndentedBuffer()
        kernel.helper_functions = HelperFunctions()
        symbols = [sympy.Symbol(name, integer=True) for name in names]
        kernel.golden_var_list = tuple(reversed(symbols))
        kernel.sorted_axis = [
            SimpleNamespace(
                name=symbol.name,
                symbol=lambda symbol=symbol: symbol,
                length=length if symbol.name.startswith("r") else 31,
                is_reduction=symbol.name.startswith("r"),
                is_split_axis=split and symbol.name.startswith("r"),
                is_tiling_axis=True,
                is_no_loop_axis=False,
                is_vectorized_split=False,
            )
            for symbol in symbols
        ]
        kernel.range_tree_nodes = dict(zip(symbols, kernel.sorted_axis))
        # Root names intentionally differ from the NPU masks emitted per axis.
        kernel.range_trees = [
            SimpleNamespace(prefix="x", mask_name=lambda: "xmask"),
            SimpleNamespace(prefix="r", mask_name=lambda: "rmask"),
        ]
        sizes = [f"{name.upper()}BLOCK_SUB" for name in names]
        kernel.dense_size_list = lambda: sizes.copy()
        kernel.dense_size_str = lambda: f"[{', '.join(sizes)}]"
        kernel.triton_tensor_ndim = lambda: len(sizes)
        return kernel

    def run_scan(self, kernel, dtype=torch.float32, shape=None):
        shape = shape if shape is not None else tuple(kernel.dense_size_list())
        value = TritonCSEVariable("input_value", ValueRanges.unknown(), dtype, shape)
        graph = SimpleNamespace(
            sizevars=SizeVarAllocator(),
            get_current_device_or_throw=lambda: torch.device("npu"),
        )

        def add(left, right):
            return kernel.cse.generate(
                kernel.compute,
                f"{left} + {right}",
                dtype=left.dtype,
                shape=get_broadcasted_shape(left.shape, right.shape),
            )

        def combine(left, right):
            return (V.ops.add(left[0], right[0]),)

        with (
            V.set_graph_handler(graph),
            V.set_kernel_handler(kernel),
            V.set_ops_handler(SimpleNamespace(add=add)),
        ):
            return kernel.scan((dtype,), combine, (value,))[0]

    def test_scan_broadcasts_shapes_and_upcasts_values(self):
        for names in (("r0", "x1"), ("x0", "r1")):
            for dtype in (torch.float16, torch.float32, torch.int64):
                with self.subTest(names=names, dtype=dtype):
                    kernel = self.make_kernel(names, persistent=True)
                    result = self.run_scan(kernel, dtype, shape=("1", "1"))
                    self.assertEqual(tuple(result.shape), tuple(kernel.dense_size_list()))
                    self.assertEqual(result.dtype, torch.float32 if dtype == torch.float16 else dtype)
                    dim = next(i for i, name in enumerate(names) if name.startswith("r"))
                    self.assertIn(f"), {dim}, _triton_helper_fn_add", kernel.compute.getvalue())
                    # The real upstream helper requires CSE values, not a count.
                    self.assertIn("arg0_0 + arg1_0", kernel.helper_functions[0])

    def test_loop_carried_scan_uses_the_actual_reduction_dimension(self):
        for names in (("r0", "x1"), ("x0", "r1"), ("x0", "r1", "x2")):
            for split in (False, True):
                with self.subTest(names=names, split=split):
                    kernel = self.make_kernel(names, persistent=False, split=split)
                    result = self.run_scan(kernel)
                    dim = next(i for i, name in enumerate(names) if name.startswith("r"))
                    axis = names[dim]
                    dense_shape = kernel.dense_size_list()
                    reduced_shape = dense_shape.copy()
                    reduced_shape[dim] = "1"
                    select_line = next(
                        line for line in kernel.compute.getvalue().splitlines()
                        if "triton_helpers.select_one" in line
                    )
                    partial_reduce = kernel.cse.varname_map[select_line.split(" = ", 1)[0]]
                    self.assertEqual(tuple(partial_reduce.shape), tuple(reduced_shape))
                    self.assertEqual(tuple(result.shape), tuple(dense_shape))
                    self.assertIn(f"dim={dim}, keep_dims=True", select_line)
                    self.assertIn(f"tl.full([{', '.join(reduced_shape)}]", kernel.body.getvalue())
                    offset = f"loop_{axis} * {axis.upper()}BLOCK_SUB"
                    if split:
                        offset = f"{axis}_offset + ({offset})"
                    self.assertIn(offset, kernel.compute.getvalue())

    def test_scan_results_keep_required_npu_axis_masks(self):
        for persistent in (False, True):
            for length in (197, 256, sympy.Symbol("s0", positive=True, integer=True)):
                with self.subTest(persistent=persistent, length=length):
                    kernel = self.make_kernel(("x0", "r1"), persistent, length)
                    result = self.run_scan(kernel)
                    expected = {"x0_mask"}
                    if not persistent or length != 256:
                        expected.add("r1_mask")
                    self.assertEqual(set(result.mask_vars), expected)


@unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
class TestCumsumNPU(TestCase):
    def check_compiled(self, fn, args, expected):
        try:
            actual, codes = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), *args
            )
            torch.npu.synchronize()
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
            code = "\n".join(
                line for source in codes for line in source.splitlines()
                if not line.lstrip().startswith("#")
            )
            self.assertIn("tl.associative_scan", code)
            self.assertNotIn("torch.ops.aten.cumsum.", code)
        finally:
            torch._dynamo.reset()

    def test_non_power_of_two_scan_on_both_axes(self):
        for dim, shape in ((0, (197, 31)), (1, (31, 197))):
            with self.subTest(dim=dim):
                value = torch.randn(shape, device="npu")
                self.check_compiled(
                    lambda x: torch.cumsum(x, dim),
                    (value,),
                    torch.cumsum(value.cpu(), dim),
                )

    def test_integer_default_and_explicit_dtype(self):
        value = torch.randint(0, 4, (31, 197), dtype=torch.int32, device="npu")
        for dtype in (None, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.check_compiled(
                    lambda x: torch.cumsum(x, 1, dtype=dtype),
                    (value,),
                    torch.cumsum(value.cpu(), 1, dtype=dtype),
                )

    def test_out_variant(self):
        value = torch.randn((31, 197), device="npu")
        output = torch.empty_like(value)
        self.check_compiled(
            lambda x, out: torch.cumsum(x, 1, out=out),
            (value, output),
            torch.cumsum(value.cpu(), 1),
        )


if __name__ == "__main__":
    run_tests()
