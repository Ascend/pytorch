import ast
from types import SimpleNamespace
import unittest

import sympy
import torch

from torch._inductor import config
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_npu._inductor.codegen.triton import (
    IterationRangesEntryNPUIndex,
    NPUIndexTritonKernel,
)
from torch_npu._inductor.codegen.triton_utils import NPUKernelType


class TestReductionMaskCodegen(TestCase):
    def make_kernel(self, length, kernel_type=NPUKernelType.SIMD_SIMT_MIX):
        kernel = object.__new__(NPUIndexTritonKernel)
        kernel.optimize_mask = True
        kernel.persistent_reduction = True
        kernel.npu_kernel_type = kernel_type
        kernel.current_subblock_axis = set()
        kernel._load_mask = None
        symbol = sympy.Symbol("r1", integer=True)
        code = IndentedBuffer()
        node = SimpleNamespace(
            name="r1", length=length, is_reduction=True,
            is_tiling_axis=True, is_split_axis=False, is_no_loop_axis=False,
            is_vectorized_split=False, symbol=lambda: symbol,
            var_directions=[SimpleNamespace(name="r1_1")], writeline=code.writeline,
        )
        kernel.sorted_axis = [node]
        kernel.range_tree_nodes = {symbol: node}
        kernel.range_trees = [SimpleNamespace(var_list=[symbol])]
        return kernel, node, code

    def test_mask_proof_uses_emitted_block_size(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        for kernel_type in NPUKernelType:
            for length in (1, 2, 31, 32, 197, 256):
                with self.subTest(kernel_type=kernel_type, length=length):
                    kernel, node, code = self.make_kernel(length, kernel_type)
                    unified = kernel_type in (NPUKernelType.SIMT_ONLY, NPUKernelType.SIMD_SIMT_MIX)
                    block = 1 << (length - 1).bit_length() if unified else length
                    with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
                        constants = IndentedBuffer()
                        kernel.codegen_static_numels(constants)
                        IterationRangesEntryNPUIndex._codegen_mask(node)
                        masks = {"r1_mask", "r1_1_mask"}
                        kernel.filter_masks(masks)
                    self.assertIn(f"R1BLOCK_SUB: tl.constexpr = {block}", constants.getvalue())
                    if length == block:
                        self.assertEqual(masks, set())
                        self.assertEqual(code.getvalue().splitlines(), ["r1_mask = True", "r1_1_mask = True"])
                    else:
                        self.assertEqual(masks, {"r1_mask", "r1_1_mask"})
                        self.assertIn("r1_mask = r1 < r1_numel", code.getvalue())

    def test_full_reduction_only_drops_output_value_masks(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        kernel, _, _ = self.make_kernel(32)
        for name in ("x0", "y2"):
            symbol = sympy.Symbol(name, integer=True)
            kernel.sorted_axis.append(SimpleNamespace(is_reduction=False, symbol=lambda s=symbol: s))
        masks = ["x0_mask", "y2_mask", "tmp0"]
        with V.set_graph_handler(graph):
            self.assertEqual(kernel._reduction_value_masks(masks[:2]), [])
            self.assertEqual(kernel._reduction_value_masks(masks), ["tmp0"])
            kernel._load_mask = "x0_mask"
            self.assertEqual(kernel._reduction_value_masks(masks), ["x0_mask", "tmp0"])
        self.assertEqual(masks, ["x0_mask", "y2_mask", "tmp0"])

    def test_unproven_reductions_keep_output_value_masks(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        symbolic_length = sympy.Symbol("s0", positive=True, integer=True)
        for case in ("tail", "dynamic", "loop", "disabled", "vectorized", "multi_axis"):
            with self.subTest(case=case):
                kernel, node, _ = self.make_kernel(31 if case == "tail" else 32)
                if case == "dynamic":
                    node.length = symbolic_length
                elif case == "loop":
                    kernel.persistent_reduction = False
                elif case == "disabled":
                    kernel.optimize_mask = False
                elif case == "vectorized":
                    node.is_vectorized_split = True
                elif case == "multi_axis":
                    kernel.range_trees[-1].var_list.append(sympy.Symbol("r2", integer=True))
                masks = ["x0_mask", "r1_mask", "tmp0"]
                with V.set_graph_handler(graph):
                    self.assertEqual(kernel._reduction_value_masks(masks), masks)


class TestRangeSplitCompatibility(TestCase):
    def test_incompatible_ranges_return_false(self):
        for groups, lengths in (
            ((6, 1), ([4], [1])),
            ((2, 3), ([10], [])),
            ((2,), ([6], [])),
            ((2,), ([2, 3], [])),
        ):
            with self.subTest(groups=groups, lengths=lengths):
                with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
                    self.assertFalse(NPUIndexTritonKernel.is_compatible(groups, lengths))

    def test_three_group_index_reconstruction(self):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            ranges, getters = NPUIndexTritonKernel._split_iteration_ranges((2, 3, 5), ([30], []))
        self.assertEqual(ranges, [[2], [3], [5]])
        indices = sympy.symbols("i0:3", integer=True, nonnegative=True)
        self.assertEqual(sympy.expand(getters[0][0](indices)), 15 * indices[0] + 5 * indices[1] + indices[2])


class TestReductionValueMaskDevice(TestCase):
    @unittest.skipIf(not torch.npu.is_available(), "requires NPU")
    def test_full_reduction_keeps_memory_masks_and_tail_identity(self):
        def fn(x):
            maximum = x.amax(0, keepdim=True)
            return maximum, (x - maximum).exp().sum(0, keepdim=True)

        for length, dynamic in ((32, False), (31, False), (31, True)):
            with self.subTest(length=length, dynamic=dynamic), config.patch({"fx_graph_cache": False}):
                cpu = -torch.arange(1, 1 + length * 31, dtype=torch.float32).reshape(length, 31)
                compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
                actual, codes = run_and_get_code(compiled, cpu.npu())
                self.assertEqual(tuple(x.cpu() for x in actual), fn(cpu), atol=1e-5, rtol=1e-5)
                calls = []
                for code in codes:
                    for node in ast.walk(ast.parse(code)):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "async_compile"
                            and node.func.attr == "triton"
                        ):
                            source = ast.literal_eval(node.args[1])
                            calls.extend(
                                n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                            )
                for op in ("tl.load", "tl.store"):
                    memory_ops = [n for n in calls if ast.unparse(n.func) == op]
                    self.assertTrue(memory_ops)
                    self.assertTrue(all("_mask" in ast.unparse(n) for n in memory_ops))
                value_masks = [
                    n for n in calls
                    if ast.unparse(n.func) == "tl.where" and "_mask" in ast.unparse(n.args[0])
                ]
                self.assertEqual(bool(value_masks), dynamic)



if __name__ == "__main__":
    run_tests()
