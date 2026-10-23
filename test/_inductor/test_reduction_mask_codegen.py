from types import SimpleNamespace

import sympy

import torch
from torch._inductor.codegen.triton import TritonCSEVariable
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges
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

    def test_unproven_masks_are_retained(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        cases = ("dynamic", "zero", "nonpersistent", "vectorized", "disabled")
        for case in cases:
            with self.subTest(case=case):
                kernel, node, _ = self.make_kernel(256)
                if case == "dynamic":
                    node.length = sympy.Symbol("s0", positive=True, integer=True)
                elif case == "zero":
                    node.length = 0
                elif case == "nonpersistent":
                    kernel.persistent_reduction = False
                elif case == "vectorized":
                    node.is_vectorized_split = True
                else:
                    kernel.optimize_mask = False
                with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
                    masks = {"r1_mask"}
                    kernel.filter_masks(masks)
                self.assertEqual(masks, {"r1_mask"})

    def test_persistent_split_axis_covers_full_extent(self):
        from triton import Config
        from torch_npu._inductor.runtime.triton_heuristics import set_reduction_runtime_blocks_to_numel

        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        for length in (197, 256):
            with self.subTest(length=length):
                kernel, node, code = self.make_kernel(length)
                node.is_split_axis = True
                configs = [Config({"R1BLOCK": block, "split_blocks": (block,)}) for block in (1, 64, 512)]
                set_reduction_runtime_blocks_to_numel(configs, [0], ["r1"], [length], ["R1BLOCK"])
                self.assertTrue(all(cfg.kwargs["R1BLOCK"] == length for cfg in configs))
                self.assertTrue(all(cfg.kwargs["split_blocks"] == (length,) for cfg in configs))
                with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
                    masks = {"r1_mask"}
                    kernel.filter_masks(masks)
                    IterationRangesEntryNPUIndex._codegen_mask(node)
                self.assertEqual(masks, set() if length == 256 else {"r1_mask"})
                self.assertIn("r1_mask = True" if length == 256 else "min(R1BLOCK+r1_offset", code.getvalue())

    def test_semantic_mask_survives_axis_mask_elimination(self):
        graph = SimpleNamespace(sizevars=SizeVarAllocator())
        kernel, _, _ = self.make_kernel(256)
        semantic_mask = TritonCSEVariable("semantic_mask", ValueRanges.unknown(), torch.bool, (256,))
        with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
            masks = {"r1_mask", semantic_mask}
            kernel.filter_masks(masks)
        self.assertEqual(masks, {semantic_mask})


if __name__ == "__main__":
    run_tests()
