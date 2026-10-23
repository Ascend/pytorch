from types import SimpleNamespace

import sympy
from torch._inductor.codegen.simd import CantSplit
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel


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

    def test_rejection_contains_divisibility_operands(self):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            with self.assertRaises(CantSplit) as error:
                NPUIndexTritonKernel._split_iteration_ranges((6, 1), ([4], [1]))
        self.assertEqual(error.exception.expr, 6)
        self.assertEqual(error.exception.remaining, 4)

    def test_three_group_index_reconstruction(self):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            ranges, getters = NPUIndexTritonKernel._split_iteration_ranges((2, 3, 5), ([30], []))
        self.assertEqual(ranges, [[2], [3], [5]])
        indices = sympy.symbols("i0:3", integer=True, nonnegative=True)
        self.assertEqual(sympy.expand(getters[0][0](indices)), 15 * indices[0] + 5 * indices[1] + indices[2])


if __name__ == "__main__":
    run_tests()
