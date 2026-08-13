import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from sympy import Symbol
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch_npu._inductor.codegen.ir import analyze_floordiv_expression, analyze_modular_expression
from torch._inductor.virtualized import V
from unittest.mock import MagicMock


class MockRangeNode:
    def __init__(self, length):
        self.length = length


def make_range_tree_nodes(mapping):
    """mapping: {symbol: length}"""
    return {sym: MockRangeNode(length) for sym, length in mapping.items()}


class TestLinearDynamic(TestUtils):

    def test_analyze_floordiv_expression(self):
        x0 = Symbol("x0", integer=True, nonnegative=True)
        x1 = Symbol("x1", integer=True, nonnegative=True)
        s0 = Symbol("s0", integer=True, positive=True)
        s1 = Symbol("s1", integer=True, positive=True)
        s2 = Symbol("s2", integer=True, positive=True)

        nodes = make_range_tree_nodes({x0: s0, x1: s1})
        expr = FloorDiv(s1 * x0 + x1, s0 * s1)
        res = analyze_floordiv_expression(expr, nodes)
        print("\n[Case3] symbolic max contains divisor symbols → remainder 0 path")
        print(res)
        assert "can_split" in res and res["can_split"]


    def test_analyze_modular_expression(self):
        x0 = Symbol("x0", integer=True, nonnegative=True)
        x1 = Symbol("x1", integer=True, nonnegative=True)
        s0 = Symbol("s0", integer=True, positive=True)
        s1 = Symbol("s1", integer=True, positive=True)
        s2 = Symbol("s2", integer=True, positive=True)

        nodes = make_range_tree_nodes({x0: s0, x1: s1})
        expr = ModularIndexing(s1 * x0 + x1, 1, s0 * s1)
        res = analyze_modular_expression(expr, nodes)
        print("\n[Case3] symbolic max contains divisor symbols → remainder 0 path")
        print(res)
        assert "can_split" in res and res["can_split"]


    def test_analyze_modular_expression_mod_is_symbol(self):
        x0 = Symbol("x0", integer=True, nonnegative=True)
        x1 = Symbol("x1", integer=True, nonnegative=True)
        s0 = Symbol("s0", integer=True, positive=True)
        s1 = Symbol("s1", integer=True, positive=True)
        s2 = Symbol("s2", integer=True, positive=True)
        s3 = Symbol("s3", integer=True, positive=True)
        mock_kernel = MagicMock()

        mock_kernel.symbol_range_map = {
            "s0": MagicMock(lower=1),
            "s1": MagicMock(lower=1),
            "s2": MagicMock(lower=1),
            "s3": MagicMock(lower=1),
        }

        with V.set_kernel_handler(mock_kernel):
            nodes = make_range_tree_nodes({x0: s3 * s2, x1: s2})
            expr = ModularIndexing((x0 + x1 * s3), s2, s3)
            res = analyze_modular_expression(expr, nodes)
            print("\n[Case3] symbolic max contains divisor symbols → remainder 0 path")
            print(res)
            assert "can_split" in res and res["can_split"]


    def op_calc_dynamic(self, x, y, batch_size, seq_len, hidden1, hidden2, dim1, dim2):
        view_1 = x.view(batch_size, seq_len, hidden1, dim1).permute(0, 1, 3, 2).reshape(batch_size, seq_len, hidden1 * dim1)
        view_3 = y.view(batch_size, seq_len, hidden2, dim2).permute(0, 1, 3, 2).reshape(batch_size, seq_len, hidden2 * dim2)
        add = view_1 + view_3
        return add


    @parametrize('dtype', ['float32'])
    def test_symbol_cases_dynamic_true(self, dtype):
        compiled_op_calc = torch.compile(self.op_calc_dynamic, backend="inductor", dynamic=True)
        for batch in [256, 512]:
            for seq in [80, 160]:
                hidden1 = 160
                hidden2 = 160
                dim1 = 40
                dim2 = 40

                x = self._generate_tensor((batch, seq, hidden1, dim1), dtype)
                torch._dynamo.mark_dynamic(x, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(x, 1, min=80, max=160)
                x = x.as_strided(
                    (batch, seq, hidden1, dim1),
                    (seq * hidden1 * dim1, hidden1 * dim1, dim1, 1)
                )
                y = self._generate_tensor((batch, seq, hidden2, dim2), dtype)
                torch._dynamo.mark_dynamic(y, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(y, 1, min=80, max=160)
                y = y.as_strided(
                    (batch, seq, hidden2, dim2),
                    (seq * hidden2 * dim2, hidden2 * dim2, dim2, 1)
                )
                std_result = self.op_calc_dynamic(x, y, batch, seq, hidden1, hidden2, dim1, dim2)
                inductor_result = compiled_op_calc(x, y, batch, seq, hidden1, hidden2, dim1, dim2)

                self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)


    @parametrize('dtype', ['float32'])
    def test_symbol_cases_dynamic_false(self, dtype):
        compiled_op_calc = torch.compile(self.op_calc_dynamic, backend="inductor", dynamic=True)
        for batch in [256, 512]:
            for seq in [80, 160]:
                hidden1 = 64
                hidden2 = 160
                dim1 = 40
                dim2 = 16

                x = self._generate_tensor((batch, seq, hidden1, dim1), dtype)
                torch._dynamo.mark_dynamic(x, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(x, 1, min=80, max=160)
                x = x.as_strided(
                    (batch, seq, hidden1, dim1),
                    (seq * hidden1 * dim1, hidden1 * dim1, dim1, 1)
                )
                y = self._generate_tensor((batch, seq, hidden2, dim2), dtype)
                torch._dynamo.mark_dynamic(y, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(y, 1, min=80, max=160)
                y = y.as_strided(
                    (batch, seq, hidden2, dim2),
                    (seq * hidden2 * dim2, hidden2 * dim2, dim2, 1)
                )

                std_result = self.op_calc_dynamic(x, y, batch, seq, hidden1, hidden2, dim1, dim2)
                inductor_result = compiled_op_calc(x, y, batch, seq, hidden1, hidden2, dim1, dim2)

                self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestLinearDynamic)


if __name__ == "__main__":
    run_tests()
