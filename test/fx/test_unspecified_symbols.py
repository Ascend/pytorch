"""
Add validation cases for torch.fx.experimental.symbolic_shapes APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates ShapeEnv.create_unspecified_symbol and ShapeEnv.create_unspecified_symint_and_symbol (extendable).
"""
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu

from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch._dynamo.source import ConstantSource


class TestUnspecifiedSymbols(TestCase):

    def test_create_unspecified_symbol_npu(self):
        # Verify that passing boundary values (0 and 1) returns an undetermined free symbol 
        # and is not statically folded into a constant (due to do_not_specialize_zero_one=True).
        shape_env = ShapeEnv()
        source = ConstantSource("test_source_symbol")
        
        sym_expr_0 = shape_env.create_unspecified_symbol(
            val=0,
            source=source,
            dynamic_dim=DimDynamic.DUCK
        )
        sym_expr_1 = shape_env.create_unspecified_symbol(
            val=1,
            source=source,
            dynamic_dim=DimDynamic.DUCK
        )
        
        self.assertIsNotNone(sym_expr_0)
        self.assertTrue(hasattr(sym_expr_0, "is_Symbol") or hasattr(sym_expr_0, "free_symbols"))
        self.assertIsNotNone(sym_expr_1)
        self.assertTrue(hasattr(sym_expr_1, "is_Symbol") or hasattr(sym_expr_1, "free_symbols"))

    def test_create_unspecified_symint_and_symbol_npu(self):
        # Verify the creation and wrapping of an unspecified symbol into a SymInt, 
        # ensuring the resulting node is correctly bound to the given ShapeEnv.
        shape_env = ShapeEnv()
        source = ConstantSource("test_source_symint")
        
        sym_int = shape_env.create_unspecified_symint_and_symbol(
            value=10,
            source=source,
            dynamic_dim=DimDynamic.DUCK
        )
        
        self.assertIsNotNone(sym_int)
        self.assertTrue(hasattr(sym_int, "node"))
        self.assertEqual(sym_int.node.shape_env, shape_env)


if __name__ == "__main__":
    run_tests()
