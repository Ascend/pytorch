"""
Add validation cases for torch.fx.experimental.symbolic_shapes.ShapeEnv APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for these APIs, so this file is added.
2. This file validates ShapeEnv.create_symboolnode and ShapeEnv.create_symfloatnode.
3. This file is extendable for other torch.fx.experimental.symbolic_shapes.ShapeEnv APIs.
"""

import sympy
import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestShapeEnvSymbolicShapes(TestCase):

    def test_create_symboolnode_with_npu_tensor_shape(self):
        # Create a tensor on the current accelerator and use its shape as
        # the backed value for symbolic boolean creation.
        tensor = torch.ones(2, 3).to(device_type)
        self.assertEqual(tensor.device.type, device_type)

        # Build a ShapeEnv symbol from the NPU tensor shape.
        # This verifies that the API works with metadata from an NPU tensor.
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(
            tensor.size(0),
            source=ConstantSource("symbool"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )

        # Create a SymBool from a symbolic expression based on the NPU tensor shape.
        symbool = shape_env.create_symboolnode(sympy.Eq(symbol, tensor.size(0)))

        # Validate both the returned type and the boolean result.
        self.assertIsInstance(symbool, torch.SymBool)
        self.assertTrue(bool(symbool))

    def test_create_symfloatnode_with_npu_tensor_shape(self):
        # Create a tensor on the current accelerator and use its shape as
        # the backed value for symbolic float creation.
        tensor = torch.ones(2, 3).to(device_type)
        self.assertEqual(tensor.device.type, device_type)

        # Build a ShapeEnv symbol from the NPU tensor shape.
        # This keeps the symbolic value connected with NPU tensor metadata.
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(
            tensor.size(0),
            source=ConstantSource("symfloat"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )

        # Create a SymFloat from a symbolic expression based on the NPU tensor shape.
        # The hint equals tensor.size(0) + 0.5.
        symfloat = shape_env.create_symfloatnode(
            symbol + sympy.Float(0.5),
            hint=float(tensor.size(0)) + 0.5,
        )

        # Validate both the returned type and the concrete hint result.
        self.assertIsInstance(symfloat, torch.SymFloat)
        self.assertEqual(float(symfloat), 2.5)


if __name__ == "__main__":
    run_tests()
