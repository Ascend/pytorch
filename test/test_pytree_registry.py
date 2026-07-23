"""
Add validation cases for torch.utils._pytree APIs on NPU:
1. PyTorch community lacks sufficient and direct validation for torch.utils._pytree.SUPPORTED_NODES.pop.
2. This file validates pytree registry mutation behavior and is extendable.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils import _pytree


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestPytreeSupportedNodes(TestCase):
    def test_pop(self):
        class CustomNode:
            def __init__(self, value):
                self.value = value

        def flatten_fn(node):
            return [node.value], None

        def unflatten_fn(values, context):
            return CustomNode(values[0])

        _pytree.register_pytree_node(CustomNode, flatten_fn, unflatten_fn)
        self.addCleanup(_pytree._deregister_pytree_node, CustomNode)
        value = torch.tensor([1.0]).to(device_type)
        node = CustomNode(value)

        leaves, _ = _pytree.tree_flatten(node)
        self.assertEqual(leaves, [value])

        node_def = _pytree.SUPPORTED_NODES.pop(CustomNode)
        self.addCleanup(_pytree.SUPPORTED_NODES.__setitem__, CustomNode, node_def)
        self.assertIs(node_def.flatten_fn, flatten_fn)
        self.assertNotIn(CustomNode, _pytree.SUPPORTED_NODES)
        self.assertIsNone(_pytree.SUPPORTED_NODES.pop(CustomNode, None))

        leaves, spec = _pytree.tree_flatten(node)
        self.assertTrue(spec.is_leaf())
        self.assertEqual(leaves, [node])


if __name__ == "__main__":
    run_tests()
