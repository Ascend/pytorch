"""
Add validation cases for torch.__getattribute__ API:
1. PyTorch community does not provide a direct standalone test for this API.
2. This file validates normal attribute lookup, dynamically added module
   attributes, and the AttributeError path for missing attributes.
"""

import types

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTorchGetattributeApi(TestCase):
    def _torch_module_dict(self):
        return types.ModuleType.__getattribute__(torch, "__dict__")

    def test_get_existing_attributes(self):
        module_dict = self._torch_module_dict()
        attrs = ("__config__", "Tensor", "nn", "empty")
        for name in attrs:
            # Use the module dict as the oracle instead of getattr, which shares
            # the same module attribute access path as torch.__getattribute__.
            self.assertIn(name, module_dict)
            self.assertIs(torch.__getattribute__(name), module_dict[name])

    def test_get_dynamic_attribute(self):
        module_dict = self._torch_module_dict()
        attr_name = "_torch_npu_getattribute_test_value"
        attr_value = object()
        self.assertNotIn(attr_name, module_dict)
        try:
            setattr(torch, attr_name, attr_value)
            # The dynamically added module attribute should be returned directly.
            self.assertIn(attr_name, module_dict)
            self.assertIs(torch.__getattribute__(attr_name), module_dict[attr_name])
        finally:
            module_dict.pop(attr_name, None)

    def test_get_missing_attribute_raises(self):
        module_dict = self._torch_module_dict()
        attr_name = "_torch_npu_missing_getattribute_test_value"
        self.assertNotIn(attr_name, module_dict)
        with self.assertRaisesRegex(AttributeError, attr_name):
            torch.__getattribute__(attr_name)


if __name__ == "__main__":
    run_tests()
