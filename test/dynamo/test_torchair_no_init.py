# Owner(s): ["module: dynamo"]
import sys
import torch.nn.init as init

import torch
from torch._dynamo.test_case import TestCase

import torch_npu
import torch_npu.dynamo


class TestTorchairNoInit(TestCase):
    def _check_torchair_no_init(self):
        self.assertTrue("torchair.npu_fx_compiler" not in sys.modules)
        torchair = sys.modules.get('torchair', None)
        self.assertTrue(torchair is not None)
        self.assertTrue(torchair._torchair is None)
        self.assertTrue(torchair._exception is None)

    def test_getattr_warningregistry(self):
        # Do not init torchair because of some bug in Python 3.8.1.
        self.assertTrue("torchair" in sys.modules)
        torchair = sys.modules.get('torchair', None)
        self.assertTrue(isinstance(torchair, torch_npu.dynamo._LazyTorchair))

        value = getattr(torchair, '__warningregistry__', None)
        self.assertTrue(value is None)
        self._check_torchair_no_init()

    def test_assert_warn_regex(self):
        # Test func TestCase.assertWarnsRegex.
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_uniform_(tensor)
        self._check_torchair_no_init()

    def test_hasattr(self):
        for m in sys.modules:
            if hasattr(sys.modules[m], '_attr_test_hasattr'):
                setattr(sys.modules[m], '_attr_test_hasattr', 1)
        
        torchair = sys.modules.get('torchair', None)
        self.assertTrue(torchair is not None)
        self.assertTrue(not hasattr(torchair, '_attr_test_hasattr'))
        self._check_torchair_no_init()

    def test_getattr(self):
        for m in sys.modules.values():
            if getattr(m, '__warningregistry__', None):
                m.__warningregistry__ = {}
        
        self._check_torchair_no_init()
        
    def test_attribute_error(self):
        torchair = sys.modules.get('torchair', None)
        self.assertTrue(torchair is not None)
        with self.assertRaisesRegex(AttributeError, 
                "Try to get torchair's attr `get_npu_backend` before torchair is initialized."):
            torchair.get_npu_backend()
        self._check_torchair_no_init()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
