import os
import shutil
import importlib.util
import unittest

# has_triton_package will be called during torch.testing init, remove existed triton due to init issue in torch 2.4
if os.path.isdir("triton"):
    shutil.rmtree("triton")
if os.path.isfile("triton"):
    os.remove("triton")

from torch.utils import _triton
from torch.testing._internal.common_utils import TestCase, run_tests

TRITON_IS_INSTALLED = importlib.util.find_spec("triton") is not None
# clear lru cache
_triton.has_triton_package.cache_clear()


class TestHasTritonPackage(TestCase):
    def setUp(self):
        super().setUp()
        if not os.path.exists("triton"):
            os.mkdir("triton")

    def tearDown(self):
        super().tearDown()
        if os.path.isdir("triton"):
            shutil.rmtree("triton")

    @unittest.skipIf(TRITON_IS_INSTALLED, "Skip this case due to triton is installed.")
    def test_has_triton_package(self):
        self.assertTrue(_triton.has_triton_package())

    @unittest.skipIf(TRITON_IS_INSTALLED, "Skip this case due to triton is installed.")
    def test_has_triton_package_with_patch(self):
        import torch_npu
        self.assertFalse(_triton.has_triton_package())


if __name__ == '__main__':
    run_tests()
