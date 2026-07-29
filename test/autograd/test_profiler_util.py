"""
Add validation cases for torch.autograd.profiler_util APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.autograd.profiler_util.Kernel (extendable).
"""

from torch.autograd.profiler_util import Kernel
from torch.testing._internal.common_utils import TestCase, run_tests


class TestProfilerUtilKernelApi(TestCase):
    def test_kernel_index_returns_field_position(self):
        # Kernel is a namedtuple with fields: name, device, duration
        kernel = Kernel(name="add", device="npu", duration=10)
        self.assertEqual(0, kernel.index("add"))
        self.assertEqual(1, kernel.index("npu"))
        self.assertEqual(2, kernel.index(10))

    def test_kernel_index_with_start_stop_bounds(self):
        kernel = Kernel(name="add", device="npu", duration=10)
        self.assertEqual(2, kernel.index(10, 0, 3))
        with self.assertRaises(ValueError):
            kernel.index(10, 0, 2)

    def test_kernel_index_raises_value_error_when_missing(self):
        kernel = Kernel(name="add", device="npu", duration=10)
        with self.assertRaises(ValueError):
            kernel.index("nonexistent")


if __name__ == "__main__":
    run_tests()
