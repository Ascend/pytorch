# Owner(s): ["module: tests"]
from itertools import product
from functools import partial
import torch
import torch_npu
import torch_npu.testing
import torch.utils.data
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, 
    run_tests)
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    instantiate_device_type_tests,
    dtypes, onlyNativeDeviceTypes)
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_dtype import all_types_and_complex_and, all_types_and

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32 

DEVICE_NAME = torch_npu.npu.get_device_name(0)

device_is_910A = False
if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
    device_is_910A = True

if device_is_910A:
    all_types_and_complex_and = all_types_and

class TestPut(TestCase):

    def test_put_empty(self, device):
        for dst_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                for accumulate in [False, True]:
                    dst = torch.randn(dst_shape, device=device)
                    indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                    src = torch.randn(indices_shape, device=device)
                    self.assertEqual(dst, dst.put_(indices, src, accumulate=accumulate))

instantiate_device_type_tests(TestPut, globals(), only_for='privateuse1')

if __name__ == "__main__":
    run_tests()