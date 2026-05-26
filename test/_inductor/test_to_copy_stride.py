# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, TestCase


RUN_NPU = torch.npu.is_available()
SOURCE_SIZE = (4, 2048, 16, 192)
SOURCE_STRIDE = (6291456, 192, 393216, 1)
CONTIGUOUS_STRIDE = (6291456, 3072, 192, 1)


def _to_copy_strided_float32(x):
    return torch.ops.aten._to_copy.default(
        x,
        dtype=torch.float32,
        layout=torch.strided,
    )


def _make_source_tensor():
    device = torch.device("npu:0")
    base_tensor = torch.empty(SOURCE_SIZE, dtype=torch.float32, device=device)
    return base_tensor.as_strided(
        size=SOURCE_SIZE,
        stride=SOURCE_STRIDE,
    )


@unittest.skip("Temporarily skip")
class TestToCopyStride(TestCase):
    def test_to_copy_fake_tensor_mode_stride(self):
        source = _make_source_tensor()

        fake_mode = FakeTensorMode()
        with fake_mode:
            fake_source = fake_mode.from_tensor(source)
            fake_copied = _to_copy_strided_float32(fake_source)
            self.assertEqual(fake_copied.shape, SOURCE_SIZE)
            self.assertEqual(fake_copied.stride(), SOURCE_STRIDE)

    def test_to_copy_compile_stride(self):
        source = _make_source_tensor()
        real_copied = _to_copy_strided_float32(source)
        self.assertEqual(real_copied.shape, SOURCE_SIZE)
        self.assertEqual(real_copied.stride(), SOURCE_STRIDE)

        compiled_to_copy = torch.compile(_to_copy_strided_float32, backend="inductor")
        compiled_copied = compiled_to_copy(source)
        self.assertEqual(compiled_copied.shape, SOURCE_SIZE)
        self.assertEqual(compiled_copied.stride(), SOURCE_STRIDE)


if __name__ == "__main__":
    run_tests()
