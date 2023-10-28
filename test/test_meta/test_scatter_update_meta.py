import torch
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
)

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestScatterUpdateMeta(TestCase):

    def test_scatter_update_meta(self):
        with FakeTensorMode() as mode:
            in_self = torch.randn(4, 4, 32, 256, dtype=torch.float16).npu()
            in_indices = torch.tensor([1, 1, 1, 1]).npu()
            in_updates = torch.randn(4, 4, 1, 256, dtype=torch.float16).npu()
            fake_self = mode.from_tensor(in_self)
            fake_indices = mode.from_tensor(in_indices)
            fake_updates = mode.from_tensor(in_updates)
            self.assertIsNotNone(fake_self)
            self.assertIsNotNone(fake_indices)
            self.assertIsNotNone(fake_updates)
            fake_result = torch.ops.npu.scatter_update(fake_self, fake_indices, fake_updates, -2)

            self.assertEqual(fake_result.shape, in_self.shape)
            self.assertEqual(fake_result.dtype, in_self.dtype)
            self.assertEqual(fake_result.device, in_self.device)
            self.assertTrue(isinstance(fake_result, FakeTensor))
            self.assertIsNot(fake_result, fake_self)
            self.assertIsNot(fake_result, in_self)


    def test_scatter_update__meta(self):
        with FakeTensorMode() as mode:
            in_self = torch.randn(4, 4, 32, 256, dtype=torch.float32).npu()
            in_indices = torch.tensor([1, 1, 1, 1]).npu()
            in_updates = torch.randn(4, 4, 1, 256, dtype=torch.float32).npu()
            fake_self = mode.from_tensor(in_self)
            fake_indices = mode.from_tensor(in_indices)
            fake_updates = mode.from_tensor(in_updates)
            self.assertIsNotNone(fake_self)
            self.assertIsNotNone(fake_indices)
            self.assertIsNotNone(fake_updates)
            fake_result = torch.ops.npu.scatter_update_(fake_self, fake_indices, fake_updates, -2)

            self.assertEqual(fake_result.shape, in_self.shape)
            self.assertEqual(fake_result.dtype, in_self.dtype)
            self.assertEqual(fake_result.device, in_self.device)
            self.assertTrue(isinstance(fake_result, FakeTensor))
            self.assertIs(fake_result, fake_self)
            self.assertIsNot(fake_result, in_self)


if __name__ == "__main__":
    run_tests()
