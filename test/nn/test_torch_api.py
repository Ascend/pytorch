import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPythonApi(TestCase):
    def test_is_storage(self):
        a = torch.rand(2, 3)
        b = torch.FloatStorage([1, 2, 3, 4, 5, 6])

        self.assertFalse(torch.is_storage(a))
        self.assertTrue(torch.is_storage(b))

    def test_storage_casts(self):
        storage = torch.IntStorage([-1, 0, 1, 2, 3, 4])

        storage_type = [
            [storage, 'torch.IntStorage', torch.int32],
            [storage.float(), 'torch.FloatStorage', torch.float32],
            [storage.half(), 'torch.HalfStorage', torch.float16],
            [storage.long(), 'torch.LongStorage', torch.int64],
            [storage.short(), 'torch.ShortStorage', torch.int16],
            [storage.char(), 'torch.CharStorage', torch.int8]
        ]
        for item in storage_type:
            self.assertEqual(item[0].size(), 6)
            self.assertEqual(item[0].tolist(), [-1, 0, 1, 2, 3, 4])
            self.assertEqual(item[0].type(), item[1])
            self.assertEqual(item[0].int().tolist(), [-1, 0, 1, 2, 3, 4])
            self.assertIs(item[0].dtype, item[2])

    def test_DataLoader(self):
        class SimpleCustomBatch:
            def __init__(self, data):
                transposed_data = list(zip(*data))
                self.inp = torch.stack(transposed_data[0], 0)
                self.tgt = torch.stack(transposed_data[1], 0)

            # custom memory pinning method on custom type
            def pin_memory(self):
                self.inp = self.inp.pin_memory()
                self.tgt = self.tgt.pin_memory()
                return self

        def collate_wrapper(batch):
            return SimpleCustomBatch(batch)

        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5).npu()
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5).npu()
        dataset = TensorDataset(inps, tgts)

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                            pin_memory=True)


if __name__ == "__main__":
    run_tests()
