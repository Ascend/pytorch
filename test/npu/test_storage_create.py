import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


class TestStorageCreate(TestCase):

    def test_storage_create(self):
        storage = torch.UntypedStorage(2, device='npu:0')
        self.assertEqual(storage.device.type, 'npu')


if __name__ == '__main__':
    run_tests()
