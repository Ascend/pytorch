# Owner(s): ["module: nested tensor"]

from collections import namedtuple, OrderedDict
from multiprocessing.reduction import ForkingPickler
import torch
import numpy as np
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNestedTensor(TestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row))
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.int64))

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row))
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.int64))

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [16, 32])
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(1, max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row))
        nested_tensor = torch.nested.nested_tensor(data)
        nested_tensor_list = nested_tensor.unbind()
        for i in range(batch_size):
            self.assertEqual(nested_tensor_list[i], nested_tensor_ref_list[i].type(torch.float32))

    def _test_unbind_case(self, a, b):
        nt = torch.nested.nested_tensor([a, b], dtype=a.dtype)
        nt_list = nt.unbind()
        self.assertEqual(len(nt_list), 2)
        self.assertEqual(nt_list[0], a)
        self.assertEqual(nt_list[1], b)
    
    def test_unbind_case1(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[7, 8], [10, 11]])
        self._test_unbind_case(a, b)
        
    def test_unbind_case2(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        b = torch.tensor([[7, 8], [10, 11]], dtype=torch.float32)
        self._test_unbind_case(a, b)

    def test_unbind_case3(self):
        a = torch.tensor([[], []])
        b = torch.tensor([[], [], []])
        self._test_unbind_case(a, b)

    def test_default_options_nested_tensor(self):
        default_nested_tensor = torch.nested.nested_tensor([])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.requires_grad, default_tensor.requires_grad)
    
    def test_nested_tensor_size(self):
        nt = torch.nested.nested_tensor([torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8], [10, 11], [12, 13]])])
        self.assertEqual(nt.size(0), 2)
        self.assertRaisesRegex(RuntimeError,
        "Given dimension 1 is irregular and does not have a size", 
        lambda: nt.size(1),
        )

        nt = torch.nested.nested_tensor([2])
        self.assertEqual(nt.size(0), 1)

if __name__ == '__main__':
    instantiate_parametrized_tests(TestNestedTensor)
    run_tests()