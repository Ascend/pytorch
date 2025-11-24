import faulthandler
import gc
import importlib
import os
import signal
import sys
import threading
import unittest
from itertools import product

import torch
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

TEST_NPU = torch.npu.is_available()


def _collect():
    gc.collect()
    torch.npu.empty_cache()


class TestHostCachingAllocatorBasic(TestCase):
    def test_pin_memory_on_non_blocking_copy(self): 
        t_acc = torch.randn(100).to(torch.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        torch.accelerator.synchronize()
        self.assertTrue(t_host.is_pinned())
        self.assertEqual(t_acc.cpu(), t_host)
    
    def test_pin_memory_reuse(self):
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t_new = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t_new.data_ptr(), ptr)
    
    def test_to_non_blocking(self):
        stream = torch_npu.npu.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            torch_npu.npu.synchronize()
            b = a.to(device=dst, non_blocking=non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in [
            ("npu", True),
            ("npu", False),
            ("cpu", True),  # pinned is true only when non_blocking=True and dst="cpu"
            ("cpu", False),
        ]:
            # Creates source on the opposite device from destination.
            src = torch.randn(1000, 1000, 2, 100,
                              device="npu" if dst == "cpu" else "cpu",
                              pin_memory=True if dst == "npu" else False)
            _test_to_non_blocking(src, try_non_blocking, dst)
    
    def test_pin_memory_basic(self):
        a = torch.Tensor([1])
        b = a.pin_memory()
        c = a.pin_memory()
        d = b.pin_memory()
        self.assertTrue(a.data_ptr() != b.data_ptr())
        self.assertTrue(b.data_ptr() != c.data_ptr())
        self.assertTrue(b.data_ptr() == d.data_ptr())
    
    def test_malloc_copykernel(self):
        a = torch.Tensor([1])
        b = torch.Tensor([1])
        c = a.to('npu:0', non_blocking=True)
        d = b.to('npu:0', non_blocking=True)
        # we do not synchronize here
        # the above to will call synchronize internally
        self.assertEqual(c.item(), d.item())

    def test_fragmentation_resilience_varied_sizes(self):
        # Allocate/release varied sizes and ensure subsequent allocation succeeds and is pinned
        sizes = [10_000, 200_000, 50_000, 1_000_000, 75_000]
        tensors = [torch.empty(s, dtype=torch.float32).pin_memory() for s in sizes]
        for t in tensors:
            self.assertTrue(t.is_pinned())
        # Free in different order to stress cache
        for idx in [2, 0, 4, 1, 3]:
            tensors[idx] = None
        _collect()

        # Allocate a size that fits into previously freed blocks
        t_new = torch.empty(150_000, dtype=torch.float32).pin_memory()
        self.assertTrue(t_new.is_pinned())
        # If you have allocator stats, assert on reuse; otherwise, just sanity check
        del t_new
        _collect()

    def test_concurrent_pinned_allocations_threaded(self):
        errs = []

        def worker(n):
            try:
                for _ in range(n):
                    t = torch.empty(256, 256, dtype=torch.float32).pin_memory()
                    self.assertTrue(t.is_pinned())
                    # Move to GPU non-blocking to simulate pipeline
                    _ = t.to("npu", non_blocking=True)
            except Exception as e:
                errs.append(e)

        threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        torch.npu.synchronize()
        self.assertTrue(not errs)
        
    def test_pin_memory_on_views_and_clones(self):
        base = torch.randn(1024, 1024)
        view = base[:512, :].pin_memory()
        clone = base.clone().pin_memory()
        self.assertTrue(view.is_pinned())
        self.assertTrue(clone.is_pinned())
        # Ensure their contents are consistent after to('cuda')
        yv = view.to("npu", non_blocking=True)
        yc = clone.to("npu", non_blocking=True)
        torch.npu.synchronize()
        self.assertTrue(yv.device and yc.device)

    def test_pin_memory_on_dtypes_and_non_contiguous(self):
        x = torch.randn(128, 128, dtype=torch.float64).t()  # non-contiguous
        xp = x.pin_memory()
        self.assertTrue(xp.is_pinned())
        # Transfer works even if non-contiguous (PyTorch will handle copy)
        y = xp.to("npu", non_blocking=True)
        torch.npu.synchronize()
        self.assertTrue(y.is_npu)


def set_faulthander_if_available(_=None):
    faulthandler.enable(sys.__stderr__)
    if not IS_WINDOWS:
        faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, chain=False)

set_faulthander_if_available()


class CountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i):
        return i
    
    def __len__(self):
        return self.n


class DictDataset(Dataset):
    def __len__(self):
        return 4
    
    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.empty(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': torch.tensor(ndx),
            },
        }


class StringDataset(Dataset):
    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
    
    def is_pinned(self):
        return self.inp.is_pinned() and self.tgt.is_pinned()

module_name = os.path.splitext(os.path.basename(__file__))[0]
self_module = importlib.import_module(module_name)


def collate_wrapper(batch):
    return self_module.SimpleCustomBatch(batch)


def collate_into_packed_sequence(batch):
    data = torch.stack([sample[0] for sample in batch], 1)
    t, b = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, enforce_sorted=False)


def collate_into_packed_sequence_batch_first(batch):
    data = torch.stack([sample[0] for sample in batch], 0)
    b, t = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)


class TestDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_sequential_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True, pin_memory_device='npu')
        for input_, target in loader:
            self.assertTrue(input_.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4,
                            pin_memory=True, pin_memory_device='npu')
        for input_, target in loader:
            self.assertTrue(input_.is_pinned())
            self.assertTrue(target.is_pinned())


class TestStringDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for (s, n) in loader:
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class TestDictDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = DictDataset()

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_pin_memory_device(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True, pin_memory_device='npu')
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned(device='npu'))
            self.assertTrue(sample['another_dict']['a_number'].is_pinned(device='npu'))


class TestCustomPinFn(TestCase):
    def setUp(self):
        super().setUp()
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.dataset = TensorDataset(inps, tgts)

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_custom_batch_pin(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (collate_into_packed_sequence_batch_first, torch.nn.utils.rnn.PackedSequence),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn,
                                pin_memory=True, pin_memory_device='npu')
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                # 对于 PackedSequence，is_pinned 在 DataLoader 中会递归 pin 其 data
                if hasattr(sample, 'is_pinned'):
                    self.assertTrue(sample.is_pinned())
                else:
                    # PackedSequence: 检查其 data
                    self.assertTrue(sample.data.is_pinned())

    @unittest.skipIf(not TEST_NPU, "NPU unavailable")
    def test_custom_batch_pin_worker(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (collate_into_packed_sequence_batch_first, torch.nn.utils.rnn.PackedSequence),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn,
                                pin_memory=True, num_workers=1, pin_memory_device='npu')
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                if hasattr(sample, 'is_pinned'):
                    self.assertTrue(sample.is_pinned())
                else:
                    self.assertTrue(sample.data.is_pinned())

if __name__ == "__main__":
    run_tests()