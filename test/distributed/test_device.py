import threading

import torch
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu


class TestDevice(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 1
    
    def test_event_create(self):
        a = torch.full((3, 4), float(0), device='npu:0')
        e = torch.npu.Event()
        s = torch.npu.Stream()

        def target_fuc(result):
            e.record(s)
            e.synchronize()
            result[0] = 1

        result = [0]
        t = threading.Thread(target=target_fuc, args=(result, ))
        t.start()
        t.join()
        self.assertEqual(result[0], 1)
    
    def test_event_isinstance(self):
        npu_event = torch.npu.Event()
        self.assertIsInstance(npu_event, torch.npu.Event)
        self.assertIsInstance(npu_event, torch.Event)
        # Check whether torch_npu.npu.Event is a subclass of torch.Event
        self.assertTrue(issubclass(torch_npu.npu.Event, torch.Event))

    def test_stream_create(self):
        s = torch_npu._C._npu_getCurrentStream(0)

    def test_tensor(self):
        a = torch.full((3, 4), float(0), device='npu:0')

        def target_fuc(result):
            b = torch.full((3, 4), float(0), device='npu:0')
            result[0] = 1

        result = [0]
        t = threading.Thread(target=target_fuc, args=(result, ))
        t.start()
        t.join()
        self.assertEqual(result[0], 1)

    def test_storage(self):
        s = torch.npu.Stream()

        def target_fuc(result):
            b = torch.npu.FloatStorage(10)
            result[0] = 1

        result = [0]
        t = threading.Thread(target=target_fuc, args=(result, ))
        t.start()
        t.join()
        self.assertEqual(result[0], 1)


if __name__ == "__main__":
    run_tests()
