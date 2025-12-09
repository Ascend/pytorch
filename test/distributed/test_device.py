import os
import threading
from ctypes import byref, c_int, c_void_p, CDLL

import torch
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDevice(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 1
    
    def _check_not_npu(self, device_id=0):
        ascendcl_h = CDLL("libascendcl.so")
        device_id = c_int(device_id)
        activate = c_int(1)
        rc = ascendcl_h.aclrtGetPrimaryCtxState(device_id, c_void_p(), byref(activate))
        if rc != 0:
            raise RuntimeError("call aclrtGetPrimaryCtxState error")
        del ascendcl_h
        self.assertEqual(activate.value, 0)

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

    @skipIfUnsupportMultiNPU(2)
    def test_set_device(self):
        torch.npu.set_device('npu:1')
        self._check_not_npu()
        device = torch.npu.current_device()
        self.assertEqual(device, 1)

    @skipIfUnsupportMultiNPU(2)
    def test_with_device(self):
        with torch.npu.device('npu:1'):
            a = torch.rand(1).npu()
            self.assertEqual(a.device.index, 1)
        self._check_not_npu()
        with torch.npu.device('npu:1'):
            b = torch.rand(1).npu()
            self.assertEqual(b.device.index, 1)
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_stream(self):
        s = torch.npu.Stream()
        self._check_not_npu(1)

    @skipIfUnsupportMultiNPU(2)
    def test_stream0(self):
        s = torch.npu.Stream(0)
        self._check_not_npu(1)

    @skipIfUnsupportMultiNPU(2)
    def test_stream1(self):
        s = torch.npu.Stream(1)
        self._check_not_npu(0)

    def test_event(self):
        s = torch.npu.Event()
        s.record()
        s.wait()

    @skipIfUnsupportMultiNPU(2)
    def test_storage0(self):
        s1 = torch.npu.FloatStorage(10)

    @skipIfUnsupportMultiNPU(2)
    def test_storage1(self):
        s = torch.UntypedStorage(10, device='npu:1')
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_empty0(self):
        a = torch.empty(2, device='npu:0')

    @skipIfUnsupportMultiNPU(2)
    def test_empty1(self):
        a = torch.empty(2, device='npu:1')
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_rand0(self):
        a = torch.rand(2, device='npu:0')

    @skipIfUnsupportMultiNPU(2)
    def test_rand1(self):
        a = torch.rand(2, device='npu:1')
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_npu0(self):
        a = torch.rand(2).npu()

    @skipIfUnsupportMultiNPU(2)
    def test_npu1(self):
        a = torch.rand(2).npu(1)
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_to0(self):
        a = torch.rand(2).to('npu:0')

    @skipIfUnsupportMultiNPU(2)
    def test_to1(self):
        a = torch.rand(2).to('npu:1')
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def test_pin_memory(self):

        def worker_function():
            pinmemory_tensor = torch.empty(32, pin_memory=True)

        a = torch.rand(2).to('npu:1')
        t = threading.Thread(target=worker_function)
        t.start()
        t.join()

        device = torch.npu.current_device()
        self.assertEqual(device, 0)
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def thread(self):
        def ff():
            a = torch.rand(2, device='npu:1')
            self.assertEqual(a.device.index, 1)

        t = threading.Thread(target=ff)
        t.start()
        t.join()
        self._check_not_npu()

    @skipIfUnsupportMultiNPU(2)
    def thread1(self):
        def ff():
            a = torch.rand(2, device='npu:1')
            self.assertEqual(a.device.index, 1)

        b = torch.rand(2).npu()
        self.assertEqual(b.device.index, 0)
        t = threading.Thread(target=ff)
        t.start()
        t.join()


if __name__ == "__main__":
    os.environ["ACL_OP_INIT_MODE"] = "1"
    run_tests()
