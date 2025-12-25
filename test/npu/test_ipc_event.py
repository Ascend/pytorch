import os
import gc
import unittest
import numpy as np
import torch.multiprocessing as mp

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def is_ipc_event_supported():
    try:
        ev = torch.npu.Event(enable_timing=False, interprocess=True)
    except RuntimeError as e:
        return False
    else:
        return True

skip_ipc_event_case = not is_ipc_event_supported()


class Test_ipc_event(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_d2d_copy1(self):
        a = torch.tensor(1.).to('npu:0')
        b = torch.tensor(1.).to('npu:1')
        b.copy_(a)
        self.assertEqual(a.cpu(), b.cpu())

    @skipIfUnsupportMultiNPU(2)
    def test_d2d_copy2(self):
        a = torch.tensor(1.).to('npu:0')
        b = a.to('npu:1')
        self.assertEqual(a.cpu(), b.cpu())

    @skipIfUnsupportMultiNPU(2)
    def test_d2d_copy3(self):
        a = torch.ones(2, 1024, 1024, 1024).to('npu:0')
        b = a.to('npu:1', non_blocking=True)
        self.assertEqual(a.cpu(), b.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_ipc_event_pickle(self):
        if skip_ipc_event_case:
            return

        ev = torch.npu.Event(enable_timing=False, interprocess=True)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        q.put(ev)

    @staticmethod
    def _child_proc1(q):
        ev = q.get()
        assert ev.device.type == 'npu'
        assert ev.device.index == 0
        ev.wait()
        ev.synchronize()

    @SupportedDevices(['Ascend910B'])
    def test_ipc_event_1(self):
        if skip_ipc_event_case:
            return

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=Test_ipc_event._child_proc1, args=(q,))
        p.start()

        dev = torch.device("npu:0")
        with torch.npu.device(dev):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                ev = torch.npu.Event(enable_timing=False, interprocess=True)
                ev.record(stream)

            q.put(ev)
        p.join()

    @staticmethod
    def _child_proc2(q1, q2):
        dev = torch.device("npu:0")
        with torch.npu.device(dev):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                ev = q1.get()
                assert ev.device.type == 'npu'
                assert ev.device.index == 0
                ev.wait()
                ev.record(stream)
                q2.put('x')
                assert q1.get() == 'y'

    @SupportedDevices(['Ascend910B'])
    def test_ipc_event_2(self):
        if skip_ipc_event_case:
            return

        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        p = ctx.Process(target=Test_ipc_event._child_proc2, args=(q1, q2))
        p.start()

        dev = torch.device("npu:0")
        with torch.npu.device(dev):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                ev = torch.npu.Event(enable_timing=False, interprocess=True)
                ev.record(stream)

            q1.put(ev)
            self.assertEqual(q2.get(), 'x')
            ev.wait()
            ev.synchronize()
            q1.put('y')
        p.join()

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B'])
    def test_event_handle_multi_npu(self):
        if skip_ipc_event_case:
            return

        d0 = torch.device("npu:0")
        d1 = torch.device("npu:1")
        with torch.npu.device(d0):
            e0 = torch.npu.Event(enable_timing=False, interprocess=True)

        with torch.npu.device(d1):
            # create handle on different device from un-recorded event
            e0.ipc_handle()

        with torch.npu.device(d0):
            e1 = torch.npu.Event(enable_timing=False, interprocess=True)
            stream = torch.npu.Stream()
            e1.record(stream)

        with torch.npu.device(d1):
            # create handle on different device from recorded event
            e1.ipc_handle()

    @staticmethod
    def _test_event_handle_importer_consumer(handle, p2c, c2p):
        e1 = torch.npu.Event.from_ipc_handle(0, handle)
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        e1.synchronize()
        c2p.put(1)  # notify synchronization is done in child
        p2c.get()  # wait for parent to finish before destructing child event

    @SupportedDevices(['Ascend910B'])
    def test_event_handle_importer(self):
        if skip_ipc_event_case:
            return

        e0 = torch.npu.Event(enable_timing=False, interprocess=True)
        self.assertTrue(e0.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=Test_ipc_event._test_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        e0.record()
        p2c.put(0)  # notify child event is recorded

        c2p.get()  # wait for synchronization in child
        self.assertTrue(e0.query())
        p2c.put(1)  # notify child that parent is done
        p.join()

    @staticmethod
    def _test_event_handle_exporter_consumer(handle, p2c, c2p):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            e1 = torch.npu.Event.from_ipc_handle(torch.npu.current_device(), handle)
            e1.record()
            c2p.put(0)
            # wait for parent process finished synchronization before
            # destructing e1
            p2c.get()

    @SupportedDevices(['Ascend910B'])
    def test_event_handle_exporter(self):
        if skip_ipc_event_case:
            return

        e0 = torch.npu.Event(enable_timing=False, interprocess=True)

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=Test_ipc_event._test_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()
        # wait for event in child process is recorded
        c2p.get()

        e0.synchronize()
        self.assertTrue(e0.query())
        p2c.put(0)
        p.join()


if __name__ == '__main__':
    run_tests()
