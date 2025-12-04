import unittest
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclSendRecvDistTest(TestCase):
    world_size = 2

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['MULTI_STREAM_MEMORY_REUSE'] = '2'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)

    @classmethod
    def _test_avoid_isend_irecv(
            cls, rank, shared_tensors, world_size, init_pg, c2p, done_event=None):
        init_pg(rank, world_size)
        res = torch.ones(400, 1024, 1024).to(shared_tensors.dtype).to(f"npu:{rank}")
        xs = shared_tensors.to(f"npu:{rank}")
        dst = 0
        src = 1
        if src == rank:
            dist.isend(xs, dst)
            xs = None
            # 内存复写
            res1 = torch.ones(400, 1024, 1024).to(shared_tensors.dtype).to(f"npu:{rank}")
            c2p.put((dst, None))
        else:
            dist.irecv(res, src)
            torch.npu.synchronize()
            c2p.put((src, res.cpu()))

        if done_event is not None:
            done_event.wait(timeout=5)

    @classmethod
    def _test_avoid_batch_isend_irecv(
            cls, rank, shared_tensors, world_size, init_pg, c2p, done_event=None):
        init_pg(rank, world_size)
        res = torch.ones(400, 1024, 1024).to(shared_tensors.dtype).to(f"npu:{rank}")
        xs = shared_tensors.to(f"npu:{rank}")
        dst = 0
        src = 1
        if src == rank:
            send_op = dist.P2POp(dist.isend, xs, dst)
            dist.batch_isend_irecv([send_op])
            xs = None
            send_op = None
            # 内存复写
            res1 = torch.ones(400, 1024, 1024).to(shared_tensors.dtype).to(f"npu:{rank}")
            c2p.put((dst, None))
        else:
            recv_op = dist.P2POp(dist.irecv, res, src)
            dist.batch_isend_irecv([recv_op])
            torch.npu.synchronize()
            c2p.put((src, res.cpu()))

        if done_event is not None:
            done_event.wait(timeout=5)

    def _test_multiprocess(self, f, shared_tensors, init_pg):
        ws = self.world_size
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        done_event = ctx.Event()
        ps = []
        expected = 0
        result = 1
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, shared_tensors, ws, init_pg, c2p, done_event))
            p.start()
            ps.append(p)

        for _ in range(2):
            pid, output = c2p.get()
            if pid == 0:
                expected = shared_tensors
            else:
                result = output

        self.assertEqual(
            expected,
            result,
            (
                "Expect rank {} to receive tensor {} but got {}."
            ).format(pid, expected, result)
        )

        done_event.set()
        for p in ps:
            p.join(2)

    @skipIfUnsupportMultiNPU(2)
    def test_avoid_isend_irecv_hccl(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_avoid_isend_irecv,
            torch.randn(400, 1024, 1024),
            HcclSendRecvDistTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_avoid_batch_isend_irecv_hccl(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_avoid_batch_isend_irecv,
            torch.randn(400, 1024, 1024),
            HcclSendRecvDistTest._init_dist_hccl)


if __name__ == '__main__':
    run_tests()
