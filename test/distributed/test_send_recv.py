import unittest
from unittest.mock import patch
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
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _init_pg_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist.new_group([0, 1])

    @classmethod
    def _test_send_recv_dist(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        res = torch.ones(2, 2).to(shared_tensors.dtype).to(f"npu:{rank}")
        xs = shared_tensors.to(f"npu:{rank}")
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst)
            c2p.put((dst, xs.cpu()))
        else:
            pg.recv(res, src)
            c2p.put((src, res.cpu()))

    @classmethod
    def _test_send_recv_dist_with_p2p(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        res = torch.ones(2, 2).to(shared_tensors.dtype).to(f"npu:{rank}")
        xs = shared_tensors.to(f"npu:{rank}")
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst)
            c2p.put((dst, xs.cpu()))
        elif rank == dst:
            pg.recv(res, src)
            c2p.put((src, res.cpu()))

    @classmethod
    def _test_send_recv_dist_recv_with_format(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        res = torch_npu.npu_format_cast(torch.ones(2, 2).to(shared_tensors.dtype).to(f"npu:{rank}"), 29)
        xs = shared_tensors.to(f"npu:{rank}")
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst)
            c2p.put((dst, xs.cpu()))
        else:
            pg.recv(res, src)
            c2p.put((src, res.cpu()))

    @classmethod
    def _test_send_recv_dist_send_with_format(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        res = torch.ones(2, 2).to(shared_tensors.dtype).to(f"npu:{rank}")
        xs = torch_npu.npu_format_cast(shared_tensors.to(f"npu:{rank}"), 29)
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst)
            c2p.put((dst, xs.cpu()))
        else:
            pg.recv(res, src)
            c2p.put((src, res.cpu()))

    @classmethod
    def _test_send_recv_group(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        tag = 1
        res = [torch.ones(2, 2).to(f"npu:{rank}")]
        xs = [shared_tensors.to(f"npu:{rank}")]
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst, tag).wait()
            c2p.put((dst, xs[0].cpu()))
        else:
            pg.recv(res, src, tag).wait()
            c2p.put((src, res[0].cpu()))

    @classmethod
    def _test_send_recv_group_recv_with_format(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        tag = 1
        res = [torch_npu.npu_format_cast(torch.ones(2, 2).to(f"npu:{rank}"), 29)]
        xs = [shared_tensors.to(f"npu:{rank}")]
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst, tag).wait()
            c2p.put((dst, xs[0].cpu()))
        else:
            pg.recv(res, src, tag).wait()
            c2p.put((src, res[0].cpu()))

    @classmethod
    def _test_send_recv_group_send_with_format(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        tag = 1
        res = [torch.ones(2, 2).to(f"npu:{rank}")]
        xs = [torch_npu.npu_format_cast(shared_tensors.to(f"npu:{rank}"), 29)]
        dst = 0
        src = 1
        if src == rank:
            pg.send(xs, dst, tag).wait()
            c2p.put((dst, xs[0].cpu()))
        else:
            pg.recv(res, src, tag).wait()
            c2p.put((src, res[0].cpu()))

    def _test_multiprocess(self, f, shared_tensors, init_pg, ws=0):
        if not ws:
            ws = self.world_size
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        expected = 0
        result = 1
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, shared_tensors, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(2):
            pid, output = c2p.get()
            if pid == 0:
                expected = output
            else:
                result = output

        self.assertEqual(
            expected,
            result,
            (
                "Expect rank {} to receive tensor {} but got {}."
            ).format(pid, expected, result)
        )

        for _ in range(2):
            p2c.put(0)

        for p in ps:
            p.join(2)

    @skipIfUnsupportMultiNPU(2)
    def test_send_recv_hccl_dist(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_dist,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_dist_hccl)
        
    @skipIfUnsupportMultiNPU(4)
    def test_send_recv_hccl_dist_with_p2p(self):
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "20"}):
            world_size = 4
            self._test_multiprocess(
                HcclSendRecvDistTest._test_send_recv_dist_with_p2p,
                torch.randn(2, 2),
                HcclSendRecvDistTest._init_dist_hccl,
                world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_send_recv_hccl_dist_with_format(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_dist_recv_with_format,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_dist_hccl)

        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_dist_send_with_format,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_send_recv_hccl_group(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_group,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_pg_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_send_recv_hccl_group_with_format(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_group_recv_with_format,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_pg_hccl)

        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_group_send_with_format,
            torch.randn(2, 2),
            HcclSendRecvDistTest._init_pg_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_send_recv_hccl_bool(self):
        self._test_multiprocess(
            HcclSendRecvDistTest._test_send_recv_dist,
            torch.randn(2, 2) > 0.5,
            HcclSendRecvDistTest._init_dist_hccl)


if __name__ == '__main__':
    run_tests()
