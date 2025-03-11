import unittest
from unittest.mock import patch
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclSilentCheckDistTest(TestCase):
    world_size = 2

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        # enable silent check
        os.environ['NPU_ASD_ENABLE'] = '2'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_silent_check_broadcast_fp32_dist(
            cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        if rank == 1:
            tensor = torch.ones((2, 2), dtype=torch.float).to(f"npu:{rank}")
        else:
            tensor = torch.full((2, 2), float('nan')).to(f"npu:{rank}")
        torch_npu._C._npu_set_module_train_state("train")
        torch_npu._C._npu_set_call_state("backward")
        pg.broadcast(tensor, src=1)
        c2p.put((rank, tensor.cpu()))

    def _test_multiprocess(self, f, init_pg, ws=0):
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
                args=(i, ws, init_pg, c2p, p2c))
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
    def test_silent_check_broadcast_fp32_dist(self):
        self._test_multiprocess(
            HcclSilentCheckDistTest._test_silent_check_broadcast_fp32_dist,
            HcclSilentCheckDistTest._init_dist_hccl)


if __name__ == '__main__':
    run_tests()
