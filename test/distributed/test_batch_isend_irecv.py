import unittest
import os
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcomBatchIsendIrecvTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist


    @classmethod
    def _test_batch_isend_irecv(cls, rank, world_size, init_pg, c2p, p2c):
        _ = init_pg(rank, world_size)
        recv_tensors = [None for _ in range(world_size)]
        expected_tensors = [None for _ in range(world_size)]
        p2p_op_list = []
        for src in range(0, world_size):
            send_tensor = torch.empty(rank + 1, rank + 1, rank + 1, dtype=torch.float).fill_(src).npu(rank)
            recv_tensors[src] = torch.empty(src + 1, src + 1, src + 1, dtype=torch.float).fill_(-1).npu(rank)
            expected_tensors[src] = torch.empty(src + 1, src + 1, src + 1, dtype=torch.float).fill_(rank)
            recv_op = dist.P2POp(dist.irecv, recv_tensors[src], src)
            p2p_op_list.append(recv_op)
            send_op = dist.P2POp(dist.isend, send_tensor, src)
            p2p_op_list.append(send_op)

        reqs = dist.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()
        c2p.put([[i for i in expected_tensors], [i.cpu() for i in recv_tensors]])
        p2c.get()


    def _test_multiprocess(self, f, init_pg, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            expected, received = c2p.get()
            self.assertEqual(expected, received)

        for _ in range(world_size):
            p2c.put(0)


        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_dist_batch_isend_irecv(self):
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(HcomBatchIsendIrecvTest._test_batch_isend_irecv,
                                    HcomBatchIsendIrecvTest._init_dist_hccl, world_size)


if __name__ == '__main__':
    run_tests()
