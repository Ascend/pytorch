import unittest
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SkipIfNotGteCANNVersion
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcomCoalescedManagerTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_all_reduce_coalesced_manager_hccl(cls, rank, input1_list, world_size, init_pg, c2p, reduce_op=dist.ReduceOp.SUM, done_event=None):
        dist_group = init_pg(rank, world_size)
        process_group = dist.distributed_c10d._get_default_group()
        dst = 0
        device = torch.device(f"npu:{rank:d}")
        input1_list = [input1.npu() for input1 in input1_list]
        with _coalescing_manager(group=process_group, device=device, async_ops=True) as cm:
            for tensor in input1_list:
                dist.all_reduce(tensor)
        cm.wait()
        c2p.put((rank, dst, [input1.cpu() for input1 in input1_list]))

        if done_event is not None:
            done_event.wait(timeout=5)


    def _test_multiprocess(self, f, init_pg, expected_list, input1_list, world_size, reduce_op=dist.ReduceOp.SUM):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        done_event = ctx.Event()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [input1.cpu() for input1 in input1_list], world_size, init_pg, c2p, reduce_op, done_event))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, output_list = c2p.get()
            output_num = len(output_list)
            if rank == dst:
                for i in range(output_num):
                    self.assertEqual(output_list[i], expected_list[i],
                                    "rank {} world_size {} dtype {} shape {} Expect receive tensor {} but got {}.".format(
                                        rank, world_size, expected_list[i].dtype, expected_list[i].shape, expected_list[i], output_list[i]))
        done_event.set()
        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, dtype=np.float32, reduce_op=dist.ReduceOp.SUM):
        expected = 0
        for _ in range(world_size):
            expected += inputs

        if reduce_op == dist.ReduceOp.AVG:
            if dtype in [np.int32, np.int8, np.int64, np.uint8]:
                expected //= world_size
            else:
                expected /= world_size

        return expected

    @skipIfUnsupportMultiNPU(2)
    @SkipIfNotGteCANNVersion("8.5.0")
    def test_all_reduce_coalesced_manager_hccl(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [2, 3, 16]]]
        op_times = 5
        input1_list = []
        expected_list = []
        for world_size in ranks:
            for shape in shape_format:
                for _ in range(op_times):
                    exp_input, input1 = create_common_tensor(shape, -10, 10)
                    expected = self._construct_excepted_result(exp_input, world_size)
                    input1_list.append(input1)
                    expected_list.append(expected)
                self._test_multiprocess(HcomCoalescedManagerTest._test_all_reduce_coalesced_manager_hccl,
                                        HcomCoalescedManagerTest._init_dist_hccl, expected_list, input1_list, world_size)


if __name__ == '__main__':
    run_tests()