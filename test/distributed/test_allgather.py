import unittest
import os
from random import randint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

import torch_npu.distributed
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclAllGatherTestBase(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expected,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _test_multiprocess_with_inputlist(self, f, init_pg, cpu_expected, inputlist, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        gather_tensor = list()
        for input_tensor in inputlist:
            gather_tensor.append(torch.empty_like(input_tensor, device="cpu"))
        
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, inputlist[i].cpu(), gather_tensor, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, cpu_expected,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, cpu_expected, output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, op=dist.all_gather):
        if op == dist.all_gather:
            return [inputs.cpu()] * world_size
        elif op == dist._all_gather_base:
            return torch.cat((inputs.cpu(), inputs.cpu()))
        elif op == dist.all_gather_into_tensor:
            return torch.cat((inputs.cpu(), inputs.cpu()))
        elif op == torch_npu.distributed.all_gather_into_tensor_uneven:
            return torch.cat((inputs.cpu(), inputs.cpu()))
        else:
            raise ValueError("Unsupported op `{}`" % (str(op)))


class HcclAllGatherTest(HcclAllGatherTestBase):

    @classmethod
    def _test_all_gather(cls, rank, input1, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = input1.npu()
        gather_tensor = [torch.empty_like(input1) for _ in range(world_size)]
        pg.all_gather(gather_tensor, input1)
        c2p.put((rank, [tensor.cpu() for tensor in gather_tensor]))
        pg.barrier()
        p2c.get()

    @classmethod
    def _test_all_gather_different_shape(cls, rank, input1, gather_tensor, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = input1.npu()
        gather_tensor = [tensor.npu() for tensor in gather_tensor]
        pg.all_gather(gather_tensor, input1)
        c2p.put((rank, [tensor.cpu() for tensor in gather_tensor]))
        pg.barrier()
        p2c.get()

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8, np.bool_]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                if shape[0] == np.bool_:
                    continue
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size)
                self._test_multiprocess(HcclAllGatherTest._test_all_gather,
                                        HcclAllGatherTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_dist_different_shape(self):
        ranks = [2]
        format_list = [0, 2, 3, 29]

        def get_random_input(dim=1, max_value=20):
            shape_list = list()
            for _ in range(dim):
                shape_list.append(randint(1, max_value))
            return create_common_tensor([np.float32, format_list[randint(0, 3)], shape_list], -10, 10)
        
        for world_size in ranks:
            cpu_excepted_result = list()
            npu_excepted_result = list()
            for _ in range(world_size):
                cpu_input, npu_input = get_random_input(randint(1, 5))
                cpu_excepted_result.append(cpu_input)
                npu_excepted_result.append(npu_input)
            self._test_multiprocess_with_inputlist(HcclAllGatherTest._test_all_gather_different_shape,
                                                   HcclAllGatherTest._init_dist_hccl, cpu_excepted_result, npu_excepted_result, world_size)


if __name__ == '__main__':
    run_tests()
