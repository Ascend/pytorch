import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import torch_npu


class HcclReduceTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_all_gather(cls, rank, input1, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input1 = input1.npu()
        gather_tensor = [torch.empty_like(input1) for _ in range(world_size)]
        pg.all_gather(gather_tensor, input1)
        c2p.put((rank, [tensor.cpu() for tensor in gather_tensor]))
        pg.barrier()

    @classmethod
    def _test_all_gather_base(cls, rank, input1, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input1 = input1.npu()
        shape = list(input1.size())
        shape[0] = shape[0] * world_size
        gather_tensor = torch.empty(shape, device=input1.device, dtype=input1.dtype)
        pg._all_gather_base(gather_tensor, input1)
        c2p.put((rank, gather_tensor.cpu()))
        pg.barrier()

    @classmethod
    def _test_all_gather_togather(cls, rank, input1, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        
        input1 = input1.npu()
        gather_tensor = torch.empty((world_size, *list(input1.size())), device=input1.device, dtype=input1.dtype)
        pg.all_gather_togather(gather_tensor, input1)
        c2p.put((rank, gather_tensor.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expected,
                            ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, op=dist.all_gather):
        if op == dist.all_gather:
            return [inputs.cpu()]*world_size
        elif op == dist.all_gather_togather:
            shape = [1]*len(inputs.size())
            return torch.unsqueeze(inputs.cpu(), 0).repeat((world_size, *shape))
        elif op == dist._all_gather_base:
            return torch.cat((inputs.cpu(), inputs.cpu()))
        else:
            ValueError("Unsupported op `{}`"%(str(op)))
        return

    def test_all_gather_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8, np.bool]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size)
                self._test_multiprocess(HcclReduceTest._test_all_gather,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

    def test_all_gather_togather_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8, np.bool]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size, dist.all_gather_togather)
                self._test_multiprocess(HcclReduceTest._test_all_gather_togather,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

    def test_all_gather_base_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size, dist._all_gather_base)
                self._test_multiprocess(HcclReduceTest._test_all_gather_base,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

if __name__ == '__main__':
    run_tests()
