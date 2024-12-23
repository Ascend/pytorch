import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclGatherTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_gather(cls, test_data, world_size, init_pg, c2p, p2c):
        rank, input1, output1 = test_data
        pg = init_pg(rank, world_size)
        dst = 0
        input1 = input1.npu()
        output1 = [i.npu() for i in output1]
        if rank == dst:
            pg.gather(input1, output1, dst=dst)
        else:
            pg.gather(input1, [], dst=dst)
        if rank == 0:
            c2p.put((rank, dst, [t.cpu() for t in output1]))
        else:
            c2p.put((rank, dst, []))
        p2c.get()

    @classmethod
    def _test_gather_object(cls, test_data, world_size, init_pg, c2p, p2c):
        rank, input1, output1 = test_data
        pg = init_pg(rank, world_size)
        dst = 0
        output1 = [i for i in output1]
        if rank == dst:
            pg.gather_object(input1, output1, dst=dst)
        else:
            pg.gather_object(input1, [], dst=dst)
        if rank == 0:
            c2p.put((rank, dst, [t.cpu() for t in output1]))
        else:
            c2p.put((rank, dst, []))
        p2c.get()


    def _test_multiprocess(self, f, init_pg, proc_data, world_size):
        input1, output1, expected = proc_data
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=((i, input1, output1), world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, output = c2p.get()
            if rank == 0:
                for i, j in zip(output, expected):
                    self.assertEqual(i, j,
                                     ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_gather_dist(self):
        ranks = [2]
        dtypes = [torch.float32, torch.float16, torch.int32, torch.uint8]
        for rank in ranks:
            for _dtype in dtypes:
                _input = torch.tensor([rank], dtype=_dtype)
                _output = [torch.empty([1], dtype=_dtype) for _ in range(rank)]
                _expected = [torch.tensor([rank], dtype=_dtype) for _ in range(rank)]
                proc_data = (_input, _output, _expected)
                self._test_multiprocess(HcclGatherTest._test_gather,
                                        HcclGatherTest._init_dist_hccl, proc_data, rank)

    @skipIfUnsupportMultiNPU(2)
    def test_gather_object_dist(self):
        ranks = [2]
        dtypes = [torch.float32, torch.float16, torch.int32, torch.uint8]
        for rank in ranks:
            for _dtype in dtypes:
                _input = torch.tensor([rank], dtype=_dtype)
                _output = [torch.empty([1], dtype=_dtype) for _ in range(rank)]
                _expected = [torch.tensor([rank], dtype=_dtype) for _ in range(rank)]
                proc_data = (_input, _output, _expected)
                self._test_multiprocess(HcclGatherTest._test_gather_object,
                                        HcclGatherTest._init_dist_hccl, proc_data, rank)


if __name__ == '__main__':
    run_tests()
