import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclAlltoAllTest(TestCase):
    world_size_2p = 2
    world_size_4p = 4

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        os.environ['HCCL_ALGO'] = "level0:fullmesh;level1:fullmesh"
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_alltoall_2p(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(2) + rank * 2
        input1 = input1.float().npu()
        input1_list = list(input1.chunk(2))
        output = torch.empty(2).float().npu()
        output_list = list(output.chunk(2))
        cout = 0
        pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout))
        p2c.get()

    @classmethod
    def _test_alltoall_2p_size(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input_list = [(torch.zeros(rank + 1, 1) + rank).float().npu() for i in range(2)]
        output_list = [torch.empty(i + 1, 1).float().npu() for i in range(2)]
        for i in range(2):
            input_list[i] = torch_npu.npu_format_cast(input_list[i], 29)
            output_list[i] = torch_npu.npu_format_cast(output_list[i], 29)
        cout = 1
        pg.all_to_all(output_list, input_list)
        for i in range(2):
            if torch_npu.get_npu_format(output_list[i]) != 29:
                raise RuntimeError("format error!")

        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout))
        p2c.get()

    @classmethod
    def _test_alltoall_2p_size_nd(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input_list = [(torch.zeros(rank + 1, rank + 1) + rank).float().npu() for i in range(2)]
        output_list = [torch.empty(i + 1, i + 1).float().npu() for i in range(2)]
        cout = 2
        pg.all_to_all(output_list, input_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout))
        p2c.get()

    def _test_multiprocess_2p(self, f, init_pg):
        ws = self.world_size_2p
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ws):
            rank, output, cout = c2p.get()
            expected = []
            if cout == 0:
                expected = list((torch.arange(2) * 2 + rank).chunk(2))
            elif cout == 1:
                expected = [(torch.zeros(i + 1, 1) + i).float() for i in range(2)]
            elif cout == 2:
                expected = [(torch.zeros(i + 1, i + 1) + i).float() for i in range(2)]

            self.assertEqual(
                output,
                expected,
                (
                    "rank {} Expect receive tensor {} but got {}."
                ).format(rank, expected, output)
            )

        for _ in range(ws):
            p2c.put(0)


        for p in ps:
            p.join(2)

    @skipIfUnsupportMultiNPU(2)
    def test_alltoall_2p_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllTest._test_alltoall_2p,
            HcclAlltoAllTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_alltoall_2p_size_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllTest._test_alltoall_2p_size,
            HcclAlltoAllTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(2)
    def test_alltoall_2p_size_nd_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllTest._test_alltoall_2p_size_nd,
            HcclAlltoAllTest._init_dist_hccl)

    @classmethod
    def _test_alltoall_4p(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.float().npu()
        input1_list = list(input1.chunk(4))
        output = torch.empty(4).float().npu()
        output_list = list(output.chunk(4))
        cout = 0
        pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout, [1, 1, 1, 1]))
        p2c.get()

    @classmethod
    def _test_alltoall_4p_size(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(7) + rank * 4
        input1 = input1.float().npu()
        x = 7
        if rank == 1:
            x = 9
        elif rank == 3:
            x = 5
        output = torch.empty(x).float().npu()
        inputsize = [[1, 2, 2, 2], [1, 3, 2, 1], [2, 3, 1, 1], [3, 1, 2, 1]]
        outsize = [[1, 1, 2, 3], [2, 3, 3, 1], [2, 2, 1, 2], [2, 1, 1, 1]]

        input1_list = list(input1.split(inputsize[rank]))
        output_list = list(output.split(outsize[rank]))
        cout = 1
        pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout, outsize[rank]))
        p2c.get()

    @classmethod
    def _test_alltoall_4p_size_nd(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input_list = [(torch.zeros(rank + 1, rank + 1) + rank).float().npu() for i in range(4)]
        output_list = [torch.empty(i + 1, i + 1).float().npu() for i in range(4)]
        cout = 2
        pg.all_to_all(output_list, input_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout, None))
        p2c.get()

    expected_dict = [
        [0, 4, 8, 9, 12, 13, 14],
        [1, 2, 5, 6, 7, 10, 11, 12, 15],
        [3, 4, 8, 9, 13, 16, 17],
        [5, 6, 10, 14, 18]
    ]

    def _test_multiprocess_4p(self, f, init_pg):
        ws = self.world_size_4p
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        p2c = ctx.Queue(4)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ws):
            rank, output, cout, outsize = c2p.get()
            expected = []
            if cout == 0:
                expected = list((torch.arange(4) * 4 + rank).split(outsize))
            elif cout == 1:
                expected = list(torch.tensor(self.expected_dict[rank]).split(outsize))
            elif cout == 2:
                expected = [(torch.zeros(i + 1, i + 1) + i).float() for i in range(4)]

            self.assertEqual(
                output,
                expected,
                (
                    "rank {} Expect receive tensor {} but got {}."
                ).format(rank, expected, output)
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(4)

    @skipIfUnsupportMultiNPU(4)
    def test_alltoall_4p_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllTest._test_alltoall_4p,
            HcclAlltoAllTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(4)
    def test_alltoall_4p_size_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllTest._test_alltoall_4p_size,
            HcclAlltoAllTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(4)
    def test_alltoall_4p_size_nd_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllTest._test_alltoall_4p_size_nd,
            HcclAlltoAllTest._init_dist_hccl)


if __name__ == '__main__':
    run_tests()
