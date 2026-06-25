from enum import IntEnum, unique

import os
import unittest
import torch
import torch.distributed as c10d
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


@unique
class Format(IntEnum):
    NCHW = 0
    ND = 2
    NC1HWC0 = 3
    NZ = 29


class ProcessGroupHCCLTest(TestCase):

    world_size = 2

    def setUp(self):
        if torch_npu.npu.device_count() < 2:
            raise unittest.SkipTest("HCCL test requires 2+ NPUs")

    @classmethod
    def _init_pg_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist.new_group([0, 1])

    def _test_multiprocess(self, f, shared_tensors, init_pg, n_output):
        ws = self.world_size
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, shared_tensors, ws, init_pg, c2p, p2c))

            p.start()
            ps.append(p)

        for _ in range(ws * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                (
                    "Expect rank {} to receive tensor {} but got {}."
                ).format(pid, expected, result)
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode.
    @classmethod
    def _test_broadcast_process(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        dtype_name = None
        if isinstance(shared_tensors, tuple):
            shared_tensors, dtype_name = shared_tensors
        pg = init_pg(rank, world_size)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]
        if dtype_name is not None:
            expected = shared_tensors[0].to(torch.float32)
            result = xs[0].to(torch.float32).cpu()
        pg.broadcast(xs).wait()
        if dtype_name is not None:
            expected = shared_tensors[0].to(torch.float32)
            result = xs[0].to(torch.float32).cpu()
        else:
            expected = shared_tensors[0]
            result = xs[0].to("cpu")
        c2p.put((rank, expected, result))
        p2c.get()

    def test_shared_broadcast_hccl(self):
        self._test_multiprocess(
            ProcessGroupHCCLTest._test_broadcast_process,
            [torch.ones(2, 2) * i for i in range(self.world_size)],
            ProcessGroupHCCLTest._init_pg_hccl,
            1)

    @classmethod
    def _test_allreduce_process(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

    def test_shared_allreduce_hccl(self):
        self._test_multiprocess(
            ProcessGroupHCCLTest._test_allreduce_process,
            [torch.ones(2, 2) for i in range(self.world_size)],
            ProcessGroupHCCLTest._init_pg_hccl,
            1)

    @classmethod
    def _test_allgather_process(
            cls, rank, shared_tensors, world_size, init_pg, c2p, p2c):
        dtype_name = None
        if isinstance(shared_tensors, tuple):
            shared_tensors, dtype_name = shared_tensors
        pg = init_pg(rank, world_size)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]
        if dtype_name is not None:
            dtype = getattr(torch, dtype_name, getattr(torch_npu, dtype_name, None))
            xs[0] = xs[0].to(dtype)
        ys = [[torch.zeros_like(xs[0]) for i in range(world_size)]]
        pg.allgather(ys, xs).wait()
        for i in range(world_size):
            if dtype_name is not None:
                expected = shared_tensors[i].to(torch.float32)
                result = ys[0][i].to(torch.float32).cpu()
            else:
                expected = shared_tensors[i]
                result = ys[0][i].to("cpu")
            c2p.put((rank, expected, result))

        p2c.get()

    def test_shared_allgather_hccl(self):
        self._test_multiprocess(
            ProcessGroupHCCLTest._test_allgather_process,
            [torch.ones(2, 2) * i for i in range(self.world_size)],
            ProcessGroupHCCLTest._init_pg_hccl,
            self.world_size)

    def test_shared_broadcast_hccl_uint_dtypes(self):
        base = torch.tensor([1, 2, 4, 8], dtype=torch.float16).reshape(2, 2)
        shared_tensors = [base * (2 ** (i * 4)) for i in range(self.world_size)]
        for dtype_name in ["uint16", "uint32", "uint64"]:
            dtype = getattr(torch, dtype_name, getattr(torch_npu, dtype_name, None))
            self.assertIsNotNone(dtype, f"{dtype_name} not available")
            self._test_multiprocess(
                ProcessGroupHCCLTest._test_broadcast_process,
                (shared_tensors, dtype_name),
                ProcessGroupHCCLTest._init_pg_hccl,
                1)


    def test_shared_allgather_hccl_uint_dtypes(self):
        base = torch.tensor([1, 2, 4, 8], dtype=torch.float16).reshape(2, 2)
        shared_tensors = [base * (2 ** (i * 4)) for i in range(self.world_size)]
        for dtype_name in ["uint16", "uint32", "uint64"]:
            dtype = getattr(torch, dtype_name, getattr(torch_npu, dtype_name, None))
            self.assertIsNotNone(dtype, f"{dtype_name} not available")
            self._test_multiprocess(
                ProcessGroupHCCLTest._test_allgather_process,
                (shared_tensors, dtype_name),
                ProcessGroupHCCLTest._init_pg_hccl,
                self.world_size)


class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch_npu.npu_format_cast(torch.empty([100, 1], dtype=torch.float).npu(), Format.NZ),
            torch.empty([200], dtype=torch.float).npu(),
            torch.empty([100], dtype=torch.float).npu(),
            torch.empty([50], dtype=torch.float).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [1792 * 4 + 1])
        expec_result = ([[0, 1, 2, 3]], [7169])
        self.assertEqual(expec_result, result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        expec_result = ([[0, 2], [1, 3], [4], [5]], [400, 400, 400, 400])
        self.assertEqual(expec_result, result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
            torch.empty([10], dtype=torch.float).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [40, 80])
        expec_result = ([[0], [1, 2], [3]], [40, 80, 80])
        self.assertEqual(expec_result, result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),

            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),

            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
            torch.empty([50], dtype=torch.float).npu(),
            torch.empty([25], dtype=torch.double).npu(),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [200, 400])
        expec_result = ([[0], [1], [2, 4], [3, 5], [6, 8], [7, 9]], [200, 200, 400, 400, 400, 400])
        self.assertEqual(expec_result, result)


if __name__ == '__main__':
    run_tests()
