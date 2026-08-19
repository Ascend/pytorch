import unittest
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclAlltoAllVCTest(TestCase):
    world_size_2p = 2
    world_size_4p = 4

    matrix_4p = [[1, 2, 0, 3],
                 [2, 1, 4, 0],
                 [0, 3, 1, 2],
                 [4, 0, 2, 1]]
    matrix_2p = [[1, 3],
                 [2, 1]]

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
    def _send_value(cls, i, j):
        return i * 10 + j

    @classmethod
    def _test_alltoallvc_2p(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        from torch_npu.distributed import all_to_all_vc
        matrix = cls.matrix_2p
        send_vals = []
        for j in range(world_size):
            send_vals += [cls._send_value(rank, j)] * matrix[rank][j]
        recv_total = sum(matrix[i][rank] for i in range(world_size))
        inp = torch.tensor(send_vals, dtype=torch.float32).npu()
        out = torch.zeros(recv_total, dtype=torch.float32).npu()
        all_to_all_vc(out, inp, matrix)
        expected = []
        for i in range(world_size):
            expected += [cls._send_value(i, rank)] * matrix[i][rank]
        c2p.put((rank, out.cpu(), 0, torch.tensor(expected)))

    @skipIfUnsupportMultiNPU(2)
    def test_alltoallvc_2p(self):
        self._test_multiprocess_2p(HcclAlltoAllVCTest._test_alltoallvc_2p,
                                   HcclAlltoAllVCTest._init_dist_hccl)

    @classmethod
    def _test_alltoallvc_4p(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        from torch_npu.distributed import all_to_all_vc
        matrix = cls.matrix_4p
        send_vals = []
        for j in range(world_size):
            send_vals += [cls._send_value(rank, j)] * matrix[rank][j]
        recv_total = sum(matrix[i][rank] for i in range(world_size))
        inp = torch.tensor(send_vals, dtype=torch.float32).npu()
        out = torch.zeros(recv_total, dtype=torch.float32).npu()
        all_to_all_vc(out, inp, matrix)
        expected = []
        for i in range(world_size):
            expected += [cls._send_value(i, rank)] * matrix[i][rank]
        c2p.put((rank, out.cpu(), 0, torch.tensor(expected)))

    @skipIfUnsupportMultiNPU(4)
    def test_alltoallvc_4p(self):
        self._test_multiprocess_4p(HcclAlltoAllVCTest._test_alltoallvc_4p,
                                   HcclAlltoAllVCTest._init_dist_hccl)

    @classmethod
    def _test_alltoallvc_4p_dtypes(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        from torch_npu.distributed import all_to_all_vc
        matrix = cls.matrix_4p
        send_vals = []
        for j in range(world_size):
            send_vals += [cls._send_value(rank, j)] * matrix[rank][j]
        recv_total = sum(matrix[i][rank] for i in range(world_size))
        all_ok = True
        for dtype in (torch.float32, torch.int32, torch.float16):
            for async_op in (False, True):
                inp = torch.tensor(send_vals, dtype=dtype).npu()
                out = torch.zeros(recv_total, dtype=dtype).npu()
                work = all_to_all_vc(out, inp, matrix, async_op=async_op)
                if async_op:
                    work.wait()
                torch.npu.synchronize()
                expected = []
                for i in range(world_size):
                    expected += [cls._send_value(i, rank)] * matrix[i][rank]
                ok = [int(v) for v in out.tolist()] == expected
                all_ok = all_ok and ok
        c2p.put((rank, torch.tensor([1 if all_ok else 0]), 1, ""))

    @skipIfUnsupportMultiNPU(4)
    def test_alltoallvc_4p_dtypes(self):
        self._test_multiprocess_4p(HcclAlltoAllVCTest._test_alltoallvc_4p_dtypes,
                                   HcclAlltoAllVCTest._init_dist_hccl)

    @classmethod
    def _test_alltoallvc_negative(cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        from torch_npu.distributed import all_to_all_vc

        cases = []

        inp = torch.tensor([10.0, 20.0], dtype=torch.float32).npu()
        out = torch.zeros(100, dtype=torch.float32).npu()
        m = [[100, 0], [0, 100]] if world_size == 2 else [
            [100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]]
        cases.append(("send_overflow", m, inp, out, "must equal send row sum"))

        inp = torch.zeros(100, dtype=torch.float32).npu()
        out = torch.zeros(2, dtype=torch.float32).npu()
        cases.append(("recv_overflow", m, inp, out, "must equal recv col sum"))

        inp = torch.zeros(10, dtype=torch.float32).npu()
        out = torch.zeros(10, dtype=torch.float32).npu()
        m = [[-1, 2], [2, 2]] if world_size == 2 else [
            [-1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        cases.append(("negative", m, inp, out, "is negative"))

        inp = torch.zeros(4, dtype=torch.float32).npu()
        out = torch.zeros(4, dtype=torch.float16).npu()
        m = [[2, 2], [2, 2]] if world_size == 2 else [
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        cases.append(("dtype_mismatch", m, inp, out, "must match"))

        inp = torch.zeros(4, dtype=torch.float32).npu()
        out = torch.zeros(4, dtype=torch.float32).npu()
        cases.append(("wrong_size", [[1, 1, 1], [1, 1, 1], [1, 1, 1]], inp, out, "rows (group size)"))

        inp = torch.zeros(4, dtype=torch.float32).npu()
        out = torch.zeros(4, dtype=torch.float32).npu()
        if world_size == 2:
            jag = [[1, 2, 3], [4]]
        else:
            jag = [[1, 2, 3], [4], [1, 2], [3, 4, 5]]
        cases.append(("jagged", jag, inp, out, "must be a square"))

        inp = torch.tensor(1.0, device="npu")
        out = torch.tensor(1.0, device="npu")
        m = [[1, 0], [0, 1]] if world_size == 2 else [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        cases.append(("scalar", m, inp, out, "Scalar tensors"))

        buf = torch.zeros(8, dtype=torch.float32).npu()
        m = [[4, 4], [4, 4]] if world_size == 2 else [
            [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        cases.append(("inplace", m, buf, buf, "in-place"))

        for label, matrix, inp, out, expect_sub in cases:
            err = ""
            try:
                all_to_all_vc(out, inp, matrix)
            except (RuntimeError, ValueError) as e:
                err = str(e)
            ok = expect_sub in err
            status = f"NEGATIVE OK [{label}]" if ok else \
                f"NEGATIVE FAIL [{label}]: expected '{expect_sub}' but got: {err}"
            c2p.put((rank, status))

    @skipIfUnsupportMultiNPU(2)
    def test_alltoallvc_negative_2p(self):
        self._test_multiprocess_err(self.world_size_2p, HcclAlltoAllVCTest._test_alltoallvc_negative,
                                    HcclAlltoAllVCTest._init_dist_hccl)

    @skipIfUnsupportMultiNPU(4)
    def test_alltoallvc_negative_4p(self):
        self._test_multiprocess_err(self.world_size_4p, HcclAlltoAllVCTest._test_alltoallvc_negative,
                                    HcclAlltoAllVCTest._init_dist_hccl)

    def _test_multiprocess_2p(self, f, init_pg):
        self._run_positive(self.world_size_2p, f, init_pg)

    def _test_multiprocess_4p(self, f, init_pg):
        self._run_positive(self.world_size_4p, f, init_pg)

    def _run_positive(self, ws, f, init_pg):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ws)
        p2c = ctx.Queue(ws)
        ps = []
        for i in range(ws):
            p = ctx.Process(target=f, args=(i, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)
        for _ in range(ws):
            rank, output, cout, exp = c2p.get()
            if cout == 0:
                self.assertEqual(output, exp, f"rank {rank} expected {exp} but got {output}")
            elif cout == 1:
                val = int(output.item()) if hasattr(output, 'item') else int(output[0])
                self.assertEqual(val, 1, f"rank {rank} dtype/async cases failed")
        for _ in range(ws):
            p2c.put(0)
        for p in ps:
            p.join(10)

    def _test_multiprocess_err(self, ws, f, init_pg):
        n_cases = 8
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ws * n_cases)
        p2c = ctx.Queue(ws)
        ps = []
        for i in range(ws):
            p = ctx.Process(target=f, args=(i, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)
        for _ in range(ws * n_cases):
            rank, status = c2p.get()
            self.assertNotIn("FAIL", status, f"rank {rank}: {status}")
        for _ in range(ws):
            p2c.put(0)
        for p in ps:
            p.join(10)


if __name__ == '__main__':
    run_tests()
