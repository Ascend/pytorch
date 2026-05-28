import os
import socket
import time
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

SUB_COMM_LOG_KEYWORD = "Create sub hccl comm by hcclCreateSubCommConfig success"
GLOBAL_COMM_LOG_KEYWORD = "Create hccl comm by hcclCommInitRootInfoConfig success"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hccl_logs")

STDERR_FD = 2


class StderrCapture:
    def __init__(self, label=""):
        self.original_stderr_fd = None
        self.log_path = None
        self.captured = ""
        self.label = label

    def __enter__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"hccl_{self.label}_{timestamp}_{os.getpid()}.log" if self.label else f"hccl_{timestamp}_{os.getpid()}.log"
        self.log_path = os.path.join(LOG_DIR, filename)
        self.original_stderr_fd = os.dup(STDERR_FD)
        self.log_fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(self.log_fd, STDERR_FD)
        return self

    def __exit__(self, *args):
        os.fsync(STDERR_FD)
        os.dup2(self.original_stderr_fd, STDERR_FD)
        os.close(self.original_stderr_fd)
        os.close(self.log_fd)
        with open(self.log_path, 'r') as f:
            self.captured = f.read()


class SubCommDerivationTest(TestCase):
    world_size = 2

    def setUp(self):
        if torch_npu.npu.device_count() < 2:
            raise unittest.SkipTest("HCCL test requires 2+ NPUs")

    @classmethod
    def _find_free_port(cls):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    _socket_port_counter = 0

    @classmethod
    def _alloc_socket_port_range(cls):
        cls._socket_port_counter += 1
        base = 16600 + cls._socket_port_counter * 100
        return f"{base}-{base + 99}"

    @classmethod
    def _init_pg_hccl(cls, rank, world_size, port, socket_port_range):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        os.environ['HCCL_NPU_SOCKET_PORT_RANGE'] = socket_port_range
        torch_npu._C._logging._LogContext.GetInstance().setLogs({"torch.distributed": 20})
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _cleanup_pg(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _test_multiprocess(self, f, shared_tensors, n_output=1, check_log=False):
        ws = self.world_size
        port = self._find_free_port()
        socket_port_range = self._alloc_socket_port_range()
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ws)
        p2c = ctx.Queue(ws)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, shared_tensors, ws, port, socket_port_range, c2p, p2c))
            p.start()
            ps.append(p)

        results = []
        for _ in range(ws * n_output):
            results.append(c2p.get())

        for rank_val, expected, result, _ in results:
            self.assertEqual(
                expected,
                result,
                f"Expect rank {rank_val} to receive tensor {expected} but got {result}."
            )

        if check_log:
            log_lines = [log_text for _, _, _, log_text in results if log_text]
            for rank_idx, log_text in enumerate(log_lines):
                self.assertIn(
                    SUB_COMM_LOG_KEYWORD,
                    log_text,
                    f"Rank {rank_idx}: expected sub comm creation log '{SUB_COMM_LOG_KEYWORD}' "
                    f"not found in captured stderr. Log dir: {LOG_DIR}"
                )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(5)

    @classmethod
    def _test_sub_comm_allreduce_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        with StderrCapture(label=f"allreduce_rank{rank}") as cap:
            dist.all_reduce(xs[0], group=sub_pg, op=dist.ReduceOp.SUM)

        expected = torch.ones(2, 2) * world_size
        c2p.put((rank, expected, xs[0].to("cpu"), cap.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_allreduce(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_allreduce_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_multiple_sub_comms_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        tp_pg = dist.new_group(backend='hccl', ranks=ranks)
        pp_pg = dist.new_group(backend='hccl', ranks=ranks)
        dp_pg = dist.new_group(backend='hccl', ranks=ranks)

        tensor_tp = xs[0].clone()
        tensor_pp = xs[0].clone()
        tensor_dp = xs[0].clone()

        with StderrCapture(label=f"multi_comms_rank{rank}") as cap:
            dist.all_reduce(tensor_tp, group=tp_pg, op=dist.ReduceOp.SUM)

        dist.all_reduce(tensor_pp, group=pp_pg, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_dp, group=dp_pg, op=dist.ReduceOp.SUM)

        expected = torch.ones(2, 2) * world_size
        c2p.put((rank, expected, tensor_tp.to("cpu"), cap.captured))
        c2p.put((rank, expected, tensor_pp.to("cpu"), ""))
        c2p.put((rank, expected, tensor_dp.to("cpu"), ""))
        p2c.get()

        dist.destroy_process_group(tp_pg)
        dist.destroy_process_group(pp_pg)
        dist.destroy_process_group(dp_pg)
        cls._cleanup_pg()

    def test_multiple_sub_comms(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_multiple_sub_comms_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            3,
            check_log=True)

    @classmethod
    def _test_sub_comm_allgather_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        ys = [torch.zeros(xs[0].shape, dtype=xs[0].dtype, device=xs[0].device) for _ in range(world_size)]

        with StderrCapture(label=f"allgather_rank{rank}") as cap:
            dist.all_gather(ys, xs[0], group=sub_pg)

        for i in range(world_size):
            expected = shared_tensors[i]
            result = ys[i].to("cpu")
            log_text = cap.captured if i == 0 else ""
            c2p.put((rank, expected, result, log_text))

        p2c.get()
        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_allgather(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_allgather_process,
            [torch.ones(2, 2) * i for i in range(self.world_size)],
            self.world_size,
            check_log=True)

    @classmethod
    def _test_sub_comm_broadcast_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        with StderrCapture(label=f"broadcast_rank{rank}") as cap:
            dist.broadcast(xs[0], src=0, group=sub_pg)

        expected = shared_tensors[0]
        result = xs[0].to("cpu")
        c2p.put((rank, expected, result, cap.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_broadcast(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_broadcast_process,
            [torch.ones(2, 2) * i for i in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_global_and_sub_comm_coexist_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        tensor1 = xs[0].clone()
        tensor2 = xs[0].clone()

        dist.all_reduce(tensor1, op=dist.ReduceOp.SUM)

        with StderrCapture(label=f"coexist_rank{rank}") as cap:
            dist.all_reduce(tensor2, group=sub_pg, op=dist.ReduceOp.SUM)

        expected = torch.ones(2, 2) * world_size
        c2p.put((rank, expected, tensor1.to("cpu"), cap.captured))
        c2p.put((rank, expected, tensor2.to("cpu"), ""))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_global_and_sub_comm_coexist(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_global_and_sub_comm_coexist_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            2,
            check_log=True)

    @classmethod
    def _test_sub_comm_with_custom_config_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {
            "group_name": "custom_config_group",
            "hccl_buffer_size": 300,
        }
        sub_pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)

        with StderrCapture(label=f"custom_config_rank{rank}") as cap:
            dist.all_reduce(xs[0], group=sub_pg, op=dist.ReduceOp.SUM)

        expected = torch.ones(2, 2) * world_size
        c2p.put((rank, expected, xs[0].to("cpu"), cap.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_with_custom_config(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_with_custom_config_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_sub_comm_async_ops_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        with StderrCapture(label=f"async_rank{rank}") as cap:
            work = dist.all_reduce(xs[0], group=sub_pg, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        expected = torch.ones(2, 2) * world_size
        c2p.put((rank, expected, xs[0].to("cpu"), cap.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_async_ops(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_async_ops_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_sub_comm_destroy_and_recreate_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))

        sub_pg1 = dist.new_group(backend='hccl', ranks=ranks)
        with StderrCapture(label=f"recreate1_rank{rank}") as cap1:
            dist.all_reduce(xs[0], group=sub_pg1, op=dist.ReduceOp.SUM)
        dist.destroy_process_group(sub_pg1)

        torch_npu.npu.synchronize()

        sub_pg2 = dist.new_group(backend='hccl', ranks=ranks)
        with StderrCapture(label=f"recreate2_rank{rank}") as cap2:
            dist.all_reduce(xs[0], group=sub_pg2, op=dist.ReduceOp.SUM)

        expected = torch.ones(2, 2) * world_size * world_size
        c2p.put((rank, expected, xs[0].to("cpu"), cap1.captured + cap2.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg2)
        cls._cleanup_pg()

    def test_sub_comm_destroy_and_recreate(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_destroy_and_recreate_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_sub_comm_reduce_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        ranks = list(range(world_size))
        sub_pg = dist.new_group(backend='hccl', ranks=ranks)

        with StderrCapture(label=f"reduce_rank{rank}") as cap:
            dist.reduce(xs[0], dst=0, op=dist.ReduceOp.SUM, group=sub_pg)

        if rank == 0:
            expected = torch.ones(2, 2) * world_size
        else:
            expected = xs[0].to("cpu")
        c2p.put((rank, expected, xs[0].to("cpu"), cap.captured))
        p2c.get()

        dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_sub_comm_reduce(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_sub_comm_reduce_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)

    @classmethod
    def _test_partial_ranks_sub_comm_process(
            cls, rank, shared_tensors, world_size, port, socket_port_range, c2p, p2c):
        pg = cls._init_pg_hccl(rank, world_size, port, socket_port_range)
        xs = [shared_tensors[rank].to(f"npu:{rank}")]

        partial_ranks = [0, 1]

        sub_pg = dist.new_group(backend='hccl', ranks=partial_ranks)

        if rank in partial_ranks:
            with StderrCapture(label=f"partial_rank{rank}") as cap:
                dist.all_reduce(xs[0], group=sub_pg, op=dist.ReduceOp.SUM)
            expected = torch.ones(2, 2) * len(partial_ranks)
        else:
            cap = None
            expected = xs[0].to("cpu")

        log_text = cap.captured if cap else ""
        c2p.put((rank, expected, xs[0].to("cpu"), log_text))
        p2c.get()

        if rank in partial_ranks:
            dist.destroy_process_group(sub_pg)
        cls._cleanup_pg()

    def test_partial_ranks_sub_comm(self):
        self._test_multiprocess(
            SubCommDerivationTest._test_partial_ranks_sub_comm_process,
            [torch.ones(2, 2) for _ in range(self.world_size)],
            1,
            check_log=True)


if __name__ == '__main__':
    run_tests()
