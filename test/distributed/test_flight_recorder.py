import os
import json
import pickle
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest import mock, skipIf

import torch
import torch.distributed as c10d
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests

import torch_npu


class HCCLTraceTestBase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ[
            "TORCH_HCCL_ENABLE_TIMING"
        ] = "0"  # see 'timing_enabled' parametrized tests
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_HCCL_DUMP_ON_TIMEOUT"] = "1"
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_HCCL_DEBUG_INFO_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_HCCL_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
        self._spawn_processes()

    @classmethod
    def _run(
        cls,
        parent_conn,
        rank: int,
        test_name: str,
        file_name: str,
        parent_pipe,
        **kwargs,
    ) -> None:
        cls.parent = parent_conn
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        return torch.device("npu", self.rank_to_NPU[self.rank][0])

    def _join_processes(self, fn):
        # We need to patch sys.exit() as skip_if will use sys.exit() and
        # the exit code from the this process will not be catched.
        with mock.patch("sys.exit") as exit_mock:
            fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self.children_pipes = []
        parent_pipes = []
        for i in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        self._start_processes(wrap)

    def _create_process_group_hccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "hccl", world_size=self.world_size, rank=self.rank, store=store
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @property
    def rank_to_NPU(self):
        # return rank to NPU map
        return {i: [i] for i in range(self.world_size)}

    def _trace_basename(self):
        # we pass the base to the env, and the dump util will append rank
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled):
        return "started" if timing_enabled else "scheduled"


class HCCLTraceTest(HCCLTraceTestBase):
    def _verify_trace(self, t, include_collectives, timing_enabled, is_json):
        ver = t["version"]
        self.assertEqual(ver, "2.4")
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        default_pg_info = pg_config["group_name_0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        pg_status = t["pg_status"]
        self.assertEqual(len(pg_status), 1)
        self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "2")
        self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "2")
        self.assertEqual(
            str(pg_status["0"]["last_started_collective"]),
            "2" if timing_enabled else "-1",
        )
        global_ranks = pg_config["group_name_0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        if include_collectives:
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            last = t[-1]
            self.assertEqual(last["process_group"], ("group_name_0", ""))
            self.assertEqual(last["state"], "completed")
            s = last["time_discovered_started_ns"]
            f = last["time_discovered_completed_ns"]
            self.assertEqual(last["record_id"], 1)
            self.assertIsNotNone(f)
            if timing_enabled:
                self.assertIsNotNone(s)
                self.assertTrue(s <= f)
            # we don't collect stack traces in JSON at the moment
            if not is_json:
                self.assertIn("test_flight_recorder.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            # HCCL_EXEC_TIMEOUT will impact watchdog timeout
            self.assertEqual(last["timeout_ms"], 3636000)
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            if timing_enabled:
                # very loose bounds, measured 0.036 ms on devnpu
                self.assertTrue(0 < last["duration_ms"] < 100)
            else:
                self.assertTrue("duration_ms" not in last)
        else:
            self.assertTrue("entries" not in t)

    @parametrize("timing_enabled", [False])
    @parametrize("include_collectives", [True, False])
    def test_short_json(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.npu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = json.loads(
            torch_npu._C._distributed_c10d._dump_hccl_trace_json(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(t, include_collectives, timing_enabled, True)
        dist.destroy_process_group()

    @parametrize("timing_enabled", [False])
    @parametrize("include_collectives", [True, False])
    def test_short_pickle(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.npu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = pickle.loads(
            torch_npu._C._distributed_c10d._dump_hccl_trace(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(
            t,
            include_collectives=include_collectives,
            timing_enabled=timing_enabled,
            is_json=True,
        )
        dist.destroy_process_group()

    def test_dump_pipe(self):
        if self.rank != self.MAIN_PROCESS_RANK:
            # now we need start heartbeatmonitor thread manually
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
            # makesure dump_pipe not heartbeat dump
            os.unsetenv("TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC")

        def open_file_with_timeout(file_path, mode, timeout=1.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(file_path):
                    return open(file_path, mode)
                time.sleep(0.1)
            raise FileNotFoundError

        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")

            dump_file = self._trace_name(rank=0)
            pipe_file = dump_file + ".pipe"
            with open_file_with_timeout(pipe_file, "w") as f:
                f.write("1\n")
            with open_file_with_timeout(dump_file, "rb", timeout=10.0) as f:
                # does not support profiling, so we use test_dump_pipe instead of all_reduce
                self.assertTrue("test_dump_pipe" in str(pickle.load(f)))

            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_hccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.npu.synchronize(device=device)
        self.parent.send("next")
        self.parent.recv()

    def test_long(self):
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            # test some other primitives to make sure
            # their strings are valid
            xs = [torch.ones(3, 4, device=device)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
        f.wait()
        torch.npu.synchronize(device=device)
        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        # profiling is not supported
        self.assertEqual(last["profiling_name"], "")
        self.assertEqual(last["state"], "completed")
        self.assertIn("test_flight_recorder.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        # timeout_ms adapt to npu
        self.assertEqual(last["timeout_ms"], 3636000)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
        dist.destroy_process_group()

    @skipIf(True, "profiling is not supported")
    def test_barrier_profiling(self):
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        f = pg.barrier()
        f = pg.allreduce(a)
        f.wait()
        torch.npu.synchronize(device=device)
        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 2)
        first = t[0]
        last = t[-1]
        self.assertEqual(first["profiling_name"], "hccl:all_reduce_barrier")
        self.assertEqual(last["profiling_name"], "hccl:all_reduce")
        dist.destroy_process_group()

    def test_trace_while_all_works_retired(self):
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        device = self.local_device
        # send more works than the buffer size to overwrite the previous entry
        for _ in range(12):
            a = [torch.ones(3, 4, device=device)]
            pg.broadcast(a).wait()
        torch.npu.synchronize(device=device)

        # wait for all works to be retired, we use sleep instead of pg._wait_for_pending_works()
        time.sleep(30)
        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        last = t[-1]
        self.assertEqual(last["retired"], True)
        self.assertEqual(last["state"], "completed")

    # timing_enabled is not supported
    @parametrize("timing_enabled", [False])
    @parametrize("only_active", [True, False])
    def test_trace_while_active(self, timing_enabled, only_active):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.npu.Event()
            e.record()
            if self.rank != 0:
                pg.allreduce(a).wait()
            e.synchronize()
            t = pickle.loads(
                torch_npu._C._distributed_c10d._dump_hccl_trace(onlyActive=only_active)
            )
            t = t["entries"]
            if only_active:
                if self.rank == 0:
                    self.assertEqual(len(t), 0)
                else:
                    self.assertEqual(len(t), 1)
            if not only_active:
                if self.rank == 0:
                    self.assertEqual(t[-1]["profiling_name"], "")
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["profiling_name"], "")
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )

            self.parent.send("next")
            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.npu.synchronize(device=device)

    @parametrize("timing_enabled", [False])
    def test_trace_while_stuck(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.npu.Event()
            e.record()

            def gather_trace():
                e.synchronize()
                # give the other thread some time to fill the npu buffer
                time.sleep(5)
                t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
                t = t["entries"]
                self.assertEqual(t[-1]["profiling_name"], "")
                if self.rank == 0:
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )
                    self.assertIsNone(t[-1]["time_discovered_completed_ns"])
                # this will eventually cause the missing rank 0
                # to continue which will unblock the non-zero ranks
                self.parent.send("next")

            if self.rank != 0:
                pg.allreduce(a).wait()
                th = threading.Thread(target=gather_trace)
                th.start()
                # fill the npu buffer, at around 1024 events
                # this will stall
                for _ in range(2000):
                    a = a + a
                th.join()
            else:
                gather_trace()

            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.npu.synchronize(device=device)

    @skipIf(True, "send_recv is not supported")
    @parametrize(
        "op_sizes_per_coalesce",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_batched_send_recv(self, op_sizes_per_coalesce, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's npu events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        num_coalesced_ops = 20
        ops_per_coalesce = len(op_sizes_per_coalesce)
        for _ in range(num_coalesced_ops):
            ops = []
            for input_sizes in op_sizes_per_coalesce:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    ops.append(dist.P2POp(dist.irecv, tensor, 1))
                elif self.rank == 1:
                    tensor *= 2
                    ops.append(dist.P2POp(dist.isend, tensor, 0))

            dist.batch_isend_irecv(ops).pop().wait()

        torch.npu.synchronize(device=self.local_device)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
        self.assertEqual(len(t["entries"]), num_coalesced_ops * (ops_per_coalesce + 1))

        expected_record_id = 0
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_coalesced_ops):
            first_op = seq * (ops_per_coalesce + 1)
            coalesced_op = first_op + ops_per_coalesce
            for p2p_op_idx, input_sizes in zip(
                range(first_op, coalesced_op, 1), op_sizes_per_coalesce
            ):
                # the indivudal ops inside the coalescing group the individual op metadata,
                # but not the timing info coming from the actual coalesced kernel
                profiling_name = (
                    "hccl:recv 0<-1" if self.rank == 0 else "hccl:send 1->0"
                )
                self.assertEqual(
                    t["entries"][p2p_op_idx]["record_id"], expected_record_id
                )
                expected_record_id += 1
                self.assertEqual(
                    t["entries"][p2p_op_idx]["profiling_name"], profiling_name
                )
                # we don't increment collective_seq_id for p2p ops.
                self.assertEqual(t["entries"][p2p_op_idx]["collective_seq_id"], 0)
                self.assertEqual(t["entries"][p2p_op_idx]["p2p_seq_id"], expected_seq)
                self.assertEqual(t["entries"][p2p_op_idx]["op_id"], expected_op_id)
                expected_op_id += 1
                self.assertEqual(t["entries"][p2p_op_idx]["input_sizes"], [input_sizes])
                self.assertEqual(
                    t["entries"][p2p_op_idx]["output_sizes"], [input_sizes]
                )
                # duration doesn't get tagged onto individual ops yet, nor is their state updated
                self.assertEqual(t["entries"][p2p_op_idx]["state"], "scheduled")
                self.assertTrue("duration_ms" not in t["entries"][p2p_op_idx])

            # the coalesced op has no metadata but indicates that coalescing was used,
            # and accurately reflects the timing and state info for the whole group
            self.assertEqual(
                t["entries"][coalesced_op]["record_id"], expected_record_id
            )
            expected_record_id += 1
            self.assertEqual(
                t["entries"][coalesced_op]["profiling_name"], "hccl:coalesced"
            )
            self.assertEqual(t["entries"][coalesced_op]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][coalesced_op]["state"], "completed")
            self.assertEqual(t["entries"][coalesced_op]["input_sizes"], [])
            self.assertEqual(t["entries"][coalesced_op]["output_sizes"], [])
            if timing_enabled:
                duration = t["entries"][coalesced_op]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][coalesced_op])
            self.assertEqual(t["entries"][coalesced_op]["timeout_ms"], 600000)

    @skipIf(True, "send_recv is not supported")
    @parametrize(
        "op_sizes",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_individual_send_recv(self, op_sizes, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's npu events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        num_repeats = 10
        ops_per_repeat = len(op_sizes)
        for _ in range(num_repeats):
            for input_sizes in op_sizes:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    dist.recv(tensor, 1)
                elif self.rank == 1:
                    tensor *= 2
                    dist.send(tensor, 0)

        torch.npu.synchronize(device=self.local_device)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())
        self.assertEqual(len(t["entries"]), num_repeats * (ops_per_repeat))
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_repeats * ops_per_repeat):
            input_sizes = op_sizes[seq % ops_per_repeat]
            profiling_name = "hccl:recv 0<-1" if self.rank == 0 else "hccl:send 1->0"
            self.assertEqual(t["entries"][seq]["profiling_name"], profiling_name)
            # we don't increment collective_seq_id for p2p ops.
            self.assertEqual(t["entries"][seq]["collective_seq_id"], 0)
            self.assertEqual(t["entries"][seq]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][seq]["op_id"], expected_op_id)
            expected_op_id += 1
            self.assertEqual(t["entries"][seq]["input_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["output_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["state"], "completed")

            if timing_enabled:
                duration = t["entries"][seq]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][seq])

    @skipIf(True, "coalescing_manager is not supported")
    @parametrize("timing_enabled", [True, False])
    def test_coalescing_manager_collective(self, timing_enabled):
        """
        The coalescing manager api works by accumulating operations in python via a contextmanager, and then making
        one call into c++ to an <op>_coalesced API.  It has limited support for ops and has been added recently to
        avoid overheads of making individual py-cpp calls.  This complicates flight recording..

        For now, flight recording of coalescing_manager collectives is less detailed than cpp coalesced collectives.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_hccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]

        # TODO(whc) make this work with bigger world or something
        self.assertEqual(self.world_size, 2, self.world_size)

        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

        torch.npu.synchronize(device=self.rank)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace())

        self.assertEqual(
            len(t["entries"]), 1
        )  # one for the reduce_scatter_tensor_coalesced
        self.assertEqual(
            t["entries"][0]["profiling_name"], "hccl:reduce_scatter_tensor_coalesced"
        )
        # collective_seq_id should be incremented once.
        self.assertEqual(t["entries"][0]["collective_seq_id"], 1)
        self.assertEqual(t["entries"][0]["input_sizes"], [[2, 2], [2, 2]])
        self.assertEqual(
            t["entries"][0]["output_sizes"],
            [
                [
                    2,
                ],
                [
                    2,
                ],
            ],
        )
        self.assertEqual(t["entries"][0]["state"], "completed")
        if timing_enabled:
            duration = t["entries"][0]["duration_ms"]
            self.assertTrue(0.001 < duration < 10000, duration)
        else:
            self.assertTrue("duration_ms" not in t["entries"][0])


def check_if_test_is_skipped(fn):
    def wrapper(self, *args, **kwargs):
        for skip in TEST_SKIPS.values():
            if self.processes[0].exitcode == skip.exit_code:
                return MultiProcessTestCase._check_return_codes(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


class HCCLTraceTestDumpOnTimeoutBase(HCCLTraceTestBase):
    timeout_sec = 60

    def _create_process_group_hccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "hccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=HCCLTraceTestDumpOnTimeoutBase.timeout_sec),
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 0)

    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None


class HCCLTraceTestDumpOnTimeout(HCCLTraceTestDumpOnTimeoutBase):
    @parametrize("timing_enabled", [False])
    def test_timeout_dumps(self, timing_enabled):
        if self.rank != self.MAIN_PROCESS_RANK:
            # dump on heartbeatmonitor thread
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
        # need rank0 to crash before looking for its output file
        os.environ["TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"] = "60"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for rank0 to crash before looking for its output file
            # we rely on rank0 holding off its abort long enough to dump the debug info
            self.assertEqual(self._wait_process(0, timeout=180), -6)
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
                self.assertEqual(t[1]["collective_seq_id"], 2)
                self.assertEqual(
                    t[1]["state"], self.started_or_scheduled(timing_enabled)
                )

            self.assertFalse(os.path.exists(self._trace_name(rank=1)))

            return

        pg = self._create_process_group_hccl()
        if timing_enabled:
            # we force disabled timing in setup, since there is no 'disable' function
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will crash before it passes the sync, but rank1 will exit quickly and cleanly
            torch.npu.synchronize(device=device)


instantiate_parametrized_tests(HCCLTraceTestDumpOnTimeout)
instantiate_parametrized_tests(HCCLTraceTest)


class HCCLTraceTestTimeoutDumpOnStuckRanks(HCCLTraceTestDumpOnTimeoutBase):
    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 0)

    def test_timeout_dumps_on_stuck_ranks(self):
        if self.rank != self.MAIN_PROCESS_RANK:
            # now we need start heartbeatmonitor thread manually
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
        # need rank0 to crash quicker after detecting timeout
        os.environ["TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"] = "60"
        # restore this env var to its prior default in case another test changed it
        os.environ["TORCH_HCCL_COORD_CHECK_MILSEC"] = "1000"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for both ranks' output
            # file, and we rely on rank1 to sleep long enough to dump the debug info.
            self.assertEqual(self._wait_process(0, timeout=180), -6)
            self.assertEqual(self._wait_process(1, timeout=180), 0)
            self.assertTrue(os.path.exists(self._trace_name(rank=1)))
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
            with open(self._trace_name(rank=1), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 1)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
            return

        pg = self._create_process_group_hccl()
        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will get stuck, timeout and then signal a timeout to all ranks.
            torch.npu.synchronize(device=device)

            if self.rank == 1:
                # Force rank 1 to sleep 120s so that it will eventually exit as well after
                # getting the global signal to dump the debugging info(won't break).
                time.sleep(120)


class HcclErrorDumpTest(HCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort with exception and rank1 to exit with exit 1
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 1)

    def test_hccl_errors_dump(self):
        if self.rank != self.MAIN_PROCESS_RANK:
            # now we need start heartbeatmonitor thread manually
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
        os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_HCCL_DUMP_ON_TIMEOUT"] = "1"
        # need rank0 to dump before abort and we update it to 30 to avoid heratbeat dump
        os.environ["TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"] = "30"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for dump
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), 1)
            # verify that the trace file exists for rank0
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            return

        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "hccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=10),
        )
        process_group = c10d.distributed_c10d._get_default_group()
        process_group.allreduce(torch.rand(10).npu(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).npu(self.rank))
            # expect an error to be raised
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # Block the current stream on the HCCL stream
                work.wait()
                # Run some NPU operations
                a = torch.rand(10).npu(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            sys.exit(1)


class HcclHeartbeatDumpTest(HCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    def test_hccl_heartbeat_dump(self):
        if self.rank != self.MAIN_PROCESS_RANK:
            # dump on heartbeatmonitor thread
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
        os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_HCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_HCCL_DUMP_ON_TIMEOUT"] = "1"
        # need rank0 to dump
        os.environ["TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"] = "10"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to finish
            self.assertEqual(self._wait_process(0, timeout=90), 0)
            self.assertEqual(self._wait_process(1, timeout=90), 0)
            # verify that the trace file exists for rank0
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            with open(self._trace_name(rank=0) + "_py_traceback", "r") as f:
                self.assertTrue("time.sleep(30)" in str(f.readlines()))
            # verify that the trace file not exists for rank1
            self.assertFalse(os.path.exists(self._trace_name(rank=1)))
            return

        pg = self._create_process_group_hccl()
        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                # sleep for heartbeat dump
                time.sleep(30)
            
            pg.allreduce(a).wait()

            torch.npu.synchronize(device=device)


class HCCLTraceTestDumpOnHcclTimeout(HCCLTraceTestBase):
    def setUp(self):
        os.environ["HCCL_EXEC_TIMEOUT"] = "60"
        os.environ["HCCL_EVENT_TIMEOUT"] = "90"
        super().setUp()

    def tearDown(self):
        # unset env to avoid impact watchdog timeout
        os.unsetenv('HCCL_EXEC_TIMEOUT')
        os.unsetenv('HCCL_EVENT_TIMEOUT')
        super().tearDown()

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to hccl exec timeout and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, 10)
        self.assertEqual(self.processes[1].exitcode, 0)

    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    @parametrize("timing_enabled", [False])
    def test_hccl_timeout_dumps(self, timing_enabled):
        if self.rank != self.MAIN_PROCESS_RANK:
            # dump on heartbeatmonitor thread
            os.environ["TORCH_HCCL_ENABLE_MONITORING"] = "1"
        # need rank0 to crash before looking for its output file
        os.environ["TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC"] = "60"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for rank0 to crash before looking for its output file
            self.assertEqual(self._wait_process(0, timeout=180), 10)
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
                self.assertEqual(t[1]["collective_seq_id"], 2)
                self.assertEqual(
                    t[1]["state"], self.started_or_scheduled(timing_enabled)
                )

            self.assertFalse(os.path.exists(self._trace_name(rank=1)))

            return

        pg = self._create_process_group_hccl()
        if timing_enabled:
            # we force disabled timing in setup, since there is no 'disable' function
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.npu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will crash before it passes the sync, but rank1 will exit quickly and cleanly
            torch.npu.synchronize(device=device)


instantiate_parametrized_tests(HCCLTraceTestDumpOnHcclTimeout)


if __name__ == "__main__":
    if torch.npu.is_available() and torch.npu.device_count() >= 2:
        run_tests()
