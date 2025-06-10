import os
import json
import pickle
import tempfile
import time
from datetime import datetime, timedelta
from unittest import mock

import torch
import torch.distributed as c10d
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
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
        return torch.device("npu", self.rank_to_GPU[self.rank][0])

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
    def rank_to_GPU(self):
        # return rank to GPU map
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
            self.assertEqual(last["timeout_ms"], 3600000)
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            if timing_enabled:
                # very loose bounds, measured 0.036 ms on devgpu
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


instantiate_parametrized_tests(HCCLTraceTest)


if __name__ == "__main__":
    if torch.npu.is_available() and torch.npu.device_count() >= 2:
        run_tests()
