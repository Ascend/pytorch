"""Two-NPU fault injection through a separately compiled LD_PRELOAD library.

Run with FR_FAULT_INJECTION_LIBRARY=/absolute/path/fr_fault_injection.so and
ASCEND_RT_VISIBLE_DEVICES set to two healthy NPUs. Child logs are retained in
FR_FAULT_LOG_DIR (a fresh temporary directory by default).
"""

import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import unittest


def worker(case, rank, directory):
    import torch
    import torch.distributed as dist
    import torch_npu

    directory = Path(directory)
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if line.endswith("/libtorch_npu.so")})
    (directory / f"library_{rank}.json").write_text(json.dumps(libraries))
    if os.environ.get("FR_EXPECT_LIBRARY"):
        assert libraries == [os.environ["FR_EXPECT_LIBRARY"]], libraries
    torch.npu.set_device(rank)
    store = dist.FileStore(str(directory / "store"), 2)
    if os.environ.get("FR_TEST_FAIL_DUMP_STORE") == "1":
        class DumpFailingStore(dist.Store):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def set(self, key, value):
                if key == "exception_dump":
                    raise RuntimeError("injected extra dump Store failure")
                return self.wrapped.set(key, value)

            def get(self, key):
                return self.wrapped.get(key)

            def add(self, key, value):
                return self.wrapped.add(key, value)

            def wait(self, keys, timeout=None):
                if timeout is None:
                    return self.wrapped.wait(keys)
                return self.wrapped.wait(keys, timeout)

            def check(self, keys):
                return self.wrapped.check(keys)

        store = DumpFailingStore(store)
    dist.init_process_group(
        "hccl", rank=rank, world_size=2, store=store,
        timeout=timedelta(seconds=3 if case.startswith("dump_") else 60),
    )
    tensor = torch.ones(128, device=f"npu:{rank}")
    dist.all_reduce(tensor)
    torch.npu.synchronize()
    # Retire warmup before arming fault injection.
    time.sleep(1)
    shim = ctypes.CDLL(os.environ["FR_FAULT_INJECTION_LIBRARY"])
    shim.fr_test_arm.argtypes = [ctypes.c_int, ctypes.c_char_p]
    modes = {"dump_stack": 1, "dump_stack_short_wait": 1, "dump_writer": 2,
             "sync_api": 3, "async_comm": 4, "async_query": 5,
             "dump_extra_race": 6, "monitor_shutdown": 7}
    if os.environ.get("FR_TEST_FAIL_DUMP_WRITER") == "1":
        shim.fr_test_fail_writer()
    if rank == 0 or not case.startswith("dump_"):
        mode = 0 if rank == 1 and os.environ.get("FR_TEST_INJECT_RANK_ZERO_ONLY") == "1" else modes[case]
        shim.fr_test_arm(mode, str(directory / f"marker_{rank}").encode())
    if case == "monitor_shutdown":
        deadline = time.monotonic() + 15
        marker = directory / f"marker_{rank}"
        while "monitor_exit_diagnostics" not in marker.read_text():
            if time.monotonic() > deadline:
                raise AssertionError("monitor did not reach exit diagnostics")
            time.sleep(0.01)
        store.set(f"stop_ready_{rank}", "1")
        store.wait(["stop_ready_0", "stop_ready_1"])
        started = time.monotonic()
        dist.destroy_process_group()
        (directory / f"shutdown_{rank}.json").write_text(json.dumps({
            "elapsed": time.monotonic() - started,
        }))
        os._exit(0)
    if case.startswith("dump_"):
        if rank == 0:
            dist.all_reduce(tensor)
            torch.npu.synchronize()
        time.sleep(90)
        raise AssertionError("monitor did not terminate the injected rank")
    if case == "sync_api":
        try:
            dist.all_reduce(tensor)
            torch.npu.synchronize()
        except RuntimeError as error:
            (directory / f"exception_{rank}").write_text(str(error))
            data = torch_npu._C._distributed_c10d._dump_hccl_trace()
            (directory / f"manual_{rank}").write_bytes(data)
            # A synchronous submission error is not a watchdog dump promise.
            os._exit(0)
        raise AssertionError("HcclAllReduce error was not propagated")
    # The cached communicator is checked even without a pending Work. Keeping
    # the completed warmup isolates the precheck from submission/timeout errors.
    time.sleep(60)
    raise AssertionError("watchdog did not terminate the injected rank")


class FlightRecorderFaultInjectionTest(unittest.TestCase):
    def run_case(self, case, wait_ms=15000, buffer_size=2000, *,
                 extra_dump=None, monitoring=True, dump_on_timeout=True,
                 fail_store=False, fail_writer=False, only_active=False,
                 nccl_extra_dump=False, status_save=False, heartbeat_seconds=120,
                 require_all_dumps=True, peer_buffer_size=None,
                 peer_monitoring=None, peer_async_error_handling=None,
                 inject_rank_zero_only=False):
        library = os.environ.get("FR_FAULT_INJECTION_LIBRARY")
        if not library:
            self.skipTest("set FR_FAULT_INJECTION_LIBRARY to the test preload .so")
        directory = Path(tempfile.mkdtemp(
            prefix=f"{case}_", dir=os.environ.get("FR_FAULT_LOG_DIR")))
        print(f"Fault injection evidence: {directory}", flush=True)
        env = os.environ.copy()
        env.update(LD_PRELOAD=library, TORCH_FR_BUFFER_SIZE=str(buffer_size),
                   TORCH_CPP_LOG_LEVEL="INFO",
                   TORCH_FR_CPP_STACK="0", TORCH_HCCL_ENABLE_TIMING="0",
                   TORCH_HCCL_ENABLE_MONITORING=str(int(monitoring)),
                   TORCH_HCCL_ASYNC_ERROR_HANDLING="1",
                   TORCH_HCCL_DUMP_ON_TIMEOUT=str(int(dump_on_timeout)),
                   TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC=str(wait_ms),
                   TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=str(heartbeat_seconds),
                   TORCH_HCCL_COORD_CHECK_MILSEC="100",
                   TORCH_FR_DUMP_TEMP_FILE=str(directory / "trace_"))
        env.pop("TORCH_FR_DUMP_DYNAMIC_FILE_NAME", None)
        env.pop("TORCH_HCCL_EXTRA_DUMP_ON_EXEC", None)
        env.pop("TORCH_NCCL_EXTRA_DUMP_ON_EXEC", None)
        if extra_dump is not None:
            env["TORCH_HCCL_EXTRA_DUMP_ON_EXEC"] = str(int(extra_dump))
        if nccl_extra_dump:
            env["TORCH_NCCL_EXTRA_DUMP_ON_EXEC"] = "1"
        env["FR_TEST_FAIL_DUMP_STORE"] = str(int(fail_store))
        env["FR_TEST_FAIL_DUMP_WRITER"] = str(int(fail_writer))
        env["FR_TEST_INJECT_RANK_ZERO_ONLY"] = str(int(inject_rank_zero_only))
        env["TORCH_INCLUDE_ONLY_ACTIVE"] = str(int(only_active))
        env["TORCH_HCCL_STATUS_SAVE_ENABLE"] = str(int(status_save))
        env["TORCH_HCCL_STATUS_SAVE_PATH"] = str(directory / "status")
        env["TORCH_INCLUDE_STACK_TRACE"] = "1"
        if case == "sync_api":
            # Make the injected submission failure synchronous with Python so
            # it tests the API exception, rather than a task-queue worker abort.
            env["TASK_QUEUE_ENABLE"] = "0"
        processes = []
        try:
            for rank in range(2):
                worker_env = env.copy()
                if rank == 1:
                    if peer_buffer_size is not None:
                        worker_env["TORCH_FR_BUFFER_SIZE"] = str(peer_buffer_size)
                    if peer_monitoring is not None:
                        worker_env["TORCH_HCCL_ENABLE_MONITORING"] = str(int(peer_monitoring))
                    if peer_async_error_handling is not None:
                        worker_env["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = str(peer_async_error_handling)
                with (directory / f"rank_{rank}.log").open("w") as output:
                    processes.append(subprocess.Popen(
                        [sys.executable, __file__, "--worker", case, str(rank), str(directory)],
                        env=worker_env, stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    ))
            deadline = time.monotonic() + 85
            codes = []
            for rank, process in enumerate(processes):
                codes.append(process.wait(timeout=max(1, deadline - time.monotonic())))
                log = (directory / f"rank_{rank}.log").read_text()
                self.assertNotIn("FR test cannot resolve", log, log[-8000:])
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        logs = [(directory / f"rank_{rank}.log").read_text() for rank in range(2)]
        expected = 0 if case in ("sync_api", "monitor_shutdown") else -signal.SIGABRT
        self.assertEqual(codes[0], expected, logs[0][-8000:])
        if not case.startswith("dump_"):
            self.assertEqual(codes[1], expected, logs[1][-8000:])
        markers = [line.split() for line in (directory / "marker_0").read_text().splitlines()]
        events = {name: float(timestamp) for timestamp, name in markers}
        if case == "monitor_shutdown":
            for rank in range(2):
                elapsed = json.loads((directory / f"shutdown_{rank}.json").read_text())["elapsed"]
                self.assertLess(elapsed, 1.0, f"rank {rank}: shutdown waited {elapsed}s")
            return
        if case == "dump_extra_race":
            self.assertIn("monitor_after_dump", events)
            self.assertNotIn("Sending extra dumping signal", logs[0])
            self.assertTrue((directory / "trace_0").is_file())
            self.assertEqual(sum(line[1] == "dump_serialized" for line in markers), 1)
            return
        if case.startswith("async_"):
            if extra_dump:
                self.assertTrue(any("Sending extra dumping signal" in log for log in logs),
                                "No rank attempted the extra dump request")
            for rank, log in enumerate(logs):
                passive_peer = inject_rank_zero_only and rank == 1
                local_error = "checkHcclComms found" in log
                if require_all_dumps and not passive_peer:
                    self.assertTrue(local_error, log[-8000:])
                else:
                    self.assertTrue(local_error or "Received a global dump signal from rank" in log,
                                    log[-8000:])
                if local_error:
                    self.assertIn("HCCL watchdog thread terminated with exception", log)
                    marker = "async_comm_error" if case == "async_comm" else "async_query_api_error"
                    self.assertIn(marker, (directory / f"marker_{rank}").read_text())
                self.assertNotIn("Watchdog caught collective operation timeout", log)
                trace = directory / f"trace_{rank}"
                rank_buffer = peer_buffer_size if rank == 1 and peer_buffer_size is not None else buffer_size
                should_write = bool(extra_dump) and rank_buffer > 0
                if should_write and not (fail_writer and rank == 0):
                    # EXTRA coordinates requests, not completed files. Under
                    # competing monitor/catch exits, only validate files that
                    # were produced; retain deterministic file checks elsewhere.
                    if require_all_dumps or trace.exists():
                        data = pickle.loads(trace.read_bytes())
                        self.assertEqual(data["version"], "2.10")
                        entries = data.get("entries", [])
                        self.assertEqual(bool(entries), not only_active)
                        if not (monitoring or (passive_peer and peer_monitoring)):
                            self.assertTrue(all(not entry.get("frames") for entry in entries))
                    if require_all_dumps and not passive_peer:
                        self.assertIn("Sending extra dumping signal", log)
                else:
                    self.assertFalse(trace.exists(), log[-8000:])
                if extra_dump and rank_buffer == 0 and not passive_peer:
                    self.assertIn("Sending extra dumping signal", log)
                if fail_store and should_write:
                    self.assertIn("Failed to broadcast extra dump signal", log)
                if fail_writer and rank == 0:
                    self.assertIn("Extra Flight Recorder dump failed: injected FR writer failure", log)
                if status_save:
                    files = list((directory / "status").glob(f"torch_hccl_status-{rank}_*.log"))
                    self.assertTrue(files, log[-8000:])
                    status = json.loads(files[-1].read_text())
                    self.assertTrue(status["last_comm_op"])
                    self.assertGreater(status["global_pg_end_time"], 0)
                    self.assertNotIn("injected FR writer failure", status["exception_message"])
            return
        if case.startswith("dump_"):
            self.assertIn("dump_without_stack", events)
            delta = events["dump_without_stack"] - events["dump_with_stack"]
            threshold = wait_ms / 1000
            self.assertGreaterEqual(delta, threshold * 0.8)
            self.assertLess(delta, threshold + 5)
            if case == "dump_writer":
                self.assertIn("writer_stalled", events)
                self.assertNotIn("write_completed", events)
                self.assertFalse((directory / "trace_0").exists())
                self.assertGreaterEqual(logs[0].count(f"timed out after {wait_ms} ms"), 2)
                return
            self.assertIn("stack_stalled", events)
            self.assertIn("write_completed", events)
            data = pickle.loads((directory / "trace_0").read_bytes())
            self.assertTrue(data["entries"])
            self.assertTrue(all(not entry.get("frames") for entry in data["entries"]))
        elif case == "sync_api":
            self.assertIn("sync_api_error", events)
            self.assertIn("error code is 4", (directory / "exception_0").read_text())
            data = pickle.loads((directory / "manual_0").read_bytes())
        self.assertEqual(data["version"], "2.10")

    def test_first_dump_stack_stall_retries_without_stack_after_15_seconds(self):
        self.run_case("dump_stack")

    def test_dump_wait_override_changes_retry_deadline(self):
        self.run_case("dump_stack_short_wait", wait_ms=500)

    def test_writer_stall_has_bounded_exit_without_promising_recovery(self):
        self.run_case("dump_writer")

    def test_synchronous_error_with_recording_disabled_is_dumpable(self):
        self.run_case("sync_api", buffer_size=0)

    def test_extra_dump_covers_both_precheck_errors_without_monitor(self):
        for case in ("async_comm", "async_query"):
            with self.subTest(case=case):
                self.run_case(case, extra_dump=True, monitoring=False, dump_on_timeout=False)

    def test_extra_dump_preserves_disabled_behavior(self):
        for case in ("async_comm", "async_query"):
            for enabled in (None, False):
                with self.subTest(case=case, extra_dump=enabled):
                    self.run_case(case, extra_dump=enabled)

    def test_extra_dump_with_monitor(self):
        # Match the community best-effort contract. A pending remote request
        # may make the local catch skip; neither path acknowledges file creation.
        for case in ("async_comm", "async_query"):
            with self.subTest(case=case):
                self.run_case(case, extra_dump=True, require_all_dumps=False)

    def test_extra_dump_respects_disabled_recorder(self):
        self.run_case("async_comm", extra_dump=True, buffer_size=0, monitoring=False)

    def test_disabled_local_recorder_still_requests_peer_dump(self):
        # Keep the peer's watchdog out of error teardown so its monitor can
        # consume the source rank's signal without a competing local exception.
        self.run_case("async_comm", extra_dump=True, buffer_size=0, monitoring=False,
                      peer_buffer_size=2000, peer_monitoring=True,
                      peer_async_error_handling=2, inject_rank_zero_only=True)

    def test_nccl_extra_dump_does_not_enable_hccl(self):
        self.run_case("async_query", monitoring=False, nccl_extra_dump=True)

    def test_extra_dump_store_failure_still_writes_locally(self):
        self.run_case("async_comm", extra_dump=True, monitoring=False, fail_store=True)

    def test_extra_dump_writer_exception_preserves_watchdog_failure(self):
        self.run_case("async_query", extra_dump=True, monitoring=False, fail_writer=True)

    def test_extra_dump_only_active_allows_empty_snapshot(self):
        self.run_case("async_comm", extra_dump=True, monitoring=False, only_active=True)

    def test_extra_dump_failure_keeps_status_save(self):
        self.run_case("async_query", extra_dump=True, monitoring=False,
                      fail_writer=True, status_save=True)

    def test_existing_timeout_request_is_not_reissued_by_extra_dump(self):
        self.run_case("dump_extra_race", wait_ms=500, extra_dump=True)

    def test_monitor_exit_grace_is_interrupted_by_shutdown(self):
        self.run_case("monitor_shutdown", dump_on_timeout=False, heartbeat_seconds=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        unittest.main()
