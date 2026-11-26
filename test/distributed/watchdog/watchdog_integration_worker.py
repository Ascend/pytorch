"""Two-rank scenarios launched by test_watchdog_integration.py, never directly by CI."""

import json
import ctypes
import errno
import faulthandler
import os
from pathlib import Path
import pickle
import resource
import signal
import struct
import sys
import threading
import time
import traceback
from datetime import timedelta

print("WORKER_STAGE: importing torch", flush=True)
import torch
import torch.distributed as dist
print("WORKER_STAGE: importing torch_npu", flush=True)
import torch_npu


def wait_until(predicate, description, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {description}")
        time.sleep(0.05)


class ControlledStore(dist.Store):
    """Delegate real communication to FileStore; inject only the requested fault.

    Event.wait releases the GIL. Only watchdog's remote_error check blocks;
    the monitor can still read dump signals and export Python diagnostics.
    """

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.block_watchdog = False
        self.fail_dump_check = False
        self.release_watchdog = threading.Event()

    def set(self, key, value):
        return self.store.set(key, value)

    def get(self, key):
        return self.store.get(key)

    def add(self, key, value):
        return self.store.add(key, value)

    def compare_set(self, key, expected, desired):
        return self.store.compare_set(key, expected, desired)

    def wait(self, keys, timeout=None):
        if timeout is None:
            return self.store.wait(keys)
        return self.store.wait(keys, timeout)

    def check(self, keys):
        if self.block_watchdog and any(key.startswith("remote_error:") for key in keys):
            print("STORE_BLOCK_ENTERED", flush=True)
            # Bound even an unexpectedly broken monitor. The parent has a second deadline.
            if not self.release_watchdog.wait(60):
                raise RuntimeError("test watchdog block exceeded 60 seconds")
        if self.fail_dump_check and "exception_dump" in keys:
            raise RuntimeError("injected dump Store failure")
        return self.store.check(keys)

    # The C++ Store trampoline uses these names for the virtual methods.
    def getNumKeys(self):
        return self.store.num_keys()

    def deleteKey(self, key):
        return self.store.delete_key(key)

    def clone(self):
        return self


def run(scenario, root, rank):
    def stage(description):
        print(f"WORKER_STAGE rank={rank}: {description}", flush=True)

    stage("initializing NPU")
    torch.npu.set_device(rank)
    stage("creating FileStore")
    store = dist.FileStore(str(root / "store"), 2)
    store.set_timeout(timedelta(seconds=30))
    controlled = ControlledStore(store)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.group_id = "watchdog_test"
    options._timeout = timedelta(seconds=30)
    # Direct backend construction keeps the supplied Store keys unprefixed.
    stage("constructing ProcessGroupHCCL")
    pg = torch_npu._C._distributed_c10d.ProcessGroupHCCL(controlled, rank, 2, options)
    stage("ProcessGroupHCCL constructed")
    result_type = torch._C._distributed_c10d.WorkResult

    def logs():
        return (root / f"rank{rank}.log").read_text(errors="replace")

    def barrier(label):
        store.set(f"{label}/{rank}", "ready")
        store.wait([f"{label}/{other}" for other in range(2)])

    def snapshot():
        return pickle.loads(torch_npu._C._distributed_c10d._dump_hccl_trace(
            includeStackTraces=False))

    def wait_for_retirement():
        # SUCCESS is published before retire_id(); wait for that separate step.
        timed = (os.environ.get("TORCH_HCCL_ENABLE_TIMING") == "1"
                 or os.environ.get("TORCH_HCCL_DESYNC_DEBUG") == "1")

        def retired():
            entries = snapshot()["entries"]
            # retire_id releases the lock before calculating duration. Its
            # retired flag can become visible before duration is written back.
            return bool(entries) and all(
                entry["retired"] and (not timed or "duration_ms" in entry)
                for entry in entries)

        wait_until(retired, "flight recorder retirement and duration")

    def collective(value=1):
        tensor = torch.full((16,), value, dtype=torch.float32, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        future = work.get_future_result()
        wait_until(future.done, "watchdog result future")
        assert int(future.value()) == int(result_type.SUCCESS), future.value()
        work.wait()
        torch.npu.synchronize()
        assert torch.equal(tensor.cpu(), torch.full((16,), value * 2, dtype=torch.float32))
        return work

    def read_dump():
        filename = root / f"trace_{rank}"
        deadline = time.monotonic() + 30
        # A writer may have created the file without finishing its pickle yet.
        while time.monotonic() < deadline:
            try:
                trace = pickle.loads(filename.read_bytes())
                assert "version" in trace and "pg_status" in trace
                return trace
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                time.sleep(0.05)
        raise AssertionError(f"No complete flight-recorder dump at {filename}")

    def manual_dump():
        stage("waiting for dump FIFO")
        filename = root / f"pipe_{rank}.pipe"
        deadline = time.monotonic() + 30
        while True:
            try:
                fd = os.open(filename, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as error:
                # The FIFO may not exist yet, or its reader may not have opened it.
                if error.errno not in (errno.ENOENT, errno.ENXIO):
                    raise
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Timed out waiting for dump FIFO reader at {filename}") from error
                time.sleep(0.05)
        try:
            os.write(fd, b"1")
        finally:
            os.close(fd)
        stage("dump request written; waiting for file")
        return read_dump()

    if scenario == "blocking_wait":
        tensor = torch.ones(16, device=f"npu:{rank}")
        pg.allreduce([tensor]).wait()
        assert torch.equal(tensor.cpu(), torch.full((16,), 2.0))
        assert "Process group watchdog thread started!" not in logs()
    else:
        stage("running initial collective")
        collective()  # Establish communicators before injecting failures.
    barrier("ready")
    stage(f"running scenario {scenario}")

    shim = None
    marker_file = root / f"marker_{rank}"
    if "WATCHDOG_TEST_FAULT_MODE" in os.environ:
        wait_for_retirement()
        shim = ctypes.CDLL(os.environ["WATCHDOG_FAULT_INJECTION_LIBRARY"])
        assert hasattr(shim, "watchdog_test_set_writer_target"), "rebuild watchdog_fault_injection for the new watchdog tests"
        shim.watchdog_test_arm.argtypes = [ctypes.c_int, ctypes.c_char_p]
        shim.watchdog_test_arm.restype = ctypes.c_int
        shim.watchdog_test_marker.argtypes = [ctypes.c_char_p]
        shim.watchdog_test_set_writer_target.argtypes = [ctypes.c_char_p]
        for name in ("watchdog_test_marker", "watchdog_test_disarm",
                     "watchdog_test_release_stalls", "watchdog_test_fail_writer", "watchdog_test_set_writer_target"):
            getattr(shim, name).restype = None
        shim.watchdog_test_set_writer_target(str(root / f"trace_{rank}").encode())

    def arm(mode=None):
        assert shim is not None, "scenario requires the test preload library"
        mode = int(os.environ["WATCHDOG_TEST_FAULT_MODE"]) if mode is None else mode
        status = shim.watchdog_test_arm(mode, str(marker_file).encode())
        if status == 1:
            raise RuntimeError("Python GIL checker is not registered")
        if status != 0:
            raise RuntimeError(f"Watchdog fault injection setup failed with status {status}; see native stderr")

    def markers():
        return marker_file.read_text() if marker_file.exists() else ""

    def finish_without_shutdown():
        # These scenarios intentionally leave a broken/unmatched group behind.
        barrier("checked_before_exit")
        (root / f"result{rank}.json").write_text(json.dumps(
            {"scenario": scenario, "rank": rank, "passed": True}))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if scenario == "blocking_wait":
        pass
    elif scenario == "multiple_groups":
        subgroup_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        subgroup_options.group_id = "watchdog_subgroup"
        subgroup_options.global_ranks_in_group = [0, 1]
        subgroup_store = dist.PrefixStore("subgroup/", controlled)
        subgroup = torch_npu._C._distributed_c10d.ProcessGroupHCCL(
            subgroup_store, rank, 2, subgroup_options)
        tensor = torch.ones(16, device=f"npu:{rank}")
        work = subgroup.allreduce([tensor])
        future = work.get_future_result()
        wait_until(future.done, "subgroup result future")
        assert int(future.value()) == int(result_type.SUCCESS)
        work.wait()
        assert torch.equal(tensor.cpu(), torch.full((16,), 2.0))
        trace = manual_dump()
        groups = {entry["process_group"][0] for entry in trace["entries"]}
        assert {"group_name_watchdog_test", "group_name_watchdog_subgroup"} <= groups, groups
        assert logs().count("PGHCCL Watchdog environments:") == 1
        subgroup.shutdown()
        collective(2)
    elif scenario in {"timeout", "timeout_teardown", "timeout_mode_zero"}:
        if rank == 0:
            pg.set_timeout(timedelta(seconds=1))
            tensor = torch.ones(16, device=f"npu:{rank}")
            work = pg.allreduce([tensor])  # Rank 1 deliberately does not participate.
            future = work.get_future_result()
            wait_until(future.done, "timeout result future")
            assert int(future.value()) == int(result_type.TIMEOUT), future.value()
            if scenario != "timeout":
                print("TIMEOUT_RESULT_CONFIRMED", flush=True)
                store.set("timeout_observed", "yes")
                if scenario == "timeout_teardown":
                    time.sleep(30)
                    raise AssertionError("timeout override did not terminate rank 0")
                read_dump()
                assert pg.get_error() == torch._C._distributed_c10d.ErrorType.TIMEOUT
                finish_without_shutdown()
            observed = time.monotonic()
            wait_until(lambda: "terminated with exception" in logs(), "watchdog timeout exception")
            assert "Watchdog caught collective operation timeout" in logs()
            if os.environ.get("TORCH_HCCL_DUMP_ON_TIMEOUT", "0") == "0":
                assert time.monotonic() - observed < 5, "watchdog waited for a disabled fault dump"
                assert not store.check(["exception_dump"]), "disabled fault dump was broadcast"
                assert not store.check(["hccl_early_dump"]), "disabled early dump was broadcast"
                assert not (root / f"trace_{rank}").exists()
            store.set("timeout_observed", "yes")
        else:
            store.wait(["timeout_observed"])
            if scenario == "timeout_teardown":
                (root / f"result{rank}.json").write_text(json.dumps(
                    {"scenario": scenario, "rank": rank, "passed": True}))
                sys.stdout.flush()
                os._exit(0)
            if scenario == "timeout_mode_zero":
                read_dump()
                finish_without_shutdown()
        barrier("timeout_finished")
        (root / f"result{rank}.json").write_text(json.dumps(
            {"scenario": scenario, "rank": rank, "passed": True}))
        # A deliberately unmatched collective cannot be synchronized at shutdown.
        # Exit only after the TIMEOUT assertion and cross-rank acknowledgement.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif scenario == "watchdog_exception":
        store.set("remote_error:group_name_watchdog_test", b"bad")
        wait_until(lambda: "terminated with exception" in logs(), "watchdog exception catch")
        if os.environ["TORCH_HCCL_EXTRA_DUMP_ON_EXEC"] == "1":
            read_dump()
            assert "Sending extra dumping signal" in logs()
        else:
            # Join below waits for catch handling to finish before checking absence.
            pg.shutdown()
            assert not (root / f"trace_{rank}").exists()
    elif scenario == "desync":
        collective(2)
        for kind in ("start", "end"):
            key = f"HCCL_{rank}_trace_{kind}"
            wait_until(lambda: store.check([key]) and struct.unpack("Q", store.get(key)[:8])[0] == 2,
                       f"desync {kind} sequence")
            assert b"ALLREDUCE" in store.get(key)
        wait_for_retirement()
        assert any("duration_ms" in entry for entry in snapshot()["entries"]), \
            "desync must use timing-capable events"
    elif scenario in {"queue", "default_settings"}:
        pending = []
        for value in range(1, 9):
            tensor = torch.full((16,), value, dtype=torch.float32, device=f"npu:{rank}")
            work = pg.allreduce([tensor])
            pending.append((tensor, work, work.get_future_result(), value))
        for tensor, work, future, value in pending:
            # Exercise user completion polling concurrently with watchdog polling.
            wait_until(lambda: work.is_completed() and future.done(), "queued work completion")
            assert int(future.value()) == int(result_type.SUCCESS)
            work.wait()
            assert torch.equal(tensor.cpu(), torch.full((16,), value * 2, dtype=torch.float32))
        collective(9)  # The queue remains usable after retiring the first batch.
        if scenario == "default_settings":
            # Check the actual Linux thread list, not just the configuration log.
            names = []
            for task in Path("/proc/self/task").iterdir():
                try:
                    names.append((task / "comm").read_text().strip())
                except FileNotFoundError:
                    pass  # An unrelated runtime thread may have exited during enumeration.
            # Match the prefix so longer heartbeat thread names truncated by Linux comm are also caught.
            assert not any(name.startswith("hccl_heart") for name in names), names
            assert "Heartbeat monitor thread started!" not in logs()
            store.set("exception_dump", struct.pack("i", rank))
            time.sleep(3)
            assert not (root / f"pipe_{rank}.pipe").exists()
            assert not (root / f"trace_{rank}").exists()
            collective(10)
    elif scenario == "shutdown_pending":
        # Rank 1 withholds its collective until rank 0 starts shutdown. Rank 0
        # therefore has a genuinely queued, incomplete work at shutdown entry.
        if rank == 1:
            wait_until(lambda: "Starting to destroy process group" in
                       (root / "rank0.log").read_text(errors="replace"),
                       "rank 0 entering shutdown")
        tensor = torch.ones(1024, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        future = work.get_future_result()
        if rank == 0:
            assert not future.done(), "rank 1 has not joined the collective yet"
        pg.shutdown()
        assert future.done(), "shutdown left a result future pending"
        assert int(future.value()) == int(result_type.SUCCESS)
        assert torch.equal(tensor.cpu(), torch.full((1024,), 2.0))
        pg.shutdown()
    elif scenario == "paused_queue":
        # STOP also prevents work enqueue. Do not create work in this state.
        pg.set_watchdog_status(2)
        time.sleep(10)
        assert "watchdog got stuck" not in logs()
        pg.set_watchdog_status(1)
        collective(2)
    elif scenario == "timing":
        collective(2)
        wait_for_retirement()
        entries = snapshot()["entries"]
        assert entries
        enabled = os.environ["TORCH_HCCL_ENABLE_TIMING"] == "1"
        for entry in entries:
            assert entry["state"] == "completed", entry
            if enabled:
                assert "duration_ms" in entry and entry["duration_ms"] >= 0, entry
                assert entry["time_discovered_started_ns"] is not None, entry
            else:
                assert "duration_ms" not in entry, entry
    elif scenario == "buffer_limit":
        for value in range(2, 8):
            collective(value)
        wait_for_retirement()
        entries = snapshot()["entries"]
        assert len(entries) == 4, entries
        assert [entry["collective_seq_id"] for entry in entries] == [4, 5, 6, 7], entries
    elif scenario == "recorder_reset":
        # Delay rank 1, rather than STOP, so rank 0's old work is queued and
        # cannot finish before reset. No production enqueue path is bypassed.
        if rank == 1:
            store.wait(["rank0_reset_and_submitted"])
        tensor = torch.ones(1024, device=f"npu:{rank}")
        old_work = pg.allreduce([tensor])
        old_future = old_work.get_future_result()
        if rank == 0:
            assert not old_future.done(), "rank 1 has not joined the old work yet"
        torch_npu._C._distributed_c10d._reset_fr_recording_hccl()
        assert not snapshot()["entries"], "reset did not clear old entries"
        # Create a new-epoch entry while old-epoch work is still queued.
        newer = torch.ones(16, device=f"npu:{rank}")
        new_work = pg.allreduce([newer])
        new_future = new_work.get_future_result()
        if rank == 0:
            store.set("rank0_reset_and_submitted", "ready")
        torch.npu.synchronize()
        wait_until(lambda: old_future.done() and new_future.done(), "both reset epochs' futures")
        for future in (old_future, new_future):
            assert int(future.value()) == int(result_type.SUCCESS)
        assert torch.equal(tensor.cpu(), torch.full((1024,), 2.0))
        assert torch.equal(newer.cpu(), torch.full((16,), 2.0))
        wait_for_retirement()
        entries = snapshot()["entries"]
        assert len(entries) == 1 and entries[0]["collective_seq_id"] == 3, entries
        assert entries[0]["state"] == "completed", entries
        collective(2)
    elif scenario == "idle":
        time.sleep(10)  # Longer than two heartbeat intervals; an empty queue is healthy.
        assert "watchdog got stuck" not in logs()
        collective(2)
    elif scenario in {"manual_dump", "only_active"}:
        trace = manual_dump()
        if scenario == "only_active":
            assert not trace.get("entries"), trace.get("entries")
        else:
            assert trace.get("entries"), "dump lost completed collective records"
        barrier("dump_read")
        if scenario == "manual_dump":
            (root / f"trace_{rank}").unlink()
            manual_dump()
        collective(2)
    elif scenario in {"stack_dump", "shutdown_grace"}:
        # Unlike manual dump, the fault-dump path reads TORCH_INCLUDE_STACK_TRACE.
        if rank == 0:
            store.set("exception_dump", struct.pack("i", 0))
        trace = read_dump()
        assert trace["entries"]
        if scenario == "stack_dump":
            included = os.environ["TORCH_INCLUDE_STACK_TRACE"] == "1"
            assert any(entry.get("frames") for entry in trace["entries"]) == included
            assert "Python traceback dump attempt finished" in logs()
        else:
            wait_until(lambda: "Finished flight recorder successfully" in logs(), "completed fault dump")
            barrier("grace_started")
            started = time.monotonic()
            pg.shutdown()
            assert time.monotonic() - started < 10, "shutdown waited for the entire 30-second grace"
    elif scenario in {"early_dump", "remote_dump", "malformed_dump"}:
        if rank == 0:
            key = "hccl_early_dump" if scenario == "early_dump" else "exception_dump"
            payload = b"bad" if scenario == "malformed_dump" else struct.pack("i", 1)
            store.set(key, payload)
        read_dump()
        if scenario == "early_dump":
            collective(2)
    elif scenario == "trace_disabled":
        time.sleep(3)
        assert not (root / f"pipe_{rank}.pipe").exists()
        assert not (root / f"trace_{rank}").exists()
        collective(2)
    elif scenario in {"remote_error", "remote_error_disabled"}:
        store.set("remote_error:group_name_watchdog_test", struct.pack("i", 1))
        if scenario == "remote_error":
            wait_until(lambda: "remote error detected from rank: 1" in logs(), "remote error log")
        else:
            time.sleep(3)
            assert "remote error detected" not in logs()
        collective(2)
    elif scenario == "store_failure":
        controlled.fail_dump_check = True
        wait_until(lambda: "injected dump Store failure" in logs(), "Store failure warning")
        controlled.fail_dump_check = False
        manual_dump()  # The same monitor must still handle requests after the exception.
        collective(2)
    elif scenario == "heartbeat_stall":
        controlled.block_watchdog = True
        wait_until(lambda: "STORE_BLOCK_ENTERED" in logs(), "blocked watchdog")
        if os.environ["TORCH_HCCL_ENABLE_MONITORING"] == "1":
            # Only the real monitor may terminate this process. A normal exit fails the test.
            time.sleep(45)
            raise AssertionError("heartbeat monitor did not terminate the process")
        # Monitoring=0 means no thread, rather than a thread that only skips abort.
        store.set("exception_dump", struct.pack("i", rank))
        time.sleep(8)  # Longer than two configured heartbeat intervals.
        assert "Heartbeat monitor thread started!" not in logs()
        assert "watchdog got stuck" not in logs()
        assert not (root / f"pipe_{rank}.pipe").exists()
        assert not (root / f"trace_{rank}").exists()
        controlled.block_watchdog = False
        controlled.release_watchdog.set()
        collective(2)
    elif scenario in {"work_error", "work_error_monitor"}:
        arm()
        tensor = torch.ones(16, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        future = work.get_future_result()
        wait_until(future.done, "injected work error result")
        assert int(future.value()) == int(result_type.COMM_ERROR), future.value()
        wait_until(lambda: pg.get_error() == torch._C._distributed_c10d.ErrorType.COMM_ERROR,
                   "PG communication error state")
        if scenario == "work_error_monitor":
            time.sleep(30)
            raise AssertionError("enabled monitor did not terminate mode 0 after its fault signal")
        mode = int(os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"])
        if mode in (1, 3) and os.environ.get("TORCH_HCCL_RETHROW_HCCL_ERRORS", "1") == "1":
            time.sleep(30)
            raise AssertionError("work error did not terminate the rank")
        if mode in (0, 2):
            wait_for_retirement()
        else:
            wait_until(lambda: "terminated with exception" in logs(), "suppressed watchdog rethrow")
        assert "work_async_error" in markers(), "fault hit the communicator precheck instead of work"
        assert ("comm_destroy" in markers()) == (mode in (1, 2)), markers()
        shim.watchdog_test_marker(b"strategy_checked")
        shim.watchdog_test_disarm()
    elif scenario == "callback_query":
        arm()  # Keep completed device work pending only in watchdog event queries.
        tensor = torch.ones(16, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        future = work.get_future_result()
        wait_until(lambda: "event_query_held" in markers(), "held watchdog event query")
        assert not future.done(), "fault injection did not hold the result future"
        callback_done = threading.Event()
        callback_errors = []

        def callback(result):
            try:
                assert int(result.value()) == int(result_type.SUCCESS)
                assert work.is_completed()
                return int(result.value())
            except BaseException as error:
                callback_errors.append(repr(error))
                raise
            finally:
                callback_done.set()

        chained = future.then(callback)
        barrier("callbacks_registered")
        shim.watchdog_test_disarm()
        assert callback_done.wait(10), "future callback deadlocked while re-entering the group"
        assert not callback_errors, callback_errors
        wait_until(chained.done, "callback result")
        assert int(chained.value()) == int(result_type.SUCCESS)
    elif scenario == "poll_shutdown":
        pending = []
        errors = []
        stop_polling = threading.Event()
        for _ in range(16):
            tensor = torch.ones(1024, device=f"npu:{rank}")
            work = pg.allreduce([tensor])
            pending.append((tensor, work, work.get_future_result()))

        def poll_work():
            try:
                while not stop_polling.is_set():
                    for _, work, _ in pending:
                        work.is_completed()
                    time.sleep(0.001)
            except Exception as error:
                errors.append(repr(error))

        poller = threading.Thread(target=poll_work, daemon=True)
        poller.start()
        try:
            pg.shutdown()
        finally:
            stop_polling.set()
            poller.join(timeout=5)
        assert not poller.is_alive(), "user completion polling was left blocked"
        assert not errors, errors
        for tensor, _, future in pending:
            assert future.done() and int(future.value()) == int(result_type.SUCCESS)
            assert torch.equal(tensor.cpu(), torch.full((1024,), 2.0))
    elif scenario == "shutdown_failure":
        if rank == 1:
            wait_until(lambda: "Starting to destroy process group" in
                       (root / "rank0.log").read_text(errors="replace"), "rank 0 shutdown")
        tensor = torch.ones(1024, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        future = work.get_future_result()
        if rank == 0:
            assert not future.done()
        arm()
        try:
            pg.shutdown()
        except RuntimeError as error:
            assert "injected shutdown stream synchronization failure" in str(error), str(error)
        else:
            raise AssertionError("shutdown swallowed the injected stream synchronization error")
        assert "shutdown_sync_error" in markers(), markers()
        # The injected exception follows a successful device sync. The watchdog
        # can still report the completed collective, and shutdown can be retried.
        wait_until(future.done, "watchdog completion after failed shutdown")
        assert int(future.value()) == int(result_type.SUCCESS), future.value()
        pg.shutdown()
    elif scenario == "watchdog_driver_exit":
        arm()
        tensor = torch.ones(16, device=f"npu:{rank}")
        pg.allreduce([tensor])
        wait_until(lambda: "main process destroyed the NPU runtime before watchdog loop exited" in logs(),
                   "watchdog driver-exit handling", timeout=80)
        assert "event_query_error" in markers(), markers()
        assert "taking the entire process down" not in logs()
        shim.watchdog_test_disarm()
    elif scenario in {"query_error", "query_recovery"}:
        pg.set_watchdog_status(2)
        tensor = torch.ones(16, device=f"npu:{rank}")
        work = pg.allreduce([tensor])
        torch.npu.synchronize()
        arm()
        started = time.monotonic()
        if scenario == "query_error":
            message = ("driver shutting down" if os.environ["WATCHDOG_TEST_FAULT_MODE"] == "12"
                       else "injected event query failure")
            try:
                work.is_completed()
            except RuntimeError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError("end-event query swallowed the injected error")
            assert time.monotonic() - started >= 59, "query skipped the device-log grace"
        else:
            assert not work.is_completed(), "recovery errors must not report completed work"
            assert time.monotonic() - started < 10, "recovery error incorrectly used the generic grace"
        assert "event_query_error" in markers(), markers()
        shim.watchdog_test_disarm()
        if scenario == "query_recovery":
            message = "FORCE STOP" if os.environ["WATCHDOG_TEST_FAULT_MODE"] == "13" else "UCE ERROR"
            try:
                pg.allreduce([torch.ones(16, device=f"npu:{rank}")])
            except RuntimeError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError("next submission did not report the HCCL recovery signal")
        pg.set_watchdog_status(1)
        collective(2)
    elif scenario in {"dump_stack_retry", "dump_writer_stall", "dump_writer_error", "gil_block", "cpp_stack"}:
        collective(2)  # Two entries distinguish a full dump from a single-work traceback.
        arm()
        if scenario == "dump_writer_error":
            shim.watchdog_test_fail_writer()
        if rank == 0:
            store.set("exception_dump", struct.pack("i", 0))
        if scenario == "dump_stack_retry":
            trace = read_dump()
            assert trace["entries"] and all(not entry.get("frames") for entry in trace["entries"])
            assert "stack_stalled" in markers() and "dump_without_stack" in markers(), markers()
            assert "timed out after 500 ms" in logs()
            events = {name: float(timestamp) for timestamp, name in
                      (line.split() for line in markers().splitlines())}
            assert events["dump_without_stack"] - events["stack_stalled"] >= 0.4, events
            shim.watchdog_test_release_stalls()
        elif scenario == "dump_writer_stall":
            wait_until(lambda: logs().count("timed out after 500 ms") >= 2, "both writer wait timeouts")
            assert "writer_stalled" in markers(), markers()
            assert not (root / f"trace_{rank}").exists()
            shim.watchdog_test_release_stalls()
        elif scenario == "dump_writer_error":
            wait_until(lambda: logs().count("injected FR writer failure") >= 2, "both writer exceptions")
            assert "writer_threw" in markers(), markers()
            assert not (root / f"trace_{rank}").exists()
            shim.watchdog_test_disarm()
        elif scenario == "gil_block":
            wait_until(lambda: "gil_check_entered" in markers(), "registered native GIL checker")
            gil_holder = ctypes.PyDLL(os.environ["WATCHDOG_FAULT_INJECTION_LIBRARY"])
            gil_holder.watchdog_test_hold_gil.argtypes = [ctypes.c_int]
            gil_holder.watchdog_test_hold_gil.restype = None
            gil_holder.watchdog_test_hold_gil(1000)
            wait_until(lambda: "Could not acquire GIL within 300 ms on exit" in logs(), "GIL stall diagnostic")
            wait_until(lambda: "gil_acquired" in markers(), "GIL checker recovery")
            events = {name: float(timestamp) for timestamp, name in
                      (line.split() for line in markers().splitlines())}
            assert events["gil_acquire_started"] - events["gil_check_entered"] < 0.25, events
            assert events["gil_acquired"] - events["gil_acquire_started"] >= 0.8, events
        else:
            enabled = os.environ["TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"] == "1"
            read_dump()
            if enabled:
                wait_until(lambda: "cpp_dump_called" in markers(), "registered C++ stack dumper")
                wait_until(lambda: "watchdog test C++ stack callback" in logs(), "C++ stack output")
            # A completed join makes the disabled-case absence check meaningful.
            pg.shutdown()
            assert ("cpp_dump_called" in markers()) == enabled, markers()
    elif scenario == "point_to_point":
        tensor = torch.full((16,), 7.0 if rank == 0 else 0.0, device=f"npu:{rank}")
        work = pg.send([tensor], 1, 0) if rank == 0 else pg.recv([tensor], 0, 0)
        future = work.get_future_result()
        wait_until(future.done, "P2P result future")
        assert int(future.value()) == int(result_type.SUCCESS)
        work.wait()
        assert torch.equal(tensor.cpu(), torch.full((16,), 7.0))
        collective(2)
    elif scenario in {"coalesced", "grouped"}:
        tensors = [torch.ones(8, device=f"npu:{rank}"),
                   torch.full((16,), 2.0, device=f"npu:{rank}")]
        if scenario == "coalesced":
            work = pg.allreduce_coalesced(tensors)
        else:
            pg._start_coalescing()
            for tensor in tensors:
                pg.allreduce([tensor])
            work = pg._end_coalescing()
        future = work.get_future_result()
        wait_until(future.done, "coalesced result future")
        assert int(future.value()) == int(result_type.SUCCESS)
        work.wait()
        for value, tensor in enumerate(tensors, 1):
            assert torch.equal(tensor.cpu(), torch.full(tensor.shape, value * 2.0))
        collective(2)
    elif scenario == "graph_capture":
        wait_for_retirement()
        before = snapshot()["pg_status"]
        tensor = torch.ones(16, device=f"npu:{rank}")
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_work = pg.allreduce([tensor])
            # Join the HCCL stream back to the capture stream before capture ends.
            captured_work.wait()
        for _ in range(2):
            tensor.fill_(1)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(tensor.cpu(), torch.full((16,), 2.0))
        assert snapshot()["pg_status"] == before, "captured work changed watchdog queue status"
        # Captured work is not tracked; do not wait for its watchdog result future.
        assert not captured_work.get_future_result().done()
        collective(2)
    else:
        raise AssertionError(f"Unknown scenario: {scenario}")

    barrier("finished")
    stage("shutting down")
    pg.shutdown()
    (root / f"result{rank}.json").write_text(json.dumps(
        {"scenario": scenario, "rank": rank, "passed": True}))


if __name__ == "__main__":
    timeout_sec = int(os.environ.get("WATCHDOG_TEST_TIMEOUT_SEC", "300"))
    if timeout_sec <= 0:
        raise ValueError("WATCHDOG_TEST_TIMEOUT_SEC must be positive")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(80, timeout_sec))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Let the parent report its timeout first; keep a fallback if parent cleanup fails.
    signal.alarm(timeout_sec + 10)
    try:
        run(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Avoid a second hang in destructors after an assertion failed.
