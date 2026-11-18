"""Black-box watchdog tests. Each scenario gets fresh ranks and environment caches."""

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


class WatchdogIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux":
            raise unittest.SkipTest("HCCL integration tests require Linux")
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.device_count() < 2:
            raise unittest.SkipTest("requires two NPUs")

    def run_scenario(self, scenario, overrides=None, fatal=False, expected_logs=(),
                     fault_mode=None, fatal_logs=None, fatal_ranks=None):
        # OptionsManager caches environment values: never reuse a worker process.
        env = os.environ.copy()
        env.pop("WATCHDOG_TEST_FAULT_MODE", None)
        for key in list(env):
            if key.startswith(("TORCH_HCCL_", "TORCH_FR_")) or key in {
                "HCCL_ASYNC_ERROR_HANDLING", "HCCL_BLOCKING_WAIT",
                "HCCL_DESYNC_DEBUG", "TORCH_INCLUDE_STACK_TRACE",
                "TORCH_INCLUDE_ONLY_ACTIVE", "TORCH_DISTRIBUTED_DEBUG",
            }:
                env.pop(key)
        env.update({
            "TORCH_CPP_LOG_LEVEL": "INFO",
            "TORCH_DISTRIBUTED_DEBUG": "OFF",
            "TORCH_HCCL_ASYNC_ERROR_HANDLING": "3",
            "TORCH_HCCL_ENABLE_MONITORING": "0",
            "TORCH_HCCL_DUMP_ON_TIMEOUT": "0",
            "TORCH_HCCL_TRACE_BUFFER_SIZE": "128",
            "TORCH_HCCL_ENABLE_TIMING": "0",
            "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "10",
            "TORCH_HCCL_COORD_CHECK_MILSEC": "50",
            "TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC": "5000",
            "TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN": "0",
            "TORCH_INCLUDE_STACK_TRACE": "0",
            "HCCL_EXEC_TIMEOUT": "120",
        })
        # Only diagnostic scenarios opt in to the monitor and automatic fault dumps.
        if scenario in {
            "multiple_groups", "manual_dump", "only_active", "early_dump",
            "remote_dump", "malformed_dump", "stack_dump", "shutdown_grace",
            "store_failure", "heartbeat_stall", "timeout_mode_zero", "work_error_monitor",
            "dump_stack_retry", "dump_writer_stall", "dump_writer_error", "gil_block", "cpp_stack",
            "trace_disabled",
        }:
            env.update({"TORCH_HCCL_ENABLE_MONITORING": "1", "TORCH_HCCL_DUMP_ON_TIMEOUT": "1"})
        # Fault tests restart ranks frequently; avoid reusing the fixed NPU port 16666.
        # Preserve an explicit port policy supplied by the caller.
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        env.update(overrides or {})
        for key, value in list(env.items()):
            if value is None:
                del env[key]

        # Include rank startup and diagnostic grace periods in the scenario budget.
        timeout_sec = int(env.get("WATCHDOG_TEST_TIMEOUT_SEC", "300"))
        self.assertGreater(timeout_sec, 0, "WATCHDOG_TEST_TIMEOUT_SEC must be positive")
        env["WATCHDOG_TEST_TIMEOUT_SEC"] = str(timeout_sec)

        if fault_mode is not None:
            library = os.environ.get("WATCHDOG_FAULT_INJECTION_LIBRARY")
            if library is None:
                library = Path(__file__).resolve().parents[2] / "build/gtest/libwatchdog_fault_injection.so"
                if not library.is_file():
                    self.skipTest("build watchdog_fault_injection or set WATCHDOG_FAULT_INJECTION_LIBRARY")
            library = Path(library).resolve()
            self.assertTrue(library.is_file(), f"Missing fault injection library: {library}")
            env["WATCHDOG_FAULT_INJECTION_LIBRARY"] = str(library)
            env["LD_PRELOAD"] = str(library) + (
                ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            env["WATCHDOG_TEST_FAULT_MODE"] = str(fault_mode)
            env["TASK_QUEUE_ENABLE"] = "0"

        worker = Path(__file__).parent / "watchdog" / "watchdog_integration_worker.py"
        with tempfile.TemporaryDirectory(prefix="hccl_watchdog_") as directory:
            root = Path(directory)
            env["TORCH_HCCL_DEBUG_INFO_TEMP_FILE"] = str(root / "trace_")
            env["TORCH_HCCL_DEBUG_INFO_PIPE_FILE"] = str(root / "pipe_")
            processes = []
            streams = []
            try:
                deadline = time.monotonic() + timeout_sec
                for rank in range(2):
                    stream = (root / f"rank{rank}.log").open("wb")
                    streams.append(stream)
                    processes.append(subprocess.Popen(
                        [sys.executable, "-u", str(worker), scenario, str(root), str(rank)],
                        env=env, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    ))
                while any(process.poll() is None for process in processes):
                    if time.monotonic() >= deadline:
                        logs = self.read_logs(root)
                        self.fail(f"{scenario} exceeded {timeout_sec} seconds\n"
                                  + self.format_logs(processes, logs))
                    time.sleep(0.1)
                logs = self.read_logs(root)
                details = self.format_logs(processes, logs)
                for rank, process in enumerate(processes):
                    rank_fatal = rank in fatal_ranks if fatal_ranks is not None else fatal
                    if rank_fatal:
                        self.assertEqual(process.returncode, -signal.SIGABRT, details)
                        # A Python exception/import failure is not a successful fatal test.
                        for message in (fatal_logs if fatal_logs is not None else (
                                "STORE_BLOCK_ENTERED", "watchdog got stuck", "Terminating the process")):
                            self.assertIn(message, logs[rank], details)
                        if scenario == "heartbeat_stall" and env.get("TORCH_HCCL_DUMP_ON_TIMEOUT", "0") == "0":
                            self.assertFalse((root / f"trace_{rank}").exists(), details)
                        if fault_mode is not None:
                            marker = (root / f"marker_{rank}").read_text()
                            self.assertIn("work_async_error", marker, details)
                            self.assertEqual("comm_destroy" in marker,
                                             env["TORCH_HCCL_ASYNC_ERROR_HANDLING"] == "1", details)
                    else:
                        # Report the exit code before secondary log assertions.
                        # Otherwise SIGALRM during startup looks like a missing INFO log.
                        self.assertEqual(process.returncode, 0, details)
                        result_file = root / f"result{rank}.json"
                        self.assertTrue(result_file.exists(), details)
                        self.assertEqual(json.loads(result_file.read_text()),
                                         {"scenario": scenario, "rank": rank, "passed": True}, details)
                    self.assertIn("PGHCCL Watchdog environments:", logs[rank], details)
                    for message in expected_logs:
                        self.assertIn(message, logs[rank], details)
            finally:
                failed = sys.exc_info()[0] is not None
                # Only kill these test-owned process groups, including their children.
                for process in processes:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
                for stream in streams:
                    stream.close()
                if failed:
                    saved = Path(tempfile.mkdtemp(prefix="hccl_watchdog_failed_"))
                    # FIFO files cannot be copied; preserve regular artifacts only.
                    for artifact in root.iterdir():
                        if artifact.is_file():
                            shutil.copy2(artifact, saved / artifact.name)
                    print(f"\n{scenario}: logs and dumps preserved at {saved}", file=sys.stderr)

    @staticmethod
    def read_logs(root):
        return [(root / f"rank{rank}.log").read_text(errors="replace") for rank in range(2)]

    @staticmethod
    def format_logs(processes, logs):
        sections = []
        for rank, (process, log) in enumerate(zip(processes, logs)):
            code = process.poll()
            reason = ""
            if code is not None and code < 0:
                try:
                    reason = f" ({signal.Signals(-code).name})"
                except ValueError:
                    pass
            sections.append(f"=== rank {rank}: exit={code}{reason} ===\n{log}")
        return "\n".join(sections)

    def test_completed_queue_and_result_futures(self):
        # Modes 1/2/3 should complete queued futures successfully and allow further collectives.
        for mode in (1, 2, 3):
            with self.subTest(mode=mode):
                self.run_scenario("queue", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": str(mode)})

    def test_multiple_groups_share_dump_but_not_work_queues(self):
        # A pipe request should make the main group's heartbeat monitor dump the shared FR,
        # including records from both groups. Closing the subgroup should leave the main group usable.
        self.run_scenario("multiple_groups")

    def test_blocking_wait_does_not_start_watchdog(self):
        # Blocking wait should produce correct results without logging a watchdog thread start.
        self.run_scenario("blocking_wait", {"TORCH_HCCL_BLOCKING_WAIT": "1"},
                          expected_logs=("NO watchdog thread is created",))

    def test_monitor_and_dump_are_disabled_by_default(self):
        # Unset both options: work futures still complete, without a monitor thread or dump files.
        self.run_scenario("default_settings", {
            "TORCH_HCCL_ENABLE_MONITORING": None,
            "TORCH_HCCL_DUMP_ON_TIMEOUT": None,
        }, expected_logs=("Whether to start the heartbeat monitor thread): 0, TORCH_HCCL_DUMP_ON_TIMEOUT: 0",))

    def test_monitor_can_handle_pipe_with_fault_dump_disabled(self):
        # Automatic fault dumping defaults to off; an enabled monitor still handles manual pipe requests.
        self.run_scenario("manual_dump", {
            "TORCH_HCCL_ENABLE_MONITORING": "1",
            "TORCH_HCCL_DUMP_ON_TIMEOUT": None,
        }, expected_logs=("Heartbeat monitor thread started!",))

    def test_disabled_fault_dump_path_does_not_broadcast_or_wait(self):
        # Either disabled option must prevent fault-dump broadcasting and the configured 15-second wait.
        for monitoring, dumping in ((None, None), ("1", None), ("0", "1")):
            with self.subTest(monitoring=monitoring, dumping=dumping):
                self.run_scenario("timeout", {
                    "TORCH_HCCL_ENABLE_MONITORING": monitoring,
                    "TORCH_HCCL_DUMP_ON_TIMEOUT": dumping,
                    "TORCH_HCCL_RETHROW_HCCL_ERRORS": "0",
                    "TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC": "15000",
                })

    def test_default_async_mode_completes_future(self):
        # Without an async mode override, collectives and result futures should complete successfully.
        self.run_scenario("queue", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": None})

    def test_legacy_async_environment_is_supported(self):
        # The legacy HCCL_ASYNC_ERROR_HANDLING variable should still allow successful completion.
        self.run_scenario("queue", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": None,
                                    "HCCL_ASYNC_ERROR_HANDLING": "3"})

    def test_new_async_environment_takes_precedence(self):
        # The new async variable should override the conflicting legacy value and select mode 3.
        self.run_scenario("queue", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "3",
                                    "HCCL_ASYNC_ERROR_HANDLING": "0"})

    def test_new_blocking_wait_environment_takes_precedence(self):
        # The new blocking-wait variable should override the legacy value and select nonblocking mode.
        self.run_scenario("queue", {"TORCH_HCCL_BLOCKING_WAIT": "0",
                                    "HCCL_BLOCKING_WAIT": "1"})

    def test_shutdown_completes_pending_result_futures_and_is_idempotent(self):
        # Shutdown should finish pending communication and its future; repeated shutdown should be harmless.
        self.run_scenario("shutdown_pending")

    def test_timeout_reports_timeout_future(self):
        # An unmatched collective should report TIMEOUT through its result future and log the timeout.
        self.run_scenario("timeout", {"TORCH_HCCL_DUMP_ON_TIMEOUT": "0",
                                      "TORCH_HCCL_RETHROW_HCCL_ERRORS": "0"})

    def test_watchdog_exception_extra_dump_is_opt_in(self):
        # Invalid remote-error data should trigger an extra dump only when EXTRA_DUMP_ON_EXEC is enabled.
        for enabled in (0, 1):
            with self.subTest(extra_dump=enabled):
                self.run_scenario("watchdog_exception", {
                    "TORCH_HCCL_PROPAGATE_ERROR": "1",
                    "TORCH_HCCL_RETHROW_HCCL_ERRORS": "0",
                    "TORCH_HCCL_EXTRA_DUMP_ON_EXEC": str(enabled),
                    "TORCH_HCCL_DUMP_ON_TIMEOUT": "0",
                }, expected_logs=("terminated with exception", "Invalid size for the timeout rank ID"))

    def test_desync_records_start_and_end(self):
        # Desync should advance Store start/end sequences and produce FR duration data.
        self.run_scenario("desync", {"TORCH_HCCL_DESYNC_DEBUG": "1"})

    def test_desync_enables_handling_when_mode_is_zero(self):
        # Desync with mode 0 should enable async handling and record communication progress.
        self.run_scenario("desync", {"TORCH_HCCL_DESYNC_DEBUG": "1",
                                     "TORCH_HCCL_ASYNC_ERROR_HANDLING": "0"},
                          expected_logs=("Enabling TORCH_HCCL_ASYNC_ERROR_HANDLING.",))

    def test_idle_watchdog_keeps_heartbeat(self):
        # An idle queue should not trigger a heartbeat stall, and later communication should succeed.
        self.run_scenario("idle", {"TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3",
                                   "TORCH_HCCL_ENABLE_MONITORING": "1"})

    def test_manual_dump_can_be_repeated_and_training_continues(self):
        # Two FIFO dump requests should both succeed without preventing later communication.
        self.run_scenario("manual_dump", expected_logs=("Flight Recorder trace successfully dumped.",))

    def test_remote_early_dump_does_not_stop_training(self):
        # An early-dump Store signal should produce a dump and allow further communication.
        self.run_scenario("early_dump")

    def test_remote_fault_dump(self):
        # A fault-dump Store signal should make each rank produce a readable FR dump.
        self.run_scenario("remote_dump", expected_logs=("Observed flight recorder dump signal",))

    def test_malformed_remote_dump_rank_still_dumps(self):
        # Invalid rank data in a fault-dump signal should log an error without preventing the dump.
        self.run_scenario("malformed_dump", expected_logs=("Failed to get timeout rank ID",))

    def test_only_active_dump_excludes_completed_work(self):
        # An active-only dump should exclude completed communication records.
        self.run_scenario("only_active", {"TORCH_INCLUDE_ONLY_ACTIVE": "1"})

    def test_stack_enabled_dump(self):
        # Enabling stack collection should include stack frames in fault-dump entries.
        self.run_scenario("stack_dump", {"TORCH_INCLUDE_STACK_TRACE": "1"})

    def test_fault_dump_can_exclude_entry_stacks(self):
        # Disabling stack collection should omit stack frames from fault-dump entries.
        self.run_scenario("stack_dump", {"TORCH_INCLUDE_STACK_TRACE": "0"})

    def test_timing_enabled_records_duration(self):
        # Timing-enabled records should contain a nonnegative duration and a discovered start time.
        self.run_scenario("timing", {"TORCH_HCCL_ENABLE_TIMING": "1"})

    def test_timing_disabled_omits_duration(self):
        # Timing-disabled records should complete without a duration_ms field.
        self.run_scenario("timing", {"TORCH_HCCL_ENABLE_TIMING": "0"})

    def test_shared_buffer_overrides_legacy_buffer_size(self):
        # The shared buffer setting should override the legacy size and retain only the latest four records.
        self.run_scenario("buffer_limit", {"TORCH_FR_BUFFER_SIZE": "4",
                                           "TORCH_HCCL_TRACE_BUFFER_SIZE": "128"})

    def test_shared_buffer_zero_disables_legacy_buffer(self):
        # A zero shared buffer size should disable FR files and FIFOs despite a nonzero legacy setting.
        self.run_scenario("trace_disabled", {"TORCH_FR_BUFFER_SIZE": "0"})

    def test_reset_during_pending_work_keeps_new_records(self):
        # Retiring old work after an FR reset must preserve new records; both futures should succeed.
        self.run_scenario("recorder_reset", {"TORCH_FR_BUFFER_SIZE": "4"})

    def test_stopped_work_polling_keeps_heartbeat_alive(self):
        # STOP should preserve the heartbeat, and communication should succeed after resuming RUN.
        self.run_scenario("paused_queue", {"TORCH_HCCL_ENABLE_MONITORING": "1",
                                           "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3"})

    def test_shutdown_interrupts_monitor_grace_wait(self):
        # Shutdown should interrupt the monitor grace period instead of waiting for the full interval.
        self.run_scenario("shutdown_grace", {"TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "30"})

    def test_trace_disabled_does_not_create_fifo(self):
        # A zero FR capacity should prevent dump files and FIFOs while collectives still succeed.
        self.run_scenario("trace_disabled", {"TORCH_HCCL_TRACE_BUFFER_SIZE": "0"})

    def test_remote_error_is_reported_while_queue_is_empty(self):
        # With propagation enabled, an empty queue should still detect and log a remote Store error.
        self.run_scenario("remote_error", {"TORCH_HCCL_PROPAGATE_ERROR": "1"},
                          expected_logs=("remote error detected from rank: 1",))

    def test_remote_error_disabled(self):
        # With propagation disabled, remote-error checks should stay inactive and communication should succeed.
        self.run_scenario("remote_error_disabled", {"TORCH_HCCL_PROPAGATE_ERROR": "0"})

    def test_store_check_failure_does_not_kill_monitor(self):
        # A Store check failure should not stop the monitor from handling later dumps and communication.
        self.run_scenario("store_failure", expected_logs=("store unavailable, not absent key",))

    def test_disabled_monitor_does_not_start_or_handle_dumps(self):
        # Even with fault dumping requested, disabled monitoring creates no thread and does not diagnose a stall.
        self.run_scenario("heartbeat_stall", {"TORCH_HCCL_PROPAGATE_ERROR": "1",
                                              "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3",
                                              "TORCH_HCCL_ENABLE_MONITORING": "0"})

    def test_heartbeat_stall_terminates_process(self):
        # With termination enabled, a watchdog stall should log the heartbeat failure and terminate each rank.
        self.run_scenario("heartbeat_stall", {"TORCH_HCCL_PROPAGATE_ERROR": "1",
                                              "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3",
                                              "TORCH_HCCL_ENABLE_MONITORING": "1"}, fatal=True)

    def test_heartbeat_stall_terminates_without_automatic_dump(self):
        # Disabling automatic dumps must not disable heartbeat stall detection or termination.
        self.run_scenario("heartbeat_stall", {"TORCH_HCCL_PROPAGATE_ERROR": "1",
                                              "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3",
                                              "TORCH_HCCL_ENABLE_MONITORING": "1",
                                              "TORCH_HCCL_DUMP_ON_TIMEOUT": None}, fatal=True)

    def test_no_handling_tracks_completed_work(self):
        # Mode 0 should still track successful work and complete its result future.
        self.run_scenario("queue", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0"})

    def test_no_handling_reports_work_error_without_cleanup(self):
        # An injected per-work error should notify COMM_ERROR without aborting communicators.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0",
                                         "TORCH_HCCL_DUMP_ON_TIMEOUT": "0"}, fault_mode=10)

    def test_cleanup_only_aborts_communicators_without_rethrow(self):
        # Mode 2 should destroy the failed work's communicator without terminating the rank.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "2",
                                         "TORCH_HCCL_DUMP_ON_TIMEOUT": "0"}, fault_mode=10)

    def test_teardown_cleans_up_then_terminates(self):
        # Mode 1 should destroy the communicator before the watchdog aborts the process.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "1"},
                          fault_mode=10, fatal=True,
                          fatal_logs=("failure detected by watchdog", "taking the entire process down"))

    def test_skip_cleanup_terminates_without_destroying_communicators(self):
        # Mode 3 should abort the process without first destroying the communicator.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "3"},
                          fault_mode=10, fatal=True,
                          fatal_logs=("failure detected by watchdog", "taking the entire process down"))

    def test_disabled_rethrow_keeps_rank_alive_after_work_error(self):
        # Suppressing the outer rethrow should preserve COMM_ERROR while allowing the rank to survive.
        for mode in (1, 3):
            with self.subTest(mode=mode):
                self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": str(mode),
                                                 "TORCH_HCCL_RETHROW_HCCL_ERRORS": "0",
                                                 "TORCH_HCCL_DUMP_ON_TIMEOUT": "0"}, fault_mode=10)

    def test_mode_zero_fault_can_still_trigger_monitor_termination(self):
        # Mode 0 skips explicit teardown, but an enabled monitor can still terminate on a fault signal.
        self.run_scenario("work_error_monitor", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0",
                                                 "TORCH_HCCL_ENABLE_MONITORING": "1",
                                                 "TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC": "3"},
                          fault_mode=10, fatal=True,
                          fatal_logs=("failure detected by watchdog", "Terminating the process"))

    def test_new_async_environment_precedence_controls_error_strategy(self):
        # New mode 3 must override legacy mode 0: terminate without communicator cleanup.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "3",
                                         "HCCL_ASYNC_ERROR_HANDLING": "0"},
                          fault_mode=10, fatal=True,
                          fatal_logs=("failure detected by watchdog", "taking the entire process down"))

    def test_new_mode_zero_overrides_legacy_teardown(self):
        # New mode 0 must override legacy mode 3 and keep the rank alive with monitoring disabled.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0",
                                         "HCCL_ASYNC_ERROR_HANDLING": "3",
                                         "TORCH_HCCL_DUMP_ON_TIMEOUT": "0"}, fault_mode=10)

    def test_desync_promotes_mode_zero_to_skip_cleanup(self):
        # Desync should promote mode 0 to mode 3, demonstrated by teardown without cleanup.
        self.run_scenario("work_error", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0",
                                         "TORCH_HCCL_DESYNC_DEBUG": "1"},
                          fault_mode=10, fatal=True,
                          fatal_logs=("Enabling TORCH_HCCL_ASYNC_ERROR_HANDLING.",
                                      "taking the entire process down"))

    def test_timeout_teardown_overrides_disabled_general_rethrow(self):
        # The timeout override should terminate rank 0 even when general rethrow is disabled.
        self.run_scenario("timeout_teardown", {"TORCH_HCCL_RETHROW_HCCL_ERRORS": "0",
                                               "TORCH_HCCL_TEARDOWN_ON_TIMEOUT": "1"},
                          fatal_ranks=(0,),
                          fatal_logs=("TIMEOUT_RESULT_CONFIRMED", "taking the entire process down"))

    def test_mode_zero_timeout_still_requests_fault_dump(self):
        # Mode 0 should report TIMEOUT and request a dump without explicit communicator cleanup.
        self.run_scenario("timeout_mode_zero", {"TORCH_HCCL_ASYNC_ERROR_HANDLING": "0"})

    def test_future_callback_can_read_work_completion(self):
        # A pending future's callback should observe SUCCESS and query its completed work.
        self.run_scenario("callback_query", fault_mode=40)

    def test_user_polling_races_with_shutdown(self):
        # Concurrent user completion checks and shutdown should not lose or duplicate future results.
        self.run_scenario("poll_shutdown")

    def test_shutdown_failure_notifies_pending_result_future(self):
        # A stream-sync exception should report COMM_ERROR to pending futures before being rethrown.
        self.run_scenario("shutdown_failure", fault_mode=30)

    def test_query_error_preserves_original_exception_after_log_grace(self):
        # An injected end-event error should wait for the log grace and preserve its original message.
        self.run_scenario("query_error", fault_mode=11)

    def test_driver_exit_query_preserves_original_exception_for_user(self):
        # A user-thread query should receive the original driver-exit error after the log grace.
        self.run_scenario("query_error", fault_mode=12)

    def test_watchdog_driver_exit_is_not_rethrown(self):
        # The watchdog entry should recognize a driver-exit query error without aborting the rank.
        self.run_scenario("watchdog_driver_exit", {"TORCH_HCCL_DUMP_ON_TIMEOUT": "0"}, fault_mode=15)

    def test_force_stop_query_uses_hccl_recovery_signal(self):
        # FORCE STOP should return incomplete, then be reported on the next communication submission.
        self.run_scenario("query_recovery", fault_mode=13)

    def test_device_error_query_uses_hccl_recovery_signal(self):
        # A recognized device error should use the HCCL recovery path instead of the generic rethrow.
        self.run_scenario("query_recovery", fault_mode=14)

    def test_stalled_stack_dump_retries_without_stack(self):
        # A stalled first stack export should time out and allow a readable no-stack retry.
        self.run_scenario("dump_stack_retry", {"TORCH_INCLUDE_STACK_TRACE": "1",
                                               "TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC": "500"}, fault_mode=1)

    def test_stalled_writer_reports_both_dump_timeouts(self):
        # A stalled writer should report both bounded waits without claiming that a file was saved.
        self.run_scenario("dump_writer_stall", {"TORCH_INCLUDE_STACK_TRACE": "1",
                                                "TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC": "500"}, fault_mode=2)

    def test_writer_exception_is_reported_by_fault_dump(self):
        # Writer exceptions should be logged for both attempts rather than escaping the monitor.
        self.run_scenario("dump_writer_error", fault_mode=2)

    def test_gil_stall_is_reported_by_registered_checker(self):
        # Hold the real Python GIL while the registered checker tries to acquire it.
        self.run_scenario("gil_block", fault_mode=21,
                          expected_logs=("Could not acquire GIL within 300 ms on exit",))

    def test_cpp_stack_dump_switch_controls_registered_callback(self):
        # The C++ stack switch should determine whether the registered dumper is actually called.
        for enabled in (0, 1):
            with self.subTest(cpp_stack=enabled):
                self.run_scenario("cpp_stack", {
                    "TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN": str(enabled)}, fault_mode=20)

    def test_point_to_point_completes_result_futures(self):
        # Send/recv should notify both result futures and preserve tensor contents.
        self.run_scenario("point_to_point")

    def test_coalesced_collective_completes_result_future(self):
        # A coalesced all-reduce should update every tensor and complete its result future.
        self.run_scenario("coalesced")

    def test_grouped_collectives_complete_aggregate_future(self):
        # The work returned by endCoalescing should track and finish the whole batch.
        self.run_scenario("grouped")

    def test_graph_capture_does_not_enqueue_watchdog_work(self):
        # Captured communication should stay out of the watchdog queue and replay correctly.
        self.run_scenario("graph_capture", {"TASK_QUEUE_ENABLE": "0"})


if __name__ == "__main__":
    unittest.main()
