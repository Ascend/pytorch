import unittest

import torch
import torch_npu
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion, SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

# The native external path (Event(external=True)) runs on A5 with CANN 9.3.0+
# (aclrtCreateEventExWithFlag / aclrtRecordEventWithFlag /
# aclrtStreamWaitEventWithFlag).


def is_ex_graph_runtime_supported():
    # Runtime capability probe: the in-graph EXTERNAL record is submitted
    # asynchronously - the call itself returns success even when the
    # device-side tsd capability check rejects the feature (the event is
    # then never recorded, so the follow-up timing API fails). The feature
    # requires a host whose device-side tsd supports the external event
    # capability, so probe it once with a real capture + timing round-trip;
    # the gate opens automatically once a capable host is used.
    try:
        g = torch.npu.NPUGraph()
        s = torch.npu.Event(enable_timing=True, external=True)
        e = torch.npu.Event(enable_timing=True, external=True)
        x = torch.ones(4, device="npu")
        with torch.npu.graph(g):
            s.record()
            x = x.add(1.0)
            e.record()
        g.replay()
        torch.npu.synchronize()
        return s.elapsed_time(e) > 0
    except Exception:  # noqa: BLE001
        return False


skip_ex_graph = not is_ex_graph_runtime_supported()

SKIP_EX_GRAPH_REASON = (
    "in-graph EXTERNAL record is rejected by the device-side tsd capability "
    "check (async 507899, the event is never recorded): requires a host whose "
    "tsd/firmware supports the external event feature"
)


def is_ipc_event_supported():
    try:
        torch.npu.Event(enable_timing=False, interprocess=True)
    except RuntimeError:
        return False
    return True


skip_ipc_event_case = not is_ipc_event_supported()


class TestExternalEvent(TestCase):
    ############################
    # Layer 1: signature level
    ############################

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_event_external_kwarg(self):
        # Event gains the external kwarg.
        ev = torch.npu.Event(external=True)
        self.assertIsInstance(ev, torch.npu.Event)
        ev.record()
        torch.npu.synchronize()
        # event_id returns the underlying ACL event handle.
        self.assertIsInstance(ev.event_id, int)
        self.assertGreater(ev.event_id, 0)
        self.assertEqual(ev.event_id, ev.npu_event)

    def test_graph_external_kwarg_removed(self):
        # graph_external was an internal kwarg of the old torch_npu Event and
        # is fully removed (API incompatible): both the Python level signature
        # and the C++ binding kwlist must reject it with a TypeError.
        with self.assertRaisesRegex(TypeError, "graph_external"):
            torch.npu.Event(graph_external=True)
        with self.assertRaisesRegex(TypeError, "graph_external"):
            torch_npu._C._NPUEventBase(graph_external=True)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_event_external_interprocess_mutex(self):
        with self.assertRaisesRegex(RuntimeError, "interprocess"):
            torch.npu.Event(external=True, interprocess=True)

    ############################
    # Layer 4: NPU device, native external path
    ############################

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_record_wait_flag_by_capture_status(self):
        # Non-capture record/wait degrade to DEFAULT flags (align with CUDA:
        # cudaEventRecordExternal outside capture behaves like a normal record).
        ev = torch.npu.Event(external=True)
        ev.record()
        ev.wait()
        torch.npu.synchronize()
        self.assertTrue(ev.query())

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_capture_record_wait_graph(self):
        # During capture, record/wait become external event nodes of the graph.
        # NOTE: a non-capture (default) record can NOT be consumed by an
        # in-capture wait (CANN constraint EE1018, covered by the application
        # level rejection test below), so only the in-capture record segment
        # is exercised here.
        ev2 = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        buf = torch.zeros(8, device="npu")
        with torch.npu.graph(g):
            buf.add_(1.0)
            ev2.record()
        g.replay()
        torch.npu.synchronize()
        self.assertTrue(ev2.query())
        self.assertEqual(buf.sum().item(), 8.0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_query_reports_real_state(self):
        # query() reports the real record-node execution state from the
        # runtime: True before the first replay (empty work set, CUDA
        # parity), True once the record node has executed. The in-flight
        # "False" window is a race (the workload may finish before the
        # query call completes) and is therefore not asserted.
        ev = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        a = torch.randn(4096, 4096, device="npu")
        with torch.npu.graph(g):
            y = a
            for _ in range(50):
                y = y @ a
            ev.record()
        self.assertTrue(ev.query())
        g.replay()
        ev.synchronize()
        self.assertTrue(ev.query())

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_flag_no_switch(self):
        # The record mode bound on first use cannot be switched afterwards (ACL
        # runtime constraint): an in-graph EXTERNAL record followed by an
        # out-of-graph DEFAULT record is rejected synchronously by the direct
        # runtime call.
        ev = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            ev.record()  # binds ACL_EVENT_RECORD_EXTERNAL
        with self.assertRaisesRegex(RuntimeError, "out-of-graph record failed"):
            ev.record()  # outside capture -> ACL_EVENT_RECORD_DEFAULT, mode switch rejected
        torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_wait_flag_no_switch(self):
        # The wait mode bound on first use cannot be switched afterwards (ACL
        # runtime constraint, symmetric to the record mode): the out-of-graph
        # DEFAULT wait is submitted directly so a mode-switch rejection
        # surfaces synchronously. The event is recorded in a SEPARATE graph:
        # the same event cannot be both EXTERNAL-recorded and EXTERNAL-waited
        # in the same graph (ACL constraint, verified by direct-ACL probe).
        ev = torch.npu.Event(external=True)
        g1 = torch.npu.NPUGraph()
        with torch.npu.graph(g1):
            ev.record()  # binds ACL_EVENT_RECORD_EXTERNAL
        g2 = torch.npu.NPUGraph()
        with torch.npu.graph(g2):
            ev.wait()  # binds ACL_EVENT_WAIT_EXTERNAL
        with self.assertRaisesRegex(RuntimeError, "out-of-graph wait failed"):
            ev.wait()  # outside capture -> ACL_EVENT_WAIT_DEFAULT, mode switch rejected
        torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_wait_after_default_record_rejected(self):
        # A default-recorded (non-capture) event can no longer be waited
        # inside graph capture (CANN constraint EE1018): the runtime
        # rejection now surfaces directly from the in-graph wait call.
        ev = torch.npu.Event(external=True)
        ev.record()  # binds ACL_EVENT_RECORD_DEFAULT (non-capture)
        torch.npu.synchronize()
        g2 = torch.npu.NPUGraph()
        with torch.npu.graph(g2):
            with self.assertRaisesRegex(RuntimeError, "in-graph wait failed"):
                ev.wait()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_same_graph_record_wait_rejected(self):
        # The same event cannot be both EXTERNAL-recorded and EXTERNAL-waited
        # in the same graph (ACL constraint, both directions rejected by the
        # runtime at the second call). CUDA allows record+wait on the same
        # event within one graph. In-graph producer-consumer choreography must
        # split into two graphs or use two distinct events.
        ev = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            ev.record()
            with self.assertRaisesRegex(RuntimeError, "in-graph wait failed"):
                ev.wait()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_uncreated_event_wait_is_noop(self):
        # An uncreated event's wait() is a silent no-op (CUDA-consistent lazy
        # creation): the in-graph wait does not bind the wait flag, so the
        # following in-graph record succeeds and the reverse same-graph mutex
        # is unreachable via the public API. After capture the record mode is
        # bound to EXTERNAL: the out-of-graph record is rejected (mode switch),
        # while the out-of-graph wait is accepted (wait mode never bound).
        ev = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            ev.wait()
            ev.record()
        g.replay()
        torch.npu.synchronize()
        ev.wait()
        with self.assertRaisesRegex(RuntimeError, "out-of-graph record failed"):
            ev.record()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_reset_rejected(self):
        # reset() is not supported for native external events: they can be
        # waited by multiple streams without reset (CUDA events have no reset).
        ev = torch.npu.Event(external=True)
        ev.record()
        with self.assertRaisesRegex(RuntimeError, "reset"):
            ev.reset(torch.npu.current_stream())

        # Legacy ExternalEvent keeps the reset behavior.
        legacy = torch.npu.ExternalEvent()
        legacy.record()
        legacy.wait()
        torch.npu.synchronize()
        legacy.reset(torch.npu.current_stream())

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_synchronize(self):
        # Out-of-graph (default) record: the event is a regular record, host
        # side synchronize and query report the real device state.
        ev = torch.npu.Event(external=True)
        ev.record()
        ev.synchronize()
        self.assertTrue(ev.query())

        # In-graph EXTERNAL record: aclrtSynchronizeEvent returns immediately
        # if the record node has not been executed and waits for the replay
        # otherwise (align with cudaEventSynchronize on graph-recorded events).
        ev2 = torch.npu.Event(external=True)
        g = torch.npu.NPUGraph()
        buf = torch.zeros(8, device="npu")
        with torch.npu.graph(g):
            buf.add_(1.0)
            ev2.record()
        g.replay()
        ev2.synchronize()
        torch.npu.synchronize()
        self.assertTrue(ev2.query())
        self.assertEqual(buf.sum().item(), 8.0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_elapsed_time_requires_timing(self):
        # Without the TIME_LINE flag (external without enable_timing),
        # elapsed_time is rejected by the enable_timing check.
        ev_a = torch.npu.Event(external=True)
        ev_b = torch.npu.Event(external=True)
        ev_a.record()
        ev_b.record()
        torch.npu.synchronize()
        with self.assertRaisesRegex(RuntimeError, "enable_timing"):
            ev_a.elapsed_time(ev_b)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    @unittest.skipIf(skip_ex_graph, SKIP_EX_GRAPH_REASON)
    def test_graph_timing_native(self):
        # An external event created with enable_timing=True can be recorded
        # inside a capture and timed after replay. Requires a driver/firmware
        # matched with the CANN version.
        g = torch.npu.NPUGraph()
        start = torch.npu.Event(enable_timing=True, external=True)
        end = torch.npu.Event(enable_timing=True, external=True)
        x = torch.ones(1024, 1024, device="npu")
        with torch.npu.graph(g):
            start.record()
            for _ in range(10):
                x = x.add(1.0).mm(x)
            end.record()
        g.replay()
        torch.npu.synchronize()
        elapsed = start.elapsed_time(end)
        self.assertGreater(elapsed, 0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_graph_external_event_capture(self):
        # Record/wait on an external event inside captures become external
        # nodes. The consumer graph should be replayed after the producer
        # graph: the in-graph EXTERNAL wait does not hold for a
        # not-yet-recorded event on the current runtime.
        g1 = torch.npu.NPUGraph()
        g2 = torch.npu.NPUGraph()
        ev = torch.npu.Event(external=True)
        s2 = torch.npu.Stream()
        s2.wait_stream(torch.npu.current_stream())

        buf1 = torch.zeros(1024, device="npu")
        buf2 = torch.zeros(1024, device="npu")

        with torch.npu.graph(g1):
            buf1.add_(1.0)
            ev.record()

        with torch.npu.stream(s2):
            with torch.npu.graph(g2, stream=s2):
                ev.wait()
                buf2.copy_(buf1)

        # Producer replayed first, then the consumer: the external wait node
        # sees the recorded event and the copy observes the producer result.
        g1.replay()
        g2.replay()
        torch.npu.synchronize()
        self.assertEqual(buf2.sum().item(), 1024.0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_cross_stream_record_wait(self):
        # Cross-stream record + wait establishes a real data dependency: the
        # consumer observes the producer's write after the eager wait.
        s = torch.npu.Stream()
        ev = torch.npu.Event(external=True)
        x = torch.zeros(1024, device="npu")
        with torch.npu.stream(s):
            x.add_(1.0)
            ev.record()
        ev.wait()
        torch.npu.synchronize()
        self.assertEqual(x.sum().item(), 1024.0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_eager_timing(self):
        # Eager (out-of-graph) timing with external events.
        start = torch.npu.Event(enable_timing=True, external=True)
        end = torch.npu.Event(enable_timing=True, external=True)
        start.record()
        x = torch.ones(1024, 1024, device="npu")
        for _ in range(10):
            x = x.add(1.0).mm(x)
        end.record()
        torch.npu.synchronize()
        self.assertGreater(start.elapsed_time(end), 0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_wrap_replay_timing(self):
        # Timing events recorded OUTSIDE the capture wrap the replay. This is
        # the fallback path for hosts whose device side does not support the
        # in-graph EXTERNAL record (no tsfw capability required).
        st = torch.npu.Stream()
        with torch.npu.stream(st):
            x = torch.ones(1024, 1024, device="npu")
            x @ x
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, stream=st):
            for _ in range(10):
                x = x.add(1.0).mm(x)
        start = torch.npu.Event(enable_timing=True, external=True)
        end = torch.npu.Event(enable_timing=True, external=True)
        start.record()
        g.replay()
        end.record()
        torch.npu.synchronize()
        self.assertGreater(start.elapsed_time(end), 0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_eager_record_then_ingraph_record_rejected(self):
        # Mirror of test_native_flag_no_switch: an eager (default) record
        # binds the DEFAULT mode first, the later in-capture record is a
        # direct runtime call, and the mode-switch rejection surfaces from
        # the call.
        ev = torch.npu.Event(external=True)
        ev.record()
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        buf = torch.zeros(8, device="npu")
        with torch.npu.graph(g):
            buf.add_(1.0)
            with self.assertRaisesRegex(RuntimeError, "in-graph record failed"):
                ev.record()

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    @unittest.skipIf(skip_ex_graph, SKIP_EX_GRAPH_REASON)
    def test_native_warmup_without_record_timing(self):
        # vllm-style timer: the warmup runs eagerly WITHOUT recording the
        # timing events, so the first record happens inside the capture, no
        # flag switch occurs and the in-graph timing works.
        st = torch.npu.Stream()
        with torch.npu.stream(st):
            x = torch.ones(1024, 1024, device="npu")
            for _ in range(3):
                x = x.add(1.0).mm(x)
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True, external=True)
        end = torch.npu.Event(enable_timing=True, external=True)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, stream=st):
            start.record()
            for _ in range(10):
                x = x.add(1.0).mm(x)
            end.record()
        g.replay()
        torch.npu.synchronize()
        self.assertGreater(start.elapsed_time(end), 0)

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_repeated_ingraph_record(self):
        # Once the EXTERNAL record flag is bound, another graph can record
        # the same event again (EXTERNAL -> EXTERNAL is allowed).
        ev = torch.npu.Event(external=True)
        st = torch.npu.Stream()
        with torch.npu.stream(st):
            x = torch.ones(8, device="npu")
            x @ x
        torch.npu.synchronize()
        for _ in range(2):
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g, stream=st):
                x @ x
                ev.record()
            g.replay()
        torch.npu.synchronize()
        self.assertTrue(ev.query())

    @SupportedDevices(["Ascend950"])
    @SkipIfNotGteCANNVersion("9.3.0")
    def test_native_consumer_first_replay_reads_stale(self):
        # An in-graph EXTERNAL wait does NOT hold for a not-yet-recorded
        # event (no-op, same as CUDA): replaying the consumer graph before
        # the producer reads stale data.
        ev = torch.npu.Event(external=True)
        s2 = torch.npu.Stream()
        s2.wait_stream(torch.npu.current_stream())
        buf1 = torch.zeros(1024, device="npu")
        buf2 = torch.zeros(1024, device="npu")
        g1 = torch.npu.NPUGraph()
        with torch.npu.graph(g1):
            buf1.add_(1.0)
            ev.record()
        g2 = torch.npu.NPUGraph()
        with torch.npu.stream(s2):
            with torch.npu.graph(g2, stream=s2):
                ev.wait()
                buf2.copy_(buf1)
        with torch.npu.stream(s2):
            g2.replay()  # consumer first: the wait does not hold
        g1.replay()
        torch.npu.synchronize()
        self.assertEqual(buf2.sum().item(), 0.0)

    ############################
    # Layer 4: NPU device, legacy ExternalEvent path
    ############################

    def test_legacy_external_event_behavior(self):
        # torch_npu.ExternalEvent keeps its original behavior unchanged.
        # NOTE: wait-before-record is NOT covered here - that sequence hangs
        # at the next device synchronize on value-write hosts (pre-existing
        # behavior, out of scope for this change).
        s = torch.npu.current_stream()
        ev = torch.npu.ExternalEvent()
        ev.record(s)
        ev.wait(s)
        torch.npu.synchronize()
        ev.reset(s)
        ev.record(s)
        ev.wait(s)
        torch.npu.synchronize()

        # record inside a capture (legacy value write path).
        ev2 = torch.npu.ExternalEvent()
        g = torch.npu.NPUGraph()
        buf = torch.zeros(8, device="npu")
        with torch.npu.graph(g):
            buf.add_(1.0)
            ev2.record()
        g.replay()
        torch.npu.synchronize()
        self.assertEqual(buf.sum().item(), 8.0)

    def test_legacy_query_state_tracking(self):
        # Legacy query() must not reach the runtime with the 32-byte memory
        # block: host-side state tracking reports not-recorded -> False,
        # recorded -> True and stays True after the wait consumed it
        # (always-True semantics of external events).
        s = torch.npu.current_stream()
        ev = torch.npu.ExternalEvent()
        self.assertFalse(ev.query())
        ev.record(s)
        torch.npu.synchronize()
        self.assertTrue(ev.query())
        ev.wait(s)
        torch.npu.synchronize()
        self.assertTrue(ev.query())

    def test_legacy_timing_not_supported(self):
        # Legacy value-write external events do not support timing: explicit
        # NOT_SUPPORT with guidance instead of the previous SIGSEGV.
        ev_a = torch.npu.ExternalEvent()
        ev_b = torch.npu.ExternalEvent()
        ev_a.record()
        ev_b.record()
        torch.npu.synchronize()
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            ev_a.elapsed_time(ev_b)
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            ev_a.recorded_time()

    def test_legacy_second_wait_reject(self):
        # A second wait without reset used to hang forever on
        # ValueWait(value == 1): it must now be rejected explicitly with a
        # reset hint.
        s = torch.npu.current_stream()
        ev = torch.npu.ExternalEvent()
        ev.record(s)
        ev.wait(s)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            ev.wait(s)

        # reset -> record -> wait reuse chain keeps working.
        ev.reset(s)
        ev.record(s)
        ev.wait(s)
        torch.npu.synchronize()

    ############################
    # Layer 4: IPC event (Ex API creation path) and profiler smoke
    ############################

    @unittest.skipIf(skip_ipc_event_case, "IPC event is not supported by the current driver/CANN")
    def test_ipc_event_ex_api(self):
        # Event(interprocess=True) is created through the Ex API path.
        ev = torch.npu.Event(enable_timing=False, interprocess=True)
        ev.record()
        torch.npu.synchronize()
        self.assertTrue(ev.query())
        handle = ev.ipc_handle()
        self.assertIsInstance(handle, bytes)
        # In-process IPC handle import is not supported by the driver: IPC
        # handles are for cross-process use only. The cross-process
        # semantics stay covered by test_ipc_event.py (multiprocessing).
        self.skipTest("in-process IPC handle round-trip is not supported by the driver (cross-process only)")

    def test_profiler_event_smoke(self):
        # Smoke test: timing events and profiler mark work under an active
        # profiler (the profiler mark path shares the event creation wrapper).
        from torch_npu.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            x = torch.ones(64, 64, device="npu")
            y = x.mm(x)
            end.record()
            torch.npu.synchronize()
            elapsed = start.elapsed_time(end)
            self.assertGreaterEqual(elapsed, 0)
        self.assertTrue(x.is_npu)


if __name__ == "__main__":
    run_tests()
