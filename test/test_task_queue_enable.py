import os
import time
import threading
import subprocess
import sys
import tempfile

import torch
import torch_npu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTaskQueueEnableEnv(TestCase):

    def setUp(self):
        super().setUp()
        self.original_mode = torch.npu.get_task_queue_enable()

    def tearDown(self):
        torch.npu.set_task_queue_enable(self.original_mode)
        super().tearDown()

    def test_blocking_override(self):
        blocking = os.environ.get("ASCEND_LAUNCH_BLOCKING", "0")
        if blocking != "1":
            print("  Skipped: ASCEND_LAUNCH_BLOCKING != 1")
            return
        torch.npu.set_task_queue_enable(2)
        self.assertEqual(torch.npu.get_task_queue_enable(), 0,
                         "ASCEND_LAUNCH_BLOCKING=1 should force mode to 0")

    def test_invalid_mode(self):
        for invalid_mode in (-1, 3, -2):
            with self.assertRaises(RuntimeError):
                torch.npu.set_task_queue_enable(invalid_mode)

    def test_mode_roundtrip_and_switch(self):
        for mode in (0, 1, 2):
            torch.npu.set_task_queue_enable(mode)
            self.assertEqual(torch.npu.get_task_queue_enable(), mode)

    def test_npu_graph_capture_rejected_at_mode2(self):
        torch.npu.set_task_queue_enable(2)
        g = torch.npu.NPUGraph()
        static_in = torch.randn(16, 16, device="npu")
        with self.assertRaisesRegex(RuntimeError, "TASK_QUEUE_ENABLE"):
            g.capture_begin()
            static_out = static_in * 2
            g.capture_end()

    def test_per_stream_queue_interaction_mode0(self):
        env = os.environ.copy()
        env["TASK_QUEUE_ENABLE"] = "1"
        env["PER_STREAM_QUEUE"] = "1"
        env.pop("ASCEND_LAUNCH_BLOCKING", None)
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "psq_mode0.py")
            with open(script, "w") as f:
                f.write(
                    "import torch\n"
                    "import torch_npu\n"
                    "s = torch.npu.Stream()\n"
                    "x = torch.randn(32, 32, device='npu')\n"
                    "with torch.npu.stream(s):\n"
                    "    z = x + 1\n"
                    "s.synchronize()\n"
                    "assert torch.equal(z.cpu(), (x + 1).cpu())\n"
                    "torch.npu.set_task_queue_enable(0)\n"
                    "z2 = x + 2\n"
                    "torch.npu.synchronize()\n"
                    "assert torch.equal(z2.cpu(), (x + 2).cpu())\n"
                    "print('OK')\n"
                )
            res = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(
                res.returncode, 0,
                f"per-stream queue + dynamic switch failed:\n{res.stderr[-2000:]}")

    def test_deterministic_level_not_reset_by_mode_switch(self):
        original_level = torch_npu._C._npu_get_deterministic_level()
        original_det = torch.are_deterministic_algorithms_enabled()
        try:
            for level in (1, 2):
                torch_npu.npu.set_deterministic_level(level)
                self.assertEqual(torch_npu._C._npu_get_deterministic_level(), level)
                for mode in (0, 1, 2):
                    torch.npu.set_task_queue_enable(mode)
                    self.assertEqual(
                        torch_npu._C._npu_get_deterministic_level(), level,
                        f"deterministic level changed after switching task queue mode={mode}")
        finally:
            torch_npu.npu.set_deterministic_level(original_level)
            torch.use_deterministic_algorithms(original_det)


class TestGetTaskQueueEnableTiming(TestCase):

    GET_REPEATS = 3
    WARMUP = 1000
    REPEATS = 200

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_task_queue_enable_timing_all_paths(self):
        for _ in range(self.GET_REPEATS):
            _ = torch.npu.get_task_queue_enable()

        torch.npu.set_task_queue_enable(1)
        torch.npu.synchronize()
        for _ in range(self.GET_REPEATS):
            _ = torch.npu.get_task_queue_enable()

    def test_get_task_queue_enable_call_latency(self):
        print("\n" + "=" * 70)
        print("[Timing] get_task_queue_enable() call latency")
        print("=" * 70)

        torch.npu.set_task_queue_enable(1)
        for _ in range(self.WARMUP):
            _ = torch.npu.get_task_queue_enable()

        start = time.perf_counter_ns()
        for _ in range(self.REPEATS):
            _ = torch.npu.get_task_queue_enable()
        fast_ns = (time.perf_counter_ns() - start) / self.REPEATS
        print(f"  fast path (set=1): {fast_ns:.1f} ns/call ({self.REPEATS} calls)")

        torch.npu.set_task_queue_enable(2)
        for _ in range(self.WARMUP):
            _ = torch.npu.get_task_queue_enable()

        start = time.perf_counter_ns()
        for _ in range(self.REPEATS):
            _ = torch.npu.get_task_queue_enable()
        cold_ns = (time.perf_counter_ns() - start) / self.REPEATS
        print(f"  set=2 path: {cold_ns:.1f} ns/call ({self.REPEATS} calls)")

        delta = cold_ns - fast_ns
        ratio = cold_ns / fast_ns if fast_ns > 0 else 0
        print(f"  delta={delta:.1f} ns, ratio={ratio:.2f}x")

    def test_multithread_get_dispatch(self):
        print("\n" + "=" * 70)
        print("[Multi] concurrent get_task_queue_enable under mode switching")
        print("=" * 70)

        for num_readers in [1, 2, 4]:
            stop_event = threading.Event()
            errors = []
            get_counts = {}

            def writer():
                try:
                    for i in range(200):
                        torch.npu.set_task_queue_enable(i % 3)
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(f"writer: {e}")
                finally:
                    stop_event.set()

            def reader(tid):
                try:
                    count = 0
                    while not stop_event.is_set():
                        mode = torch.npu.get_task_queue_enable()
                        if mode not in (0, 1, 2):
                            errors.append(f"reader {tid}: invalid mode {mode}")
                        count += 1
                    get_counts[tid] = count
                except Exception as e:
                    errors.append(f"reader {tid}: {e}")

            wt = threading.Thread(target=writer)
            rts = [threading.Thread(target=reader, args=(i,)) for i in range(num_readers)]

            start = time.perf_counter()
            wt.start()
            for t in rts:
                t.start()
            wt.join()
            for t in rts:
                t.join()
            elapsed = time.perf_counter() - start

            total_gets = sum(get_counts.values())
            avg_ns = (elapsed / total_gets) * 1e9 if total_gets > 0 else 0
            print(f"  {num_readers} reader(s): {total_gets} gets, avg={avg_ns:.1f} ns/get")
            self.assertEqual(len(errors), 0, f"Errors: {errors}")


if __name__ == "__main__":
    run_tests()
