"""
mspti backend test.

Drives the libkineto PrivateUse1 plugin via torch.profiler. Device operators
export with bare names (matching torch.profiler GPU output), the NPU lane sorts
below the CPU lane, and each kernel carries an ac2g flow back to the operation
that launched it.

TestMsptiBackend    a single profiling window, and re-arming across cycles
TestMsptiAc2g       kernels linked to the operations that launched them
"""

import glob
import json
import os
import sys
import tempfile
import unittest

sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.getcwd()]

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401  # registers the Ascend PrivateUse1 backend


def _workload(device, steps=20):
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(512, 4096), nn.ReLU(), nn.Linear(4096, 512)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    x = torch.randn(256, 512, device=device)
    y = torch.randn(256, 512, device=device)
    for _ in range(steps):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    torch.npu.synchronize()


def _profile(steps=20):
    from torch.profiler import profile, ProfilerActivity
    device = torch.device("npu:0")
    _workload(device, steps=3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        _workload(device, steps=steps)
    with tempfile.TemporaryDirectory(prefix="mspti_") as tmpdir:
        path = os.path.join(tmpdir, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as fh:
            data = json.load(fh)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def _of_cat(events, cat):
    return [e for e in events if isinstance(e, dict) and str(e.get("cat", "")).lower() == cat]


@unittest.skipUnless(torch.npu.is_available(), "NPU not available")
class TestMsptiBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.events = _profile(steps=20)
        cls.kernels = _of_cat(cls.events, "kernel")

    def test_device_kernels_emitted(self):
        self.assertGreater(len(self.kernels), 0, "backend emitted no device kernels")

    def test_kernel_args_metadata(self):
        typed = [e for e in self.kernels
                 if isinstance(e.get("args"), dict)
                 and "type" in e["args"] and "streamId" in e["args"]]
        self.assertGreater(len(typed), 0, "kernel args missing type/streamId")

    def test_no_mspti_prefix(self):
        prefixed = [e for e in self.events
                    if isinstance(e, dict) and str(e.get("name", "")).startswith("mspti::")]
        self.assertEqual(prefixed, [], "activity names must not carry the 'mspti::' prefix")

    def test_device_lane_sorts_below_cpu(self):
        sort_idx = {e.get("pid"): e.get("args", {}).get("sort_index") for e in self.events
                    if isinstance(e, dict) and e.get("name") == "process_sort_index"}
        kernel_pids = {e.get("pid") for e in self.kernels}
        cpu_pids = {e.get("pid") for e in _of_cat(self.events, "cpu_op")}
        self.assertTrue(kernel_pids and cpu_pids)
        for kp in kernel_pids:
            for cp in cpu_pids:
                self.assertGreater(sort_idx.get(kp, 0), sort_idx.get(cp, 0),
                                   "NPU lane must sort below CPU lane")

    def test_every_schedule_cycle_collects(self):
        from torch.profiler import profile, schedule, ProfilerActivity
        device = torch.device("npu:0")
        repeat, holder = 2, {"n": 0}
        with tempfile.TemporaryDirectory(prefix="mspti_sched_") as outdir:
            def on_ready(prof):
                holder["n"] += 1
                prof.export_chrome_trace(os.path.join(outdir, "c%d.json" % holder["n"]))

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                         schedule=schedule(wait=1, warmup=1, active=3, repeat=repeat),
                         on_trace_ready=on_ready) as prof:
                for _ in range(repeat * 5):
                    _workload(device, steps=1)
                    prof.step()
            paths = sorted(glob.glob(os.path.join(outdir, "*.json")))
            self.assertEqual(len(paths), repeat)
            for path in paths:
                with open(path) as fh:
                    data = json.load(fh)
                events = data.get("traceEvents", data) if isinstance(data, dict) else data
                self.assertGreater(len(_of_cat(events, "kernel")), 0,
                                   "%s: no kernels; backend did not re-arm"
                                   % os.path.basename(path))


@unittest.skipUnless(torch.npu.is_available(), "NPU not available")
class TestMsptiAc2g(unittest.TestCase):
    """Counts are never asserted - mspti delivers records asynchronously, so
    kernel counts vary between runs. The assertions are structural instead."""

    @classmethod
    def setUpClass(cls):
        cls.events = _profile(steps=20)
        cls.flows = _of_cat(cls.events, "ac2g")
        cls.starts = [e for e in cls.flows if e.get("ph") == "s"]
        cls.finishes = [e for e in cls.flows if e.get("ph") == "f"]

    def test_flows_are_emitted(self):
        self.assertGreater(len(self.starts), 0, "no flow starts")
        self.assertGreater(len(self.finishes), 0, "no flow finishes")

    def test_every_start_has_a_finish(self):
        start_ids = {e["id"] for e in self.starts}
        finish_ids = {e["id"] for e in self.finishes}
        self.assertEqual(start_ids - finish_ids, set(), "flow starts with no finish")
        self.assertEqual(finish_ids - start_ids, set(), "flow finishes with no start")

    def test_one_start_per_operation(self):
        ids = [e["id"] for e in self.starts]
        self.assertEqual(len(ids), len(set(ids)), "duplicate flow start for one operation")

    def test_no_flow_points_backwards_in_time(self):
        started = {e["id"]: e["ts"] for e in self.starts}
        backwards = [e for e in self.finishes
                     if e["id"] in started and e["ts"] < started[e["id"]]]
        self.assertEqual(backwards, [], "flow arrows pointing backwards in time")

    def test_flow_ids_match_recorded_operations(self):
        flow_ids = {e["id"] for e in self.flows}
        op_ids = {e["args"]["External id"] for e in _of_cat(self.events, "cpu_op")
                  if isinstance(e.get("args"), dict) and "External id" in e["args"]}
        self.assertTrue(flow_ids & op_ids, "no flow id matches an operation in the trace")

    def test_ends_land_on_different_lanes(self):
        kernel_pids = {e.get("pid") for e in _of_cat(self.events, "kernel")}
        self.assertTrue(kernel_pids)
        for e in self.finishes:
            self.assertIn(e.get("pid"), kernel_pids)
        for e in self.starts:
            self.assertNotIn(e.get("pid"), kernel_pids)

    def test_flows_share_the_trace_timeline(self):
        ts = [e["ts"] for e in self.events
              if isinstance(e, dict) and isinstance(e.get("ts"), (int, float))]
        self.assertLess(max(ts) - min(ts), 60e6, "trace spans more than a minute")

if __name__ == "__main__":
    unittest.main(verbosity=2)
