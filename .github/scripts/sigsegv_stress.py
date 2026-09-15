#!/usr/bin/env python3
"""
Reproduction harness for the intermittent test_foreach worker SIGSEGV.

Background
    PR CI executes pre-collected cases through run_npu_test_shard.py in
    same-file batches of <=100: one resident worker subprocess per batch,
    one pytest.main() call per case, one NPU device per batch (round robin
    by batch id). Workers occasionally die with SIGSEGV between or early
    in cases. The runner then blames the first unreported case ("Worker
    killed by signal"), which is not necessarily the culprit: both
    foreach_clone (run 34837447197) and foreach_floor (run 34841388595)
    were innocent victims of this misattribution.

What this harness does
    Replays a chosen case range (default: batch 10857-10956 of the tensor
    shard, the batch that crashed in the runs above) through the same code
    path: run_npu_test_shard.py --worker with an identical batch_input.json,
    on the same NPU device, with the same env_updates the CI runner builds,
    in waves of N concurrent workers, until a crash or time budget is
    reached. On every worker death by signal it captures forensics:
    faulthandler output, core dump -> gdb full backtrace, per-iteration
    CANN plog, and the true in-flight case attribution.

Modes (--modes, comma separated, executed in this order)
    worker_batch        replay the whole batch per worker (reproduction)
    pair_case           two 2-case batches: the suspect pair (clamp_min +
                        clone, exercises aclnnForeachMaximumList + backward
                        through _foreach_clamp_min -> _foreach_maximum) and
                        a control pair (cos + cosh)
    subprocess_per_case one fresh pytest subprocess per case (control that
                        removes the pytest.main() in-process reuse variable)

Outputs (under --workdir)
    stats.jsonl         one JSON line per worker/case run
    stats.json          aggregate totals
    summary.md          human readable summary (also appended to
                        $GITHUB_STEP_SUMMARY when set)
    crash-forensics/    one directory per crash: worker.log (faulthandler),
                        gdb_bt_*.txt, plog.tar.gz, attribution.json,
                        cores.json, system snapshots

Exit codes
    0  finished (crashes, if any, are recorded in stats/forensics)
    2  environment-health abort, e.g. wheels/image mismatch
"""

import argparse
import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from pathlib import Path

NPU_FATAL_EXIT_CODE = 70  # keep in sync with run_npu_test_shard.py
MODE_BUDGET_SECONDS = {
    "worker_batch": 3600,
    "pair-suspect": 900,
    "pair-control": 900,
    "subprocess_per_case": 1800,
}
HISTORICAL_VICTIMS = (
    "test_outplace_with_invalid_grads__foreach_clone_npu_float32",
    "test_outplace_with_invalid_grads__foreach_floor_npu_float32",
)
STARTING_RE = re.compile(r"^\[(\d+)\] Starting: (.*)$")
MAX_KEEP_ITER_DIRS = 2
MAX_KEEP_CORES = 2


def signal_name(num):
    try:
        return signal.Signals(num).name
    except ValueError:
        return "SIG%d" % num


def classify_rc(rc, stopped_by_budget):
    if stopped_by_budget and rc < 0:
        return "stopped", signal_name(-rc)
    if rc == 0:
        return "ok", ""
    if rc == NPU_FATAL_EXIT_CODE:
        return "npu_fatal", ""
    if rc < 0:
        sig = signal_name(-rc)
        return "crash:" + sig, sig
    return "abnormal:%d" % rc, ""


@dataclasses.dataclass
class RunRecord:
    mode: str
    iter_num: int
    slot: int
    batch_id: int
    rc: int
    cls: str
    sig: str
    cases_completed: int
    last_completed_idx: int
    last_completed_nodeid: str
    inflight_idx: int
    inflight_nodeid: str
    duration_s: float
    pid: int
    crash_dir: str = ""


def parse_range(spec):
    parts = spec.split("-")
    if len(parts) != 2:
        raise SystemExit("range must look like 10857-10956, got %r" % spec)
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit("range must look like 10857-10956, got %r" % spec) from None
    if not 1 <= lo <= hi:
        raise SystemExit("invalid range %s" % spec)
    return lo, hi


def load_cases(cases_json):
    data = json.loads(Path(cases_json).read_text(encoding="utf-8"))
    planned = data.get("cases", [])
    return [
        {"case_idx": i, "nodeid": c["nodeid"], "test_file": c.get("file", "")}
        for i, c in enumerate(planned, 1)
    ]


def select_range(cases, spec):
    lo, hi = parse_range(spec)
    out = [c for c in cases if lo <= c["case_idx"] <= hi]
    if not out:
        raise SystemExit("no cases found in range %s" % spec)
    return out


def query_torch_root(python):
    code = "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))"
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode == 0:
        return proc.stdout.strip()
    return ""


def build_env_updates(runner_script, test_dir, python):
    # Mirror run_npu_test_shard.py build_execution_env(): the CI runner
    # passes exactly these env_updates to every worker via batch_input.json.
    test_dir_abs = Path(test_dir).resolve()
    parts = [str(Path(runner_script).resolve().parent)]
    torch_root = query_torch_root(python)
    if torch_root:
        parts.append(torch_root)
    parts.extend([str(test_dir_abs.parent), str(test_dir_abs)])
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return {
        "PYTHONPATH": os.pathsep.join(parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": "privateuse1",
    }


class Harness:
    def __init__(self, args, all_cases, env_updates):
        self.python = args.python
        self.runner_script = Path(args.runner_script).resolve()
        self.test_dir = Path(args.test_dir).resolve()
        self.workdir = Path(args.workdir).resolve()
        self.cores_dir = self.workdir / "cores"
        self.device = int(args.device)
        self.workers = max(1, int(args.workers))
        self.env_updates = env_updates
        self.deadline = time.monotonic() + float(args.max_total_minutes) * 60.0
        self.mode_end = self.deadline
        self.stats_path = self.workdir / "stats.jsonl"
        self.case_by_idx = {c["case_idx"]: c for c in all_cases}
        self.keep_iters = {}
        self.crash_counters = {}
        self.kept_cores = 0
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------- execution

    def run_wave(self, mode_label, iter_num, wave_cases, case_mode=False):
        iter_dir = self.workdir / mode_label / ("iter-%04d" % iter_num)
        iter_dir.mkdir(parents=True, exist_ok=True)
        records = [None] * self.workers
        stop_evt = threading.Event()
        live = []

        def _slot(slot):
            i = slot - 1
            case = wave_cases[i % len(wave_cases)]
            slot_dir = iter_dir / ("w%d" % slot)
            try:
                if case_mode:
                    records[i] = self._run_case_proc(mode_label, iter_num, slot, slot_dir, case, stop_evt, live)
                else:
                    records[i] = self._run_worker(mode_label, iter_num, slot, slot_dir, wave_cases, stop_evt, live)
            except Exception as exc:
                records[i] = RunRecord(
                    mode=mode_label, iter_num=iter_num, slot=slot, batch_id=0, rc=-1,
                    cls="spawn_error:%s" % type(exc).__name__, sig="", cases_completed=0,
                    last_completed_idx=0, last_completed_nodeid="",
                    inflight_idx=case["case_idx"], inflight_nodeid=case["nodeid"],
                    duration_s=0.0, pid=0)

        threads = []
        for slot in range(1, self.workers + 1):
            t = threading.Thread(target=_slot, args=(slot,), daemon=True)
            t.start()
            threads.append(t)

        wave_deadline = min(self.deadline, self.mode_end)
        while any(t.is_alive() for t in threads):
            if time.monotonic() >= wave_deadline:
                stop_evt.set()
                for proc in live:
                    if proc.poll() is None:
                        proc.terminate()
                break
            time.sleep(1.0)
        for t in threads:
            t.join(timeout=30.0)
        for proc in live:
            if proc.poll() is None:
                proc.kill()
        for proc in live:
            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pass
        return records, iter_dir

    def _run_worker(self, mode_label, iter_num, slot, slot_dir, batch_cases, stop_evt, live):
        slot_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = slot_dir / "reports"
        plog_dir = slot_dir / "plog"
        reports_dir.mkdir(exist_ok=True)
        plog_dir.mkdir(exist_ok=True)
        batch_id = iter_num * 1000 + slot
        batch_input = {
            "batch_id": batch_id,
            "test_dir": str(self.test_dir),
            "report_dir": str(reports_dir),
            "env_updates": self.env_updates,
            "timeout": 1200,
            "verbose": True,
            "shard": 1,
            "shard_type": "tensor",
            "npu_device_id": self.device,
            "cases": batch_cases,
        }
        bi_path = slot_dir / "batch_input.json"
        bi_path.write_text(json.dumps(batch_input, indent=1), encoding="utf-8")
        env = os.environ.copy()
        env.update(self.env_updates)
        env["PYTHONFAULTHANDLER"] = "1"
        env["ASCEND_PROCESS_LOG_PATH"] = str(plog_dir)
        cmd = [
            self.python, "-u", str(self.runner_script),
            "--worker", str(bi_path), "--test-dir", str(self.test_dir),
        ]
        return self._spawn_and_read(mode_label, iter_num, slot, slot_dir, cmd, env, batch_id, stop_evt, live)

    def _run_case_proc(self, mode_label, iter_num, slot, slot_dir, case, stop_evt, live):
        slot_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = slot_dir / "reports"
        plog_dir = slot_dir / "plog"
        reports_dir.mkdir(exist_ok=True)
        plog_dir.mkdir(exist_ok=True)
        nodeid = case["nodeid"]
        if nodeid.startswith("test/"):
            nodeid = nodeid[5:]
        xml = reports_dir / ("case-%d.xml" % case["case_idx"])
        cmd = [
            self.python, "-m", "pytest", "--color=no", "-ra", "--tb=short",
            nodeid, "--junitxml=%s" % xml, "--timeout=1200", "-vv",
        ]
        env = os.environ.copy()
        env.update(self.env_updates)
        env["PYTHONFAULTHANDLER"] = "1"
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device)
        env["ASCEND_PROCESS_LOG_PATH"] = str(plog_dir)
        (slot_dir / "cmd.json").write_text(
            json.dumps({"case": case, "cmd": cmd}, indent=1), encoding="utf-8")
        return self._spawn_and_read(
            mode_label, iter_num, slot, slot_dir, cmd, env, iter_num * 1000 + slot,
            stop_evt, live, case_mode=True, case=case)

    def _spawn_and_read(self, mode_label, iter_num, slot, slot_dir, cmd, env, batch_id,
                        stop_evt, live, case_mode=False, case=None):
        log_path = slot_dir / "worker.log"
        state = {"inflight": None, "completed": 0, "last_completed": (0, "")}
        wave_start_ts = time.time()
        t0 = time.monotonic()
        pid = 0
        rc = -1
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(self.test_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            pid = proc.pid
            live.append(proc)
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                if not case_mode:
                    self._parse_worker_line(line, state)
            rc = proc.wait()
        duration = time.monotonic() - t0
        cls, sig = classify_rc(rc, stop_evt.is_set())

        if case_mode:
            if rc >= 0:
                state["completed"] = 1
                state["last_completed"] = (case["case_idx"], case["nodeid"])
            else:
                state["inflight"] = case["case_idx"]

        inflight_idx = state["inflight"] or 0
        inflight_nodeid = self.case_by_idx.get(inflight_idx, {}).get("nodeid", "")
        last_idx, last_nodeid = state["last_completed"]
        rec = RunRecord(
            mode=mode_label, iter_num=iter_num, slot=slot, batch_id=batch_id, rc=rc,
            cls=cls, sig=sig, cases_completed=state["completed"],
            last_completed_idx=last_idx, last_completed_nodeid=last_nodeid,
            inflight_idx=inflight_idx, inflight_nodeid=inflight_nodeid,
            duration_s=duration, pid=pid)
        if cls.startswith("crash:"):
            self._forensics(rec, slot_dir, wave_start_ts, mode_label)
        self._write_stats(rec)
        return rec

    def _parse_worker_line(self, line, state):
        s = line.strip()
        if not s:
            return
        m = STARTING_RE.match(s)
        if m:
            try:
                state["inflight"] = int(m.group(1))
            except ValueError:
                return
            return
        if s.startswith("{") and '"case_idx"' in s:
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                return
            if isinstance(d, dict) and isinstance(d.get("case_idx"), int):
                state["completed"] += 1
                state["last_completed"] = (d["case_idx"], d.get("nodeid", ""))
                if state["inflight"] == d["case_idx"]:
                    state["inflight"] = None

    # --------------------------------------------------------------- drivers

    def run_batch_mode(self, mode_label, cases, budget_s, stop_crashes, health_guard=False):
        self.mode_end = time.monotonic() + budget_s
        crashes = 0
        bad_waves = 0
        iter_num = 0
        while True:
            if time.monotonic() >= min(self.mode_end, self.deadline):
                break
            if stop_crashes and crashes >= stop_crashes:
                break
            iter_num += 1
            records, iter_dir = self.run_wave(mode_label, iter_num, cases)
            wave_crash, wave_progress = self._after_wave(mode_label, iter_num, records, crashes)
            if wave_crash:
                crashes += 1
                shutil.rmtree(iter_dir, ignore_errors=True)
            else:
                self._rolling_keep(mode_label, iter_dir)
                bad_waves = bad_waves + 1 if not wave_progress else 0
                if health_guard and bad_waves >= 3:
                    self._env_abort(mode_label)
                    return crashes, True
        return crashes, False

    def run_subprocess_mode(self, mode_label, cases, budget_s, stop_crashes):
        self.mode_end = time.monotonic() + budget_s
        crashes = 0
        iter_num = 0
        n = len(cases)
        while True:
            if time.monotonic() >= min(self.mode_end, self.deadline):
                break
            if stop_crashes and crashes >= stop_crashes:
                break
            iter_num += 1
            wave_cases = [cases[((iter_num - 1) * self.workers + s) % n] for s in range(self.workers)]
            records, iter_dir = self.run_wave(mode_label, iter_num, wave_cases, case_mode=True)
            wave_crash, _ = self._after_wave(mode_label, iter_num, records, crashes)
            if wave_crash:
                crashes += 1
                shutil.rmtree(iter_dir, ignore_errors=True)
            else:
                self._rolling_keep(mode_label, iter_dir)
        return crashes, False

    def _after_wave(self, mode_label, iter_num, records, crashes):
        wave_crash = False
        wave_progress = False
        parts = []
        for rec in records:
            if rec is None:
                continue
            if rec.cls.startswith("crash:"):
                wave_crash = True
            if rec.cls in ("ok", "npu_fatal") or rec.cases_completed > 0 or rec.cls.startswith("crash:"):
                wave_progress = True
            parts.append("%s(rc=%d,done=%d)" % (rec.cls, rec.rc, rec.cases_completed))
        print("[%s] iter %04d: %s | crashes=%d" % (mode_label, iter_num, ",".join(parts), crashes), flush=True)
        return wave_crash, wave_progress

    # -------------------------------------------------------------- forensics

    def _forensics(self, rec, slot_dir, wave_start_ts, mode_label):
        n = self.crash_counters.get(mode_label, 0) + 1
        self.crash_counters[mode_label] = n
        crash_dir = self.workdir / "crash-forensics" / ("%s-crash-%02d" % (mode_label, n))
        crash_dir.mkdir(parents=True, exist_ok=True)
        for name in ("worker.log", "batch_input.json", "cmd.json"):
            src = slot_dir / name
            if src.exists():
                shutil.move(str(src), str(crash_dir / name))
        junit_dir = slot_dir / "reports" / "junit_xmls"
        if junit_dir.is_dir():
            shutil.copytree(str(junit_dir), str(crash_dir / "junit_xmls"), dirs_exist_ok=True)
        plog_dir = slot_dir / "plog"
        if plog_dir.is_dir() and any(plog_dir.iterdir()):
            try:
                with tarfile.open(str(crash_dir / "plog.tar.gz"), "w:gz") as tf:
                    tf.add(str(plog_dir), arcname="plog")
            except OSError as exc:
                (crash_dir / "plog_error.txt").write_text(str(exc), encoding="utf-8")
        attribution = {
            "mode": rec.mode,
            "signal": rec.sig,
            "returncode": rec.rc,
            "pid": rec.pid,
            "cases_completed_in_batch": rec.cases_completed,
            "last_completed": {
                "case_idx": rec.last_completed_idx,
                "nodeid": rec.last_completed_nodeid,
            },
            "inflight_when_crash": {
                "case_idx": rec.inflight_idx,
                "nodeid": rec.inflight_nodeid,
            },
            "historical_ci_victims": list(HISTORICAL_VICTIMS),
        }
        (crash_dir / "attribution.json").write_text(json.dumps(attribution, indent=1), encoding="utf-8")

        cores = self._find_cores(wave_start_ts, rec.pid)
        core_manifest = []
        for core in cores:
            bt = crash_dir / ("gdb_bt_%s.txt" % core.name)
            gdb_ok = self._gdb_backtrace(core, bt)
            if self.kept_cores < MAX_KEEP_CORES:
                dest = crash_dir / core.name
                shutil.move(str(core), str(dest))
                self.kept_cores += 1
                core_manifest.append({"file": str(dest), "kept": True, "gdb": gdb_ok})
            else:
                try:
                    size = core.stat().st_size
                except OSError:
                    size = -1
                core_manifest.append({"file": str(core), "kept": False, "gdb": gdb_ok, "size": size})
                core.unlink(missing_ok=True)
        if core_manifest:
            (crash_dir / "cores.json").write_text(json.dumps(core_manifest, indent=1), encoding="utf-8")
        self._system_snapshot(crash_dir)
        rec.crash_dir = str(crash_dir)
        print("!!! [%s] CRASH %s (pid %d, rc %d) -> %s" % (
            mode_label, rec.sig, rec.pid, rec.rc, crash_dir), flush=True)

    def _find_cores(self, wave_start_ts, pid):
        candidates = []
        for d in (self.cores_dir, self.test_dir):
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if not p.name.startswith("core"):
                    continue
                try:
                    if not p.is_file() or p.stat().st_mtime < wave_start_ts - 2:
                        continue
                except OSError:
                    continue
                candidates.append(p)
        matched = [p for p in candidates if str(pid) in p.name]
        return matched if matched else candidates

    def _gdb_backtrace(self, core, out_path):
        gdb = shutil.which("gdb")
        if gdb is None:
            out_path.write_text("gdb not available\n", encoding="utf-8")
            return False
        cmd = [
            gdb, "-batch",
            "-ex", "set pagination off",
            "-ex", "thread apply all bt full",
            "-ex", "info threads",
            "-ex", "info sharedlibrary",
            "--core", str(core), self.python,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            out_path.write_text("gdb timed out after 240s\n", encoding="utf-8")
            return False
        except OSError as exc:
            out_path.write_text("gdb failed: %s\n" % exc, encoding="utf-8")
            return False
        out_path.write_text(proc.stdout + "\n=== stderr ===\n" + proc.stderr, encoding="utf-8")
        return proc.returncode == 0

    def _system_snapshot(self, crash_dir):
        def _run(cmd):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except Exception as exc:
                return "failed: %s" % exc
            out = proc.stdout or ""
            if (proc.stderr or "").strip():
                out += "\n[stderr]\n" + proc.stderr
            return out

        (crash_dir / "npu_smi.txt").write_text(_run(["npu-smi", "info"]), encoding="utf-8")
        dmesg = _run(["dmesg"]).splitlines()
        (crash_dir / "dmesg_tail.txt").write_text("\n".join(dmesg[-100:]), encoding="utf-8")
        ps = _run(["ps", "-eLf"]).splitlines()
        (crash_dir / "ps_snapshot.txt").write_text("\n".join(ps[:2000]), encoding="utf-8")

    # -------------------------------------------------------------- utilities

    def _rolling_keep(self, mode_label, iter_dir):
        q = self.keep_iters.setdefault(mode_label, deque())
        q.append(iter_dir)
        while len(q) > MAX_KEEP_ITER_DIRS:
            shutil.rmtree(q.popleft(), ignore_errors=True)

    def _env_abort(self, mode_label):
        msg = (
            "ENVIRONMENT ABORT: %s completed no case in 3 consecutive waves with no crash.\n"
            "The wheels (from the source run) may be incompatible with the selected docker\n"
            "image (e.g. old torch_npu wheels inside a newer CANN image). If a new image was\n"
            "used, retry with the original image via the docker_image workflow input." % mode_label)
        print(msg, flush=True)
        (self.workdir / "ENV_ABORT.txt").write_text(msg, encoding="utf-8")

    def _write_stats(self, rec):
        row = dataclasses.asdict(rec)
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._stats_lock:
            with open(self.stats_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def dry_run(self, modes, all_cases, selected, args):
        print("\n=== dry run: planned invocations ===")
        for mode in modes:
            if mode == "worker_batch":
                self._print_batch_plan("worker_batch", selected)
            elif mode == "pair_case":
                for label, spec in (("pair-suspect", args.pair_suspect), ("pair-control", args.pair_control)):
                    self._print_batch_plan(label, select_range(all_cases, spec))
            else:
                case = selected[0]
                nodeid = case["nodeid"]
                if nodeid.startswith("test/"):
                    nodeid = nodeid[5:]
                print("\n[subprocess_per_case] case cmd (cycles through the whole range):")
                print("  cwd=%s" % self.test_dir)
                print("  %s -m pytest --color=no -ra --tb=short %s "
                      "--junitxml=<reports>/case-%d.xml --timeout=1200 -vv"
                      % (self.python, nodeid, case["case_idx"]))

    def _print_batch_plan(self, label, cases):
        batch_input = {
            "batch_id": 1001,
            "test_dir": str(self.test_dir),
            "report_dir": "<iter-dir>/w1/reports",
            "env_updates": self.env_updates,
            "timeout": 1200,
            "verbose": True,
            "shard": 1,
            "shard_type": "tensor",
            "npu_device_id": self.device,
            "cases": cases,
        }
        print("\n[%s] batch_input.json (slot 1 of %d):" % (label, self.workers))
        print(json.dumps(batch_input, indent=1))
        print("[%s] worker cmd:" % label)
        print("  cwd=%s" % self.test_dir)
        print("  %s -u %s --worker <batch_input.json> --test-dir %s"
              % (self.python, self.runner_script, self.test_dir))


def summarize(workdir):
    workdir = Path(workdir)
    rows = []
    stats_path = workdir / "stats.jsonl"
    if stats_path.is_file():
        with open(stats_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    by_mode = {}
    for r in rows:
        d = by_mode.setdefault(
            r.get("mode", "?"),
            {"runs": 0, "ok": 0, "npu_fatal": 0, "crashes": {}, "abnormal": 0,
             "stopped": 0, "spawn_error": 0})
        d["runs"] += 1
        cls = r.get("cls", "")
        if cls == "ok":
            d["ok"] += 1
        elif cls == "npu_fatal":
            d["npu_fatal"] += 1
        elif cls == "stopped":
            d["stopped"] += 1
        elif cls.startswith("crash:"):
            d["crashes"][cls] = d["crashes"].get(cls, 0) + 1
        elif cls.startswith("abnormal:"):
            d["abnormal"] += 1
        else:
            d["spawn_error"] += 1
    crash_rows = [r for r in rows if str(r.get("cls", "")).startswith("crash:")]
    victim_hits = 0
    for r in crash_rows:
        nodeid = r.get("inflight_nodeid") or ""
        if any(v in nodeid for v in HISTORICAL_VICTIMS):
            victim_hits += 1

    lines = []
    lines.append("# SIGSEGV repro summary")
    lines.append("")
    lines.append("| mode | runs | ok | npu_fatal | crashes | abnormal | stopped | spawn_error |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in sorted(by_mode):
        d = by_mode[m]
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (
            m, d["runs"], d["ok"], d["npu_fatal"], sum(d["crashes"].values()),
            d["abnormal"], d["stopped"], d["spawn_error"]))
    if crash_rows:
        lines.append("")
        lines.append("## Crashes (%d captured)" % len(crash_rows))
        lines.append("")
        lines.append("| mode | iter | slot | signal | completed | in-flight case |")
        lines.append("|---|---|---|---|---|---|")
        for r in crash_rows:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                r.get("mode"), r.get("iter_num"), r.get("slot"), r.get("sig"),
                r.get("cases_completed"), r.get("inflight_nodeid") or "(unknown)"))
        lines.append("")
        lines.append("In-flight case matched a historical CI victim in %d/%d crashes."
                     % (victim_hits, len(crash_rows)))
    else:
        lines.append("")
        lines.append("No crashes captured.")
    text = "\n".join(lines) + "\n"
    try:
        (workdir / "summary.md").write_text(text, encoding="utf-8")
        (workdir / "stats.json").write_text(
            json.dumps({"by_mode": by_mode, "crashes": crash_rows}, indent=1), encoding="utf-8")
    except OSError:
        pass
    print(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay a CI case batch to reproduce and forensic-capture worker SIGSEGV crashes.")
    parser.add_argument("--cases-json", help="tensor_cases_shard_1.json from the cases-shards artifact")
    parser.add_argument("--case-range", default="10857-10956",
                        help="1-based case_idx range replayed as one batch")
    parser.add_argument("--runner-script", help="path to run_npu_test_shard.py")
    parser.add_argument("--test-dir", default="pytorch/test")
    parser.add_argument("--workdir", default="repro-run")
    parser.add_argument("--device", default="5", help="NPU device id pinned for all workers")
    parser.add_argument("--workers", default="4", help="concurrent workers per wave")
    parser.add_argument("--modes", default="worker_batch,pair_case,subprocess_per_case")
    parser.add_argument("--max-total-minutes", default="120")
    parser.add_argument("--stop-after-crashes", default="3", help="crash samples per mode (0 = full budget)")
    parser.add_argument("--pair-suspect", default="10857-10858",
                        help="pair_case suspect range (clamp_min + clone)")
    parser.add_argument("--pair-control", default="10859-10860",
                        help="pair_case control range (cos + cosh)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="print planned commands and exit")
    parser.add_argument("--summarize-only", action="store_true", help="aggregate stats.jsonl and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only:
        summarize(args.workdir)
        return 0
    if not args.cases_json or not args.runner_script:
        raise SystemExit("--cases-json and --runner-script are required")
    if not Path(args.cases_json).is_file():
        raise SystemExit("cases json not found: %s" % args.cases_json)
    if not Path(args.runner_script).is_file():
        raise SystemExit("runner script not found: %s" % args.runner_script)
    if not Path(args.test_dir).is_dir():
        raise SystemExit("test dir not found: %s" % args.test_dir)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = ("worker_batch", "pair_case", "subprocess_per_case")
    unknown = [m for m in modes if m not in valid]
    if unknown or not modes:
        raise SystemExit("--modes must be a non-empty subset of %s" % ",".join(valid))

    all_cases = load_cases(args.cases_json)
    selected = select_range(all_cases, args.case_range)
    env_updates = build_env_updates(args.runner_script, args.test_dir, args.python)

    print("=" * 80, flush=True)
    print("SIGSEGV repro harness", flush=True)
    print("  cases: %d total, range %s -> %d cases" % (len(all_cases), args.case_range, len(selected)))
    print("  first: %s" % selected[0]["nodeid"])
    print("  last : %s" % selected[-1]["nodeid"])
    print("  device=%s workers=%s modes=%s budget=%smin stop_after_crashes=%s" % (
        args.device, args.workers, args.modes, args.max_total_minutes, args.stop_after_crashes))
    print("  env_updates=%s" % json.dumps(env_updates), flush=True)

    harness = Harness(args, all_cases, env_updates)
    if args.dry_run:
        harness.dry_run(modes, all_cases, selected, args)
        return 0

    harness.workdir.mkdir(parents=True, exist_ok=True)
    harness.cores_dir.mkdir(parents=True, exist_ok=True)

    aborted = False
    stop_crashes = int(args.stop_after_crashes)
    for mode in modes:
        remaining = harness.deadline - time.monotonic()
        if remaining < 20:
            print("skip %s: only %.0fs of budget left" % (mode, remaining), flush=True)
            continue
        if mode == "worker_batch":
            budget = min(MODE_BUDGET_SECONDS["worker_batch"], remaining)
            _, aborted = harness.run_batch_mode(
                "worker_batch", selected, budget, stop_crashes, health_guard=True)
        elif mode == "pair_case":
            suspect = select_range(all_cases, args.pair_suspect)
            control = select_range(all_cases, args.pair_control)
            for label, pair in (("pair-suspect", suspect), ("pair-control", control)):
                remaining = harness.deadline - time.monotonic()
                if remaining < 20:
                    break
                print("[pair_case] %s: %s" % (
                    label, ", ".join(c["nodeid"].split("::")[-1] for c in pair)), flush=True)
                harness.run_batch_mode(
                    label, pair, min(MODE_BUDGET_SECONDS[label], remaining), stop_crashes)
        else:
            harness.run_subprocess_mode(
                "subprocess_per_case", selected,
                min(MODE_BUDGET_SECONDS["subprocess_per_case"], remaining), stop_crashes)
        if aborted:
            break

    summarize(args.workdir)
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
