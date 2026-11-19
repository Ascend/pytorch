# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Run cache stages in fresh processes; each child owns an empty cache root.

Cross-process reuse cannot be observed inside one interpreter: the package, the
FX graph cache and the AOT cache all key on state a single process already holds.
A stage therefore runs as a child with its own ``TORCHINDUCTOR_CACHE_DIR``, and a
consumer is pointed at the producer's root.

``--source-tree`` overlays the Python tree on top of the installed native
extension, so the children exercise the code in this checkout rather than the
wheel. That matters: without it a change here is invisible to the tests.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ENTRIES = ("package", "fx", "aot")


def create_overlay(installed, target, replacements, prefix=""):
    target.mkdir(exist_ok=True)
    names = {path.name for path in installed.iterdir()}
    names.update(key[len(prefix):].split("/")[0] for key in replacements if key.startswith(prefix))
    for name in names:
        key = prefix + name
        if key in replacements:
            (target / name).symlink_to(replacements[key])
        elif any(keypath.startswith(key + "/") for keypath in replacements):
            create_overlay(installed / name, target / name, replacements, key + "/")
        else:
            (target / name).symlink_to(installed / name)


class EntryRunner:
    """One child per stage, with a cache root the caller controls."""

    REPLACEMENTS = (
        "_inductor", "_compat", "npu/__init__.py", "utils/_dynamo.py",
        "utils/_dynamo_device.py", "utils/_triton.py", "dynamo/_package_compat",
    )

    def __init__(self, root, *, source_tree=None, timeout=900, device=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.scripts = Path(__file__).resolve().parent
        self.timeout = timeout
        self.env = os.environ.copy()
        self.env.update(TORCHINDUCTOR_NPU_BACKEND="dvm", TORCH_DEVICE_BACKEND_AUTOLOAD="0",
                        TORCH_WARM_POOL="0", TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="1")
        self._ensure_stdlibcxx()
        if device is not None:
            self.env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
        if source_tree:
            installed = Path(importlib.util.find_spec("torch_npu").origin).resolve().parent
            source = Path(source_tree).resolve() / "torch_npu"
            if installed == source:
                raise RuntimeError("Launch outside the source root, with an installed native extension")
            overlay = self.root / "overlay"
            (overlay / "torch_npu").mkdir(parents=True)
            create_overlay(installed, overlay / "torch_npu",
                           {name: source / name for name in self.REPLACEMENTS})
            self.env["PYTHONPATH"] = str(overlay) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.summary = {"runs": {}}

    def _ensure_stdlibcxx(self):
        """The cached Triton host extension needs a newer libstdc++ than the OS.

        Its ``npu_utils.so`` is linked against GLIBCXX_3.4.29 while the system
        copy stops at 3.4.28, so a build with a conda toolchain fails at import.
        Preload the environment's own copy when it is newer; leave a host that
        already satisfies it alone.
        """
        if os.environ.get("LD_PRELOAD"):
            return
        system = "/usr/lib64/libstdc++.so.6"
        candidate = Path(sys.prefix) / "lib" / "libstdc++.so.6"
        if not candidate.exists():
            return
        try:
            have = subprocess.run(["strings", system], capture_output=True, text=True).stdout
            want = subprocess.run(["strings", str(candidate)], capture_output=True, text=True).stdout
        except OSError:
            return
        if "GLIBCXX_3.4.29" in have or "GLIBCXX_3.4.29" not in want:
            return
        self.env["LD_PRELOAD"] = str(candidate)

    def run(self, name, entry, phase, *, cache, change_config=False, backend=None,
            backward=False, flip_init=False, fx_fallback=False):
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        child_env = self.env.copy()
        if backend is not None:
            # A consumer can be launched under a different backend than the one
            # that produced the cache, to check that restoration republishes
            # the producer's backend instead of trusting the environment.
            child_env["TORCHINDUCTOR_NPU_BACKEND"] = backend
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        child_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        # TRITON_CACHE_DIR, TMPDIR and TORCH_EXTENSIONS_DIR stay as inherited.
        # Triton keeps its npu_utils host extension under $HOME/.triton; pointing
        # its cache elsewhere makes every child rebuild that extension with the
        # toolchain's own -std=c++17, which this torch (C++20 ATen headers)
        # rejects. Only the Inductor cache root has to be per-stage, and the
        # Triton kernel cache it normally contains is simply left shared.
        command = [sys.executable, "-u", str(self.scripts / "compile_package_entry_worker.py"),
                   "--phase", phase, "--entry", entry, "--artifact", str(folder / "result.json")]
        if change_config:
            command.append("--change-config")
        if backward:
            command.append("--backward")
        if flip_init:
            command.append("--flip-init")
        if fx_fallback:
            command.append("--fx-fallback")
        started = time.monotonic()
        with (folder / "log.txt").open("w") as stream:
            child = subprocess.Popen(command, cwd=folder, env=child_env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._stop(child)
                code = 124
            except BaseException:
                self._stop(child)
                raise
        row = {"returncode": code, "seconds": round(time.monotonic() - started, 1)}
        result = folder / "result.json"
        if result.exists():
            try:
                row["result"] = json.loads(result.read_text())
            except ValueError as error:
                row["error"] = f"unreadable result.json: {error}"
        if code == 0:
            payload = row.get("result")
            if not isinstance(payload, dict) or payload.get("status") != "pass":
                row.update(returncode=1, error="child did not report a passing result.json")
        self.summary["runs"][name] = row
        return row

    @staticmethod
    def _stop(child):
        # The process group belongs to this runner, including compiler descendants.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                return
            try:
                child.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                continue
        child.wait()

    def pair(self, entry, *, cache=None, consumer_kwargs=None):
        """Producer then consumer on one cache root; returns both rows."""
        cache = cache or (self.root / entry / "cache")
        producer = self.run(f"{entry}_producer", entry, "producer", cache=cache)
        consumer = self.run(f"{entry}_consumer", entry, "consumer", cache=cache,
                            **(consumer_kwargs or {}))
        return producer, consumer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, help="Repository root to overlay")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entry", choices=(*ENTRIES, "all"), default="all")
    parser.add_argument("--device")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)

    root = args.output or Path(tempfile.mkdtemp(prefix="npu-entries-")) / "run"
    runner = EntryRunner(root, source_tree=args.source_tree, timeout=args.timeout,
                         device=args.device)
    entries = ENTRIES if args.entry == "all" else (args.entry,)
    for entry in entries:
        runner.pair(entry)
    runner.summary["status"] = "pass" if all(
        row["returncode"] == 0 for row in runner.summary["runs"].values()) else "fail"
    (root / "summary.json").write_text(json.dumps(runner.summary, indent=2))
    print(json.dumps({name: {"returncode": row["returncode"], "seconds": row["seconds"]}
                      for name, row in runner.summary["runs"].items()}, indent=2))
    print(f"{runner.summary['status']}: {root / 'summary.json'}")
    return int(runner.summary["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
