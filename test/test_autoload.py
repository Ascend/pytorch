# Owner(s): ["module: unknown"]

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase, run_tests)

AUTO_LOAD = hasattr(torch, "_is_device_backend_autoload_enabled") and torch._is_device_backend_autoload_enabled()
RUN_NPU = AUTO_LOAD and torch.npu.is_available()


class AutoloadTest(TestCase):

    # torch_npu should be imported implicitly after running 'import torch'
    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload(self):
        self.assertTrue("torch_npu" in sys.modules)

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload_tensor(self):
        ones_npu = torch.ones(5, 5, device="npu")
        self.assertEqual(ones_npu.device.type, "npu")

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload_model(self):
        class Model(nn.Module):
            def __init__(self, input_size, num_classes):
                super(Model, self).__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                out = self.fc(x)
                return out

        model = Model(10, 2)
        model = model.to("npu")

        x = torch.randn(64, 10, device="npu")
        outputs = model(x)
        self.assertEqual(outputs.device.type, "npu")
        self.assertTupleEqual(outputs.shape, (64, 2))


ENV_KEY = "TORCH_DEVICE_BACKEND_AUTOLOAD"

# Runs in a fresh interpreter so that the import order and the starting
# environment are fully controlled, which is impossible in-process due to
# the sys.modules cache. After the import sequence it spawns/forks
# multiprocessing children to also verify env inheritance and autoload
# behavior in child processes. Written to a temp file because the spawn
# start method requires an importable __main__ (not possible with -c).
# Each probe result is printed as one "RESULT:" line for robust parsing.
_PARENT_SCRIPT = """
import multiprocessing as mp
import os
import sys


def _env_probe(q):
    # Reads the inherited value WITHOUT importing torch, to isolate
    # environment inheritance from autoload behavior.
    q.put(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))


def _import_probe(q):
    pre = "torch_npu" in sys.modules
    import torch
    post = "torch_npu" in sys.modules
    if post:
        # An explicit import after autoload must be a pure sys.modules hit.
        import torch_npu
        assert torch_npu is sys.modules["torch_npu"]
    q.put(f"{pre},{post}")


def _run_probe(kind, target):
    ctx = mp.get_context(kind)
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    p.join(300)
    assert p.exitcode == 0, f"{kind} probe failed, exitcode={p.exitcode}"
    return q.get(timeout=30)


if __name__ == "__main__":
    for mod in sys.argv[1].split(","):
        __import__(mod)
    print("RESULT:" + str(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")))
    print("RESULT:" + str("torch_npu" in sys.modules))
    print("RESULT:" + str(_run_probe("spawn", _env_probe)))
    print("RESULT:" + str(_run_probe("fork", _env_probe)))
    print("RESULT:" + str(_run_probe("spawn", _import_probe)))
    print("RESULT:" + str(_run_probe("fork", _import_probe)))
"""

# Import orders to verify: 'order' is a comma-separated import sequence.
_ORDERS = (
    "torch",             # import torch only (torch_npu autoloaded)
    "torch_npu",         # import torch_npu only (torch pulled in)
    "torch,torch_npu",   # torch first, explicit torch_npu second
    "torch_npu,torch",   # torch_npu first, explicit torch second
)


class AutoloadEnvVarTest(TestCase):

    @contextlib.contextmanager
    def _child_script(self):
        # Create the temp dir and the script together, and delete the dir
        # in the same scope right after use (G.FIO.04).
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "autoload_env_child.py")
            with open(path, "w") as f:
                f.write(_PARENT_SCRIPT)
            yield path
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _results_after_imports(self, script, order, preset):
        env = {k: v for k, v in os.environ.items() if k != ENV_KEY}
        if preset is not None:
            env[ENV_KEY] = preset
        proc = subprocess.run(
            [sys.executable, script, order],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"order={order!r} preset={preset!r} failed:\n{proc.stderr}",
        )
        lines = [l[len("RESULT:"):] for l in proc.stdout.splitlines()
                 if l.startswith("RESULT:")]
        self.assertEqual(len(lines), 6, f"unexpected output:\n{proc.stdout}")
        return lines

    def _check_all_orders(self, preset):
        expected = preset if preset is not None else "1"
        with self._child_script() as script:
            for order in _ORDERS:
                with self.subTest(order=order, preset=preset):
                    lines = self._results_after_imports(script, order, preset)
                    # Main process: env value, and whether torch_npu was loaded
                    # ('torch' only order relies on autoload, which '0' disables).
                    loaded = not (order == "torch" and preset == "0")
                    self.assertEqual(lines[0], expected)
                    self.assertEqual(lines[1], str(loaded))
                    # mp spawn/fork children inherit the same env value.
                    self.assertEqual(lines[2], expected)
                    self.assertEqual(lines[3], expected)
                    # spawn child (fresh interpreter): 'import torch' autoloads
                    # torch_npu unless inherited env is "0"; explicit import
                    # afterwards is a pure cache hit.
                    self.assertEqual(lines[4], f"False,{preset != '0'}")
                    # fork child: inherits sys.modules, no re-initialization.
                    self.assertEqual(lines[5], f"{loaded},{loaded}")

    # 1-4: default env (unset), the effective value stays "1" for every
    # import order; in particular the temporary "0" set by torch_npu before
    # 'import torch' must not leak.
    def test_env_var_default(self):
        self._check_all_orders(preset=None)

    # 5: user-set value must be preserved for every import order.
    def test_env_var_user_set_0(self):
        self._check_all_orders(preset="0")

    def test_env_var_user_set_1(self):
        self._check_all_orders(preset="1")


if __name__ == "__main__":
    run_tests()
