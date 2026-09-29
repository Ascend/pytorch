# Copyright (c) 2026, Huawei Technologies Co., Ltd
# LICENSE: BSD-3-Clause (same as the repository)

"""Regression tests for import-time device queries in torch_npu._inductor.

`import torch_npu._inductor` runs inside Inductor forked compile workers
(the generated kernel modules import triton_experimental.npu_triton_
heuristics).  Device initialization there raises

    RuntimeError: Cannot re-initialize NPU in forked subprocess

and kills every >=2-kernel compilation, so the import path must stay free
of _lazy_init().  These tests pin the three contracts of the fix:

1. importing torch_npu._inductor performs no device initialization;
2. the C-level soc query is a driver-level read-only lookup that returns
   the same value from an uninitialized process, an initialized process,
   and a forked child of an initialized one;
3. the lazily-resolved config/heuristics values resolve to the same
   numbers the eager code produced, and the lazy accessors cache.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_npu  # noqa: F401


def _run_py(code: str, backend: str = None) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter so process-global NPU state from the
    pytest host cannot leak into (or invalidate) the assertions."""
    env = dict(os.environ)
    if backend is None:
        env.pop("TORCHINDUCTOR_NPU_BACKEND", None)
    else:
        env["TORCHINDUCTOR_NPU_BACKEND"] = backend
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=420,
        env=env,
    )


def test_import_inductor_does_not_initialize_npu():
    # Scoped to the triton_experimental chain: its loader (re-)runs inside
    # forked compile workers, so its import must stay free of device
    # initialization.  Other backends are out of scope (the default
    # backend's import chain has its own, unrelated eager init, and its
    # kernel modules never import this chain inside workers).
    code = (
        "import torch, torch_npu, torch_npu._inductor\n"
        "assert not torch.npu.is_initialized(), "
        "'importing torch_npu._inductor initialized the device'\n"
    )
    r = _run_py(code, backend="triton_experimental")
    assert r.returncode == 0, r.stderr


def test_soc_query_stable_across_initialization():
    code = (
        "import torch, torch_npu\n"
        "before = torch_npu._C._npu_get_soc_version()\n"
        "_ = (torch.randn(8, 8, device='npu') @ torch.randn(8, 8, device='npu')).sum()\n"
        "torch.npu.synchronize()\n"
        "after = torch_npu._C._npu_get_soc_version()\n"
        "assert before == after, (before, after)\n"
    )
    r = _run_py(code)
    assert r.returncode == 0, r.stderr


def test_soc_query_safe_in_forked_child():
    # A fork child of an NPU-initialized process is exactly the state of an
    # Inductor compile worker: _lazy_init() there raises, the raw C query
    # must not.
    code = (
        "import os, torch, torch_npu\n"
        "_ = (torch.randn(8, 8, device='npu') @ torch.randn(8, 8, device='npu')).sum()\n"
        "torch.npu.synchronize()\n"
        "parent_soc = torch_npu._C._npu_get_soc_version()\n"
        "r, w = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    try:\n"
        "        child_soc = torch_npu._C._npu_get_soc_version()\n"
        "        os.write(w, ('OK %d' % child_soc).encode())\n"
        "    except BaseException as e:  # noqa: B036\n"
        "        os.write(w, ('FAIL %s: %s' % (type(e).__name__, e)).encode()[:200])\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "msg = os.read(r, 256).decode()\n"
        "assert msg == 'OK %d' % parent_soc, msg\n"
    )
    r = _run_py(code)
    assert r.returncode == 0, r.stderr


def test_config_lazy_attrs_resolve_and_cache():
    from torch_npu._inductor import config as cfg
    from torch_npu._inductor.config import _get_core_nums

    vec = cfg.num_vector_core
    assert isinstance(vec, int) and vec > 0
    # Same object identity for the tuple members across attribute accesses
    # (single cached resolution).
    assert cfg.num_vector_core == vec
    assert cfg.num_cube_core > 0
    assert cfg.prop is not None
    assert _get_core_nums.cache_info().currsize == 1


def test_config_unknown_attr_raises():
    from torch_npu._inductor import config as cfg

    with pytest.raises(AttributeError):
        _ = cfg.definitely_not_an_attribute


def test_device_props_soc_matches_public_api():
    if not torch.npu.is_initialized():
        torch.npu.init()
    from torch_npu._inductor.triton_experimental import device_props as dp
    from torch_npu.npu._backends import get_soc_version

    assert dp._soc_version_no_init() == get_soc_version()


def test_heuristics_total_cores_lazy_and_consistent():
    from torch_npu._inductor.triton_experimental import (
        device_props,
        npu_triton_heuristics as h,
    )

    total = h._npu_total_cores()
    assert isinstance(total, int) and total > 0
    assert total == device_props.get_npu_vector_core_count()
    assert h._npu_total_cores() == total


def test_compile_in_forked_workers_after_eager_op():
    """End-to-end regression: with the default (fork-pool) parallel compile,
    an eager-first flow with a >=2-kernel graph must compile and match.
    The model is rebuilt inline (identical to the repro that validated the
    fix) because a -c snippet cannot reference host test objects."""
    code = (
        "import torch, torch_npu\n"
        "a = (torch.randn(64, 512, device='npu') @ torch.randn(512, 512, device='npu')).relu()\n"
        "torch.npu.synchronize()\n"
        "class M(torch.nn.Module):\n"
        "    def forward(self, a):\n"
        "        idx = torch.arange(a.shape[0], device=a.device)\n"
        "        y = a[idx] * 1.0001 + 0.5\n"
        "        return (y.sum(dim=1) * torch.arange(y.shape[1], device=y.device).sum()).relu()\n"
        "m = M().npu()\n"
        "x = torch.randn(64, 512, device='npu')\n"
        "ref = m(x)\n"
        "c = torch.compile(m, options={'npu_backend': 'triton_experimental'})\n"
        "out = c(x)\n"
        "assert torch.allclose(ref, out, rtol=1e-4, atol=1e-4), (ref, out)\n"
    )
    r = _run_py(code)
    assert r.returncode == 0, r.stderr
