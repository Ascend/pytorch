import os

# Some ops (e.g. aten.matmul_backward) only get their NPU meta registration when
# compatible impl mode is enabled. This env var is read at torch_npu import time,
# so it must be set before importing torch_npu.
os.environ.setdefault("TORCH_NPU_USE_COMPATIBLE_IMPL", "1")

import contextlib
import gc
import math
import re
import sys
import unittest
import warnings
import weakref
from io import StringIO

import torch
import torch.nn as nn
import torch._dynamo.config as dynamo_config
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.utils import run_and_get_code
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils._python_dispatch import TorchDispatchMode

import torch_npu  # noqa: F401
from torch_npu.npu._graph_tree import get_container

TEST_NPU = torch.npu.is_available()
aten = torch.ops.aten

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_num_partitions(code):
    """Get the number of graph partitions from generated code."""
    code = "".join(code)
    found = re.search(r"partitions=\[(.*)\]", code)
    assert found is not None, "Could not find partitions in generated code"
    partitions = found.group(1)
    return len([p for p in partitions.split(",") if p])


class capture_stderr(list):
    """Replace sys.stderr with a temporary StringIO."""

    def __enter__(self):
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------


class TestCase(InductorTestCase):
    device = "npu"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


# ===========================================================================
# Graph Partition Tests — Codegen correctness
# (ported from test_torchinductor.py, device-agnostic via self.device)
# ===========================================================================


@unittest.skipIf(not TEST_NPU, "requires NPU")
class TestGraphPartitionCodegen(TestCase):
    """Tests that graph partition generates correct code and produces correct
    results. These tests do NOT verify npugraph tree state."""

    @config.patch("graph_partition", True)
    def test_graph_partition_refcount(self):
        # Trigger NPU backend registration: compile_fx_inner is a low-level
        # entry that bypasses dynamo's lazy loading of torch_npu._inductor,
        # so without this warmup get_wrapper_codegen_for_device('npu') returns
        # None and init_wrapper_code asserts "Device npu not supported".
        @torch.compile
        def _warmup(x):
            return x + 1
        _warmup(torch.ones(2, device=self.device))

        contexts = [
            contextlib.nullcontext,
            lambda: config.patch({"triton.cudagraphs": True}),
        ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

                def fn(x, y):
                    a = x + y
                    return (a @ a,)

                fn_fx = make_fx(fn)(inps[0], inps[1])
                fn_compiled = compile_fx_inner(fn_fx, inps)

                matmul_seen = False

                class TestRefMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs if kwargs else {}

                        nonlocal inps
                        nonlocal inp_refs
                        nonlocal matmul_seen

                        gc.collect()
                        if func is aten.mm.out:
                            matmul_seen = True
                            assert len(inps) == 0
                            assert inp_refs[0]() is None
                            assert inp_refs[1]() is None

                        return func(*args, **kwargs)

                with TestRefMode():
                    fn_compiled(inps)

                # do an extra run to make sure we are deallocating on warmup and record
                inps.extend(
                    [
                        torch.rand([5, 5]).to(self.device),
                        torch.rand([5, 5]).to(self.device),
                    ]
                )
                inp_refs.extend([weakref.ref(inp) for inp in inps])
                matmul_seen = False

                with TestRefMode():
                    fn_compiled(inps)

                assert len(inps) == 0