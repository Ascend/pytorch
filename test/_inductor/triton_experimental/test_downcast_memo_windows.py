# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Boundary-downcast memo behavior across NPU graph-tree windows.

The triton_experimental launcher memoizes i64/fp64 -> i32/fp32 boundary
casts in a module-level dictionary keyed by the source tensor. A memo
entry created while the thread is diverted to an ACLGraph private pool
(warmup or capture) pins a pool block that no recorded graph accounts
for, which breaks ``check_memory_pool()``. These tests pin the guard:
ordinary execution still dedups, capture and graph-tree warmup do not.
"""

import gc
import os
import unittest
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch._inductor import config

from torch_npu._inductor.triton_experimental import config as ncfg, npu_triton_heuristics


class TestDowncastMemoWindows(unittest.TestCase):
    def setUp(self):
        super().setUp()
        npu_triton_heuristics._downcast_memo.clear()

    def tearDown(self):
        npu_triton_heuristics._downcast_memo.clear()
        super().tearDown()

    def test_downcast_is_memoized_without_cudagraphs(self):
        src = torch.tensor([7], dtype=torch.int64, device="npu")

        with (
            config.patch(
                {"triton.cudagraphs": False, "triton.cudagraph_trees": True}
            ),
            ncfg.patch(dedup_downcast=True),
            mock.patch.object(
                npu_triton_heuristics.torch_npu.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            mock.patch.object(
                npu_triton_heuristics,
                "_is_npugraph_tree_warmup_active",
                return_value=False,
            ),
        ):
            first = npu_triton_heuristics._memoized_downcast(src, torch.int32)
            second = npu_triton_heuristics._memoized_downcast(src, torch.int32)

        self.assertIs(first, second)
        self.assertIn(id(src), npu_triton_heuristics._downcast_memo)

        src_id = id(src)
        del src
        gc.collect()
        self.assertNotIn(src_id, npu_triton_heuristics._downcast_memo)

    def test_downcast_is_not_memoized_during_npugraph_tree_capture(self):
        src = torch.tensor([7], dtype=torch.int64, device="npu")
        with (
            config.patch(
                {"triton.cudagraphs": False, "triton.cudagraph_trees": True}
            ),
            ncfg.patch(dedup_downcast=True),
            mock.patch.object(
                npu_triton_heuristics.torch_npu.npu,
                "is_current_stream_capturing",
                return_value=True,
            ),
            mock.patch.object(
                npu_triton_heuristics,
                "_is_npugraph_tree_warmup_active",
                return_value=False,
            ),
        ):
            first = npu_triton_heuristics._memoized_downcast(src, torch.int32)
            second = npu_triton_heuristics._memoized_downcast(src, torch.int32)

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(npu_triton_heuristics._downcast_memo, {})

    def test_downcast_is_not_memoized_during_npugraph_tree_warmup(self):
        src = torch.tensor([7], dtype=torch.int64, device="npu")
        with (
            config.patch(
                {"triton.cudagraphs": False, "triton.cudagraph_trees": True}
            ),
            ncfg.patch(dedup_downcast=True),
            mock.patch.object(
                npu_triton_heuristics.torch_npu.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            mock.patch.object(
                npu_triton_heuristics,
                "_is_npugraph_tree_warmup_active",
                return_value=True,
            ),
        ):
            first = npu_triton_heuristics._memoized_downcast(src, torch.int32)
            second = npu_triton_heuristics._memoized_downcast(src, torch.int32)

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(npu_triton_heuristics._downcast_memo, {})


if __name__ == "__main__":
    unittest.main()
