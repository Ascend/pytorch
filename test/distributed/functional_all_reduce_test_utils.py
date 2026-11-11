"""Small shared fixture for the repository's functional AllReduce tests."""
import os
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase

import torch_npu
from torch_npu.npu.utils import get_soc_version


def assert_close(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def premul_reference(input, factor, group):
    """Exercise the original HCCL allreduce on a caller-owned copy."""
    output = input.detach().clone()
    dist.all_reduce(output, op=torch_npu.distributed._make_hccl_premul_sum(float(factor)), group=group)
    return output


class NpuCollectiveTestCase(MultiProcessTestCase):
    """Use the existing spawn/join/error propagation instead of torchrun."""
    requires_a5 = True

    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device("npu", self.rank)

    def setUp(self):
        if not torch_npu.npu.is_available() or torch_npu.npu.device_count() < self.world_size:
            self.skipTest(f"Requires {self.world_size} NPU devices")
        is_a5 = get_soc_version() >= 260
        if self.requires_a5 != is_a5:
            self.skipTest("Requires A5" if self.requires_a5 else "Requires a pre-A5 device")
        super().setUp()
        self._spawn_processes()

    @contextmanager
    def communication(self, debug_level=dist.DebugLevel.OFF):
        with patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": "CCU_MS"}):
            os.environ.pop("HCCL_DETERMINISTIC", None)
            torch_npu.npu.set_device(self.device)
            previous_debug_level = dist.get_debug_level()
            dist.set_debug_level(debug_level)
            try:
                dist.init_process_group(
                    "hccl", store=dist.FileStore(self.file_name, self.world_size),
                    rank=self.rank, world_size=self.world_size, timeout=timedelta(seconds=120),
                )
                yield dist.group.WORLD
            finally:
                torch.compiler.reset()
                try:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                finally:
                    dist.set_debug_level(previous_debug_level)
