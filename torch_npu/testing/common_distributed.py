import unittest
from functools import wraps
from typing import (
    Tuple,
    Dict,
    Any,
)
from collections import namedtuple
import sys
import torch
import torch.distributed as dist
import torch_npu


TestSkip = namedtuple('TestSkip', 'exit_code, message')
TEST_SKIPS = {
    "multi-npu": TestSkip(75, "Need at least 2 ASCEND devices"),
    "multi-npu-1": TestSkip(75, "Need at least 1 ASCEND devices"),
    "multi-npu-2": TestSkip(75, "Need at least 2 ASCEND devices"),
    "multi-npu-3": TestSkip(75, "Need at least 3 ASCEND devices"),
    "multi-npu-4": TestSkip(75, "Need at least 4 ASCEND devices"),
    "multi-npu-5": TestSkip(75, "Need at least 5 ASCEND devices"),
    "multi-npu-6": TestSkip(75, "Need at least 6 ASCEND devices"),
    "multi-npu-7": TestSkip(75, "Need at least 7 ASCEND devices"),
    "multi-npu-8": TestSkip(75, "Need at least 8 ASCEND devices"),
    "hccl":TestSkip(76, "c10d not compiled with HCCL support"),
    "known_issues":TestSkip(77, "Test skipped due to known issues"),
}


def skipIfUnsupportMultiNPU(npu_number_needed):
    def skip_dec(func):
        def wrapper(self):
            if not torch.npu.is_available() or torch.npu.device_count() < npu_number_needed:
                return unittest.SkipTest("Multi-NPU condition not satisfied")
            return func(self)
        return wrapper
    return skip_dec


def with_comms(func):
    if func is None:
        raise RuntimeError("Test function is None.")

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        # if backend not specified, and npu available, then use hccl, else gloo
        if torch.npu.is_available() and torch.npu.device_count() >= self.world_size:
            self.device_type = "npu"
        else:
            self.device_type = "cpu"

        pg_backend = (
            "hccl" if self.device_type == "npu" else "gloo"
        )
        if pg_backend == "hccl" and torch.npu.device_count() < self.world_size:
            raise RuntimeError(TEST_SKIPS[f"multi-npu-{self.world_size}"].message)

        init_pg(backend=pg_backend, world_size=self.world_size, rank=self.rank, file_name=self.file_name)

        torch.npu.manual_seed(0)
        torch.npu.initial_seed()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper


def init_pg(backend: str = "hccl", world_size=1, rank=0, file_name="file://") -> None:
    if backend == "hccl" and torch.npu.device_count() < world_size:
        raise RuntimeError(TEST_SKIPS[f"multi-npu-{world_size}"].message)

    if backend not in ["hccl", "gloo"]:
        raise RuntimeError(f"Backend {backend} not supported!")

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,  # pyre-ignore[16]
        init_method=f"file://{file_name}",  # pyre-ignore[16]
    )

    # set device for hccl pg for collectives
    if backend == "hccl":
        torch.npu.set_device(rank)
