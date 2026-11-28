"""Shared entry points for opt-in distributed API tests."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ENABLED = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
if ENABLED and getattr(torch, "npu", None) is not None and torch.npu.is_available():
    from torch_npu.testing import testcase

    TestCase = testcase.TestCase
    run_tests = testcase.run_tests
else:
    from unittest import TestCase

    run_tests = unittest.main


@unittest.skipUnless(ENABLED, "Set RUN_SUPPLEMENT_API_TESTS=1 for full regression")
class SupplementTestCase(TestCase):
    pass


def _run_rank(rank, callback, device_type, rendezvous, skip_reasons=None):
    device = torch.device(device_type, rank)
    torch.get_device_module(device).set_device(device)
    if rendezvous == "env://":
        store, _, _ = next(dist.rendezvous(rendezvous))
        options = {"store": dist.PrefixStore(callback.__module__ + "." + callback.__name__, store)}
    else:
        options = {"init_method": rendezvous}
    dist.init_process_group(
        "hccl" if device_type == "npu" else "nccl",
        rank=rank,
        world_size=2,
        **options,
        timeout=timedelta(seconds=90),
    )
    try:
        callback(unittest.TestCase(), rank, device)
    except unittest.SkipTest as error:
        if skip_reasons is None:
            raise
        skip_reasons.put(str(error))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TwoRankTestCase(SupplementTestCase):
    def run_on_two_ranks(self, callback, *, shmem_ops=()):
        device = torch.accelerator.current_accelerator()
        if device is None or torch.accelerator.device_count() < 2:
            self.skipTest("Two accelerator devices are required")
        if device.type not in ("npu", "cuda"):
            self.skipTest(f"No collective backend configured for {device.type}")
        key = "PrivateUse1" if device.type == "npu" else "CUDA"
        for name in shmem_ops:
            op = f"symm_mem::{name}"
            # A generic CPU fallback cannot perform remote accelerator memory access.
            if not any(
                torch._C._dispatch_has_kernel_for_dispatch_key(op, dispatch_key)
                for dispatch_key in (key, "CompositeExplicitAutograd", "CompositeImplicitAutograd")
            ):
                self.skipTest(f"{op} has no kernel for {key}")
        if "TORCHELASTIC_RUN_ID" in os.environ:
            if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "2":
                self.skipTest("The external launcher must provide two ranks on one host")
            _run_rank(int(os.environ["LOCAL_RANK"]), callback, device.type, "env://")
            return
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory).resolve() / "store").as_uri()
            skip_reasons = mp.get_context("spawn").SimpleQueue()
            try:
                mp.spawn(
                    _run_rank, args=(callback, device.type, rendezvous, skip_reasons), nprocs=2
                )
                if not skip_reasons.empty():
                    self.skipTest(skip_reasons.get())
            finally:
                skip_reasons.close()


def sharded_linear(device):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    mesh = DeviceMesh(device.type, [0, 1])
    model = torch.nn.Linear(4, 4, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1)
    fully_shard(model, mesh=mesh)
    return model


def backward_on_rank(model, rank, device):
    model.zero_grad(set_to_none=True)
    local_input = torch.full((1, 4), float(rank + 1), device=device)
    model(local_input).sum().backward()


def check_reduced_gradient(model, expected=1.5):
    local_grad = model.weight.grad.to_local()
    torch.testing.assert_close(local_grad, torch.full_like(local_grad, expected))


def init_symmetric_memory(case, device, *, nvshmem=True, multicast=False):
    import ctypes

    from torch.distributed import _symmetric_memory as symm_mem

    if device.type == "cuda":
        if not torch.cuda.can_device_access_peer(0, 1):
            case.skipTest("CUDA peer access is required")
        if nvshmem:
            if not symm_mem.is_nvshmem_available():
                case.skipTest("The CUDA build lacks NVSHMEM")
            symm_mem.set_backend("NVSHMEM")
        if multicast:
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
                case.skipTest("CUDA multicast support is required")
    else:
        try:
            ctypes.CDLL("libshmem.so", mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            case.skipTest(f"CANN SHMEM runtime unavailable: {error}")
    return dist.group.WORLD.group_name
