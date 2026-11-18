# Owner(s): ["module: c10d"]
"""NPU symmetric-memory buffer views and device-side signal protocol tests."""

import ctypes
import time
from unittest import skipIf

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the NPU backend
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


def _shmem_available() -> bool:
    try:
        ctypes.CDLL("libshmem.so")
    except OSError:
        return False
    return True


@instantiate_parametrized_tests
@skipIf(not torch.npu.is_available() or torch.npu.device_count() < 2,
        "requires at least two NPUs")
@skipIf(not _shmem_available(), "requires loadable CANN SHMEM runtime (libshmem.so)")
class NPUSymmetricMemoryTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device("npu", self.rank)

    def _init_process(self) -> None:
        torch.npu.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="hccl", world_size=self.world_size, rank=self.rank, store=store
        )
        # NPU allocations require the group to be enabled before rendezvous.
        self.group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(self.group_name)

    def test_barrier(self) -> None:
        self._init_process()
        numel = 64
        tensor = symm_mem.empty(numel, dtype=torch.float32, device=self.device)
        tensor.fill_(self.rank)
        hdl = symm_mem.rendezvous(tensor, group=self.group_name)

        hdl.barrier()
        for peer in range(self.world_size):
            buf = hdl.get_buffer(peer, (numel,), torch.float32)
            self.assertTrue(buf.eq(peer).all())
        hdl.barrier()
        torch.npu.synchronize()

    def test_put_wait_signal(self) -> None:
        self._init_process()
        numel = 64
        tensor = symm_mem.empty(numel, dtype=torch.float32, device=self.device)
        tensor.fill_(-1)
        hdl = symm_mem.rendezvous(tensor, group=self.group_name)
        # The receiver's initialization must land before the remote write.
        hdl.barrier()

        if self.rank == 0:
            hdl.wait_signal(src_rank=1)
            buf = hdl.get_buffer(0, (numel,), torch.float32)
            self.assertTrue(buf.eq(42).all())
        elif self.rank == 1:
            peer_buf = hdl.get_buffer(0, (numel,), torch.float32)
            peer_buf.fill_(42)
            hdl.put_signal(dst_rank=0)
        hdl.barrier()
        torch.npu.synchronize()

    @parametrize("last_channel", [False, True])
    def test_barrier_after_signal(self, last_channel: bool) -> None:
        self._init_process()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        num_channels = hdl.signal_pad_size // (4 * self.world_size)
        channel = num_channels - 1 if last_channel else 0
        timeout_ms = 15000

        signal_warm, barrier_warm = [c for c in range(num_channels) if c != channel][:2]
        if self.rank == 0:
            hdl.put_signal(dst_rank=1, channel=signal_warm, timeout_ms=timeout_ms)
        elif self.rank == 1:
            hdl.wait_signal(src_rank=0, channel=signal_warm, timeout_ms=timeout_ms)
        hdl.barrier(channel=barrier_warm, timeout_ms=timeout_ms)
        torch.npu.synchronize()

        if self.rank == 0:
            hdl.put_signal(dst_rank=1, channel=channel, timeout_ms=timeout_ms)
            torch.npu.synchronize()
        dist.barrier()
        if self.rank == 1:
            time.sleep(1.0)
            hdl.wait_signal(src_rank=0, channel=channel, timeout_ms=timeout_ms)
        hdl.barrier(channel=channel, timeout_ms=timeout_ms)
        torch.npu.synchronize()

    def test_barrier_channel_out_of_bounds(self) -> None:
        self._init_process()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        num_slots = hdl.signal_pad_size // 4
        max_channel = num_slots // self.world_size

        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            hdl.barrier(channel=max_channel)
        torch.npu.synchronize()
        if max_channel > 1:
            hdl.barrier(channel=max_channel - 1)
        torch.npu.synchronize()

    def test_put_wait_signal_channel_out_of_bounds(self) -> None:
        self._init_process()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        num_slots = hdl.signal_pad_size // 4
        max_channel = num_slots // self.world_size
        peer = (self.rank + 1) % self.world_size

        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            hdl.put_signal(dst_rank=peer, channel=max_channel)
        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            hdl.wait_signal(src_rank=peer, channel=max_channel)
        src = (self.rank - 1) % self.world_size
        hdl.put_signal(dst_rank=peer, channel=max_channel - 1)
        hdl.wait_signal(src_rank=src, channel=max_channel - 1)
        torch.npu.synchronize()

    def test_put_wait_signal_rank_out_of_bounds(self) -> None:
        self._init_process()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        for bad_rank in (-1, self.world_size):
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                hdl.put_signal(dst_rank=bad_rank)
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                hdl.wait_signal(src_rank=bad_rank)
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                hdl.get_signal_pad(bad_rank)
        torch.npu.synchronize()

    def test_get_buffer_storage_offset(self) -> None:
        self._init_process()
        tensor = symm_mem.empty(16, dtype=torch.int32, device=self.device)
        tensor.copy_(torch.arange(16, dtype=torch.int32, device=self.device) + self.rank * 100)
        hdl = symm_mem.rendezvous(tensor, group=self.group_name)
        hdl.barrier()
        peer = (self.rank + 1) % self.world_size
        local = hdl.get_buffer(self.rank, (16,), torch.int32)
        remote = hdl.get_buffer(peer, (16,), torch.int32)
        self.assertEqual(hdl.buffer_size, tensor.numel() * tensor.element_size())
        self.assertEqual(local.data_ptr(), tensor.data_ptr())
        self.assertEqual(remote.data_ptr(), hdl.buffer_ptrs[peer])
        self.assertEqual(local.storage_offset(), 0)
        self.assertEqual(local.untyped_storage().size(), hdl.buffer_size)
        torch.testing.assert_close(local, tensor)
        torch.testing.assert_close(
            remote, torch.arange(16, dtype=torch.int32, device=self.device) + peer * 100
        )
        # storage_offset is in elements, not bytes.
        remote_slice = hdl.get_buffer(peer, (4,), torch.int32, 4)
        self.assertEqual(remote_slice.data_ptr(), hdl.buffer_ptrs[peer] + 4 * tensor.element_size())
        torch.testing.assert_close(
            remote_slice, torch.arange(4, 8, dtype=torch.int32, device=self.device) + peer * 100
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds the allocated size"):
            hdl.get_buffer(peer, (4,), torch.int32, 14)
        hdl.barrier()
        torch.npu.synchronize()

    def _check_signal_pad_views(self, hdl) -> None:
        peer_rank = (self.rank + 1) % self.world_size
        signal_pad = hdl.get_signal_pad(self.rank)
        self.assertEqual(signal_pad.data_ptr(), hdl.signal_pad_ptrs[hdl.rank])
        signal_pad = hdl.get_signal_pad(peer_rank)
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), hdl.signal_pad_size // 4)
        signal_pad = hdl.get_signal_pad(peer_rank, (8, 8))
        self.assertEqual(signal_pad.dtype, torch.uint32)
        self.assertEqual(signal_pad.numel(), 64)
        signal_pad = hdl.get_signal_pad(peer_rank, dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), hdl.signal_pad_size // 8)
        signal_pad = hdl.get_signal_pad(peer_rank, (8, 8), dtype=torch.uint64)
        self.assertEqual(signal_pad.dtype, torch.uint64)
        self.assertEqual(signal_pad.numel(), 64)

    def test_get_signal_pad(self) -> None:
        self._init_process()
        t = symm_mem.empty(1, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        self._check_signal_pad_views(hdl)

        t = symm_mem.empty(1, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        signal_pad = hdl.get_signal_pad(self.rank)
        signal_pad.fill_(42)
        t.fill_(0)
        self.assertTrue(signal_pad.eq(42).all())
        torch.npu.synchronize()
        # Do not run a handle barrier on a deliberately overwritten signal pad.
        dist.barrier()

    def test_get_signal_pad_views_read_only(self) -> None:
        self._init_process()
        t = symm_mem.empty(1, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        self._check_signal_pad_views(hdl)
        torch.npu.synchronize()
        dist.barrier()

    def test_signal_pad_slot_layout(self) -> None:
        self._init_process()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=self.group_name)
        channel = 3
        src = (self.rank - 1) % self.world_size
        hdl.put_signal(dst_rank=(self.rank + 1) % self.world_size, channel=channel)
        torch.npu.synchronize()
        dist.barrier()
        pad = hdl.get_signal_pad(self.rank).view(torch.int32)
        set_words = pad.nonzero().flatten().tolist()
        hdl.wait_signal(src_rank=src, channel=channel)
        torch.npu.synchronize()
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, set_words)
        expected = [
            [self.world_size * channel + (r - 1) % self.world_size]
            for r in range(self.world_size)
        ]
        self.assertEqual(gathered, expected)
        self.assertEqual(pad.nonzero().numel(), 0)

    def test_put_wait_signal_repeated_non_default_stream(self) -> None:
        self._init_process()
        tensor = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        tensor.fill_(-1)
        hdl = symm_mem.rendezvous(tensor, group=self.group_name)
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            hdl.barrier(timeout_ms=15000)
            buf = hdl.get_buffer(0, (64,), torch.float32)
            for value in range(42, 50):
                if self.rank == 1:
                    buf.fill_(value)
                    hdl.put_signal(dst_rank=0, channel=1, timeout_ms=15000)
                    # A consumed notification alone does not prove the payload
                    # has been read. Wait for an ACK before overwriting it.
                    hdl.wait_signal(src_rank=0, channel=2, timeout_ms=15000)
                else:
                    hdl.wait_signal(src_rank=1, channel=1, timeout_ms=15000)
                    self.assertTrue(buf.eq(value).all())
                    hdl.put_signal(dst_rank=1, channel=2, timeout_ms=15000)
            hdl.barrier(timeout_ms=15000)
        stream.synchronize()


if __name__ == "__main__":
    run_tests()
