import ctypes
from unittest import skipIf

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

# acl binding for device->host readback of buffers_dev_/signal_pads_dev_
import acl

ACL_MEMCPY_DEVICE_TO_HOST = 2

# So that tests are written in device-agnostic way
device_type = "npu"
device_module = torch.get_device_module(device_type)


def _shmem_available() -> bool:
    """Return whether the NPU and CANN SHMEM runtime are available."""
    if not torch.npu.is_available():
        return False
    try:
        ctypes.CDLL("libshmem.so", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


# Highest heap byte carved by the ptp tests in this process: stale cache lines
# below it shadow RMA reads, so later receive buffers carve above this mark.
_heap_high_water = 0


def _record_heap_high_water(*tensors) -> None:
    global _heap_high_water
    for tensor in tensors:
        _heap_high_water = max(
            _heap_high_water, tensor.data_ptr() + tensor.numel() * tensor.element_size()
        )


# Chunk size test_shmem_quiet_batch uses to wedge the reused low heap region;
# must stay smaller than that test's smallest receive buffer (40MB).
_WEDGE_CHUNK_BYTES = 32 * 1024 * 1024


@instantiate_parametrized_tests
@skipIf(not _shmem_available(), "CANN SHMEM is not available")
class NPUSHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # Testing with HCCL backend
        return "hccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.npu.device_count()
        cls.device = torch.device(f"npu:{dev_id}")

    def _init_device(self) -> None:
        device_module.set_device(self.device)
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _make_get_buffer_fixture(self):
        """Create one two-rank symmetric allocation with distinguishable values."""
        rank = self.rank
        peer = 1 - rank
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # rank 0 保存 [0..15]，rank 1 保存 [100..115]。
        tensor = symm_mem.empty(16, dtype=torch.int32, device=self.device)
        tensor.copy_(torch.arange(16, dtype=torch.int32, device=self.device) + rank * 100)
        handle = symm_mem.rendezvous(tensor, group=group_name)
        dist.barrier(device_ids=[rank])
        return rank, peer, tensor, handle

    def _read_dev_ptr_array(self, dev_ptr: int, n: int) -> list:
        """Read back a device array (void** of len n) to a host list of ints via
        acl.rt.memcpy. Used to verify buffer_ptrs_dev / signal_pad_ptrs_dev point
        to a real, initialized device buffer whose content matches the host arrays."""
        self.assertNotEqual(dev_ptr, 0, "device pointer must not be null/0x0")
        arr = np.zeros(n, dtype=np.uint64)
        nbytes = arr.nbytes
        dst = arr.ctypes.data
        ret = acl.rt.memcpy(dst, nbytes, dev_ptr, nbytes, ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == 0, f"acl.rt.memcpy failed ret={ret}"
        torch.npu.current_stream().synchronize()
        return [int(v) for v in arr]

    def _make_ptp_fixture(self, dtype, numel, value):
        """Create one two-rank symmetric allocation filled with an identifiable value."""
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(value)
        symm_mem.rendezvous(tensor, group=group_name)
        return tensor

    @skipIfUnsupportMultiNPU(2)
    @parametrize("use_nbi", [False, True])
    def test_shmem_ptp_one_directional(self, use_nbi: bool) -> None:
        """Deterministic put/get content check on never-written receive buffers;
        the nbi variant closes a mixed put+get batch with a single quiet."""
        self._init_device()

        rank = self.rank
        peer = 1 - rank
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Larger than every earlier allocation: the first-fit heap carves from
        # the untouched tail, so no reused-address cache lines shadow the reads.
        f_numel = 16 * 1024 * 1024  # float32, 64MB
        i_numel = 24 * 1024 * 1024  # int32, 96MB (distinct size: no reuse)
        f_tensor = symm_mem.empty(f_numel, dtype=torch.float, device=self.device)
        i_tensor = symm_mem.empty(i_numel, dtype=torch.int32, device=self.device)
        symm_mem.rendezvous(f_tensor, group=group_name)
        symm_mem.rendezvous(i_tensor, group=group_name)
        _record_heap_high_water(f_tensor, i_tensor)

        # rank 0 fills f (put source), rank 1 fills i (get source): each rank
        # keeps one never-written buffer to receive the peer's RMA write.
        if rank == 0:
            f_tensor.fill_(3.25)
        else:
            i_tensor.fill_(-77)
        dist.barrier(device_ids=[rank])

        if rank == 0:
            put_op = (
                torch.ops.symm_mem.npushmem_put_nbi
                if use_nbi
                else torch.ops.symm_mem.npushmem_put
            )
            get_op = (
                torch.ops.symm_mem.npushmem_get_nbi
                if use_nbi
                else torch.ops.symm_mem.npushmem_get
            )
            put_op(f_tensor, peer)
            get_op(i_tensor, peer)
            if use_nbi:
                # Completion rests on quiet alone: a synchronize here would
                # wait for the transfers and mask the guarantee it provides.
                torch.ops.symm_mem.npushmem_quiet(f_tensor)
            else:
                # Blocking ops are stream-ordered; the quiet has no pending nbi.
                torch.npu.synchronize()
                torch.ops.symm_mem.npushmem_quiet(f_tensor)
        dist.barrier(device_ids=[rank])

        # Each side asserts the peer-landed buffer (cache miss, HBM truth) and
        # its own filled source; a broken transfer fails on the peer's read.
        torch.testing.assert_close(
            f_tensor, torch.full((f_numel,), 3.25, dtype=torch.float, device=self.device)
        )
        torch.testing.assert_close(
            i_tensor, torch.full((i_numel,), -77, dtype=torch.int32, device=self.device)
        )
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_quiet_batch(self) -> None:
        """Per-buffer content check for the multi-op nbi + one-quiet model:
        four distinct buffers per direction, each asserted by the receiver."""
        dtype = torch.float
        self._init_device()

        rank = self.rank
        peer = 1 - rank
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Fixed-size chunks wedge the reused low heap region so the buffers below
        # carve from the never-written tail (mechanism: _WEDGE_CHUNK_BYTES).
        wedge_chunks = []
        while True:
            chunk = symm_mem.empty(_WEDGE_CHUNK_BYTES, dtype=torch.uint8, device=self.device)
            wedge_chunks.append(chunk)
            if chunk.data_ptr() >= _heap_high_water:
                break
        numels = (10, 12, 14, 16)  # million elements: 40/48/56/64MB, distinct
        a_tensors = [
            symm_mem.empty(n * 1024 * 1024, dtype=dtype, device=self.device)
            for n in numels
        ]
        b_tensors = [
            symm_mem.empty(n * 1024 * 1024, dtype=dtype, device=self.device)
            for n in numels
        ]
        for tensor in a_tensors + b_tensors:
            symm_mem.rendezvous(tensor, group=group_name)
        # The wedge must have pushed every buffer above the reused region.
        self.assertGreaterEqual(a_tensors[0].data_ptr(), _heap_high_water)
        self.assertGreaterEqual(b_tensors[0].data_ptr(), _heap_high_water)

        # rank 0 fills the A set, rank 1 the B set: every read buffer was never
        # written by its reader, and values are distinct per buffer/direction.
        if rank == 0:
            for i, tensor in enumerate(a_tensors):
                tensor.fill_(1000 + i)
        else:
            for i, tensor in enumerate(b_tensors):
                tensor.fill_(3000 + i)
        dist.barrier(device_ids=[rank])

        # Put batch: four submissions, one quiet, receiver asserts each buffer.
        if rank == 0:
            for tensor in a_tensors:
                torch.ops.symm_mem.npushmem_put_nbi(tensor, peer)
            torch.ops.symm_mem.npushmem_quiet(a_tensors[0])
        dist.barrier(device_ids=[rank])
        if rank == 1:
            for i, tensor in enumerate(a_tensors):
                torch.testing.assert_close(
                    tensor,
                    torch.full((tensor.numel(),), 1000 + i, dtype=dtype, device=self.device),
                )
        dist.barrier(device_ids=[rank])

        # Get batch: four pulls, one quiet, puller asserts each buffer.
        if rank == 0:
            for tensor in b_tensors:
                torch.ops.symm_mem.npushmem_get_nbi(tensor, peer)
            torch.ops.symm_mem.npushmem_quiet(b_tensors[0])
        dist.barrier(device_ids=[rank])
        if rank == 0:
            for i, tensor in enumerate(b_tensors):
                torch.testing.assert_close(
                    tensor,
                    torch.full((tensor.numel(),), 3000 + i, dtype=dtype, device=self.device),
                )
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    @parametrize("op_name", ["npushmem_put", "npushmem_get"])
    def test_negative_peer(self, op_name: str) -> None:
        self._init_device()

        tensor = self._make_ptp_fixture(torch.int32, 16, self.rank)
        # rendezvous first so both ranks hit the checks symmetrically.
        dist.barrier(device_ids=[self.rank])
        op = getattr(torch.ops.symm_mem, op_name)
        # peer validation is a host-side TORCH_CHECK before any submission.
        with self.assertRaisesRegex(RuntimeError, "peer must be in range"):
            op(tensor, -1)
        with self.assertRaisesRegex(RuntimeError, "peer must be in range"):
            op(tensor, 2)  # world_size == 2
        dist.barrier(device_ids=[self.rank])

    @skipIfUnsupportMultiNPU(2)
    @parametrize("op_name", ["npushmem_put", "npushmem_get"])
    def test_negative_contiguous(self, op_name: str) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        tensor = symm_mem.empty((4, 4), dtype=torch.int32, device=self.device)
        symm_mem.rendezvous(tensor, group=group_name)
        non_contig = tensor.t()

        op = getattr(torch.ops.symm_mem, op_name)
        # put- and get-family error messages are asserted separately.
        op_label = "get op" if "get" in op_name else "put op"
        with self.assertRaisesRegex(
            RuntimeError, f"{op_label} currently supports contiguous"
        ):
            op(non_contig, 1 - self.rank)
        dist.barrier(device_ids=[self.rank])

    @skipIfUnsupportMultiNPU(2)
    def test_alloc(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skipIfUnsupportMultiNPU(2)
    def test_alloc_free(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)
        del out

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_copy(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)
        self.assertEqual(shmem_tensor, tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_matmul(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)
        tensor1 = torch.randn(shape, dtype=dtype, device=self.device)

        matmul = torch.matmul(tensor, tensor1)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)

        shmem_matmul = torch.matmul(shmem_tensor, tensor1)
        self.assertEqual(shmem_matmul, matmul)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_matmul1(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        shape = (512, 512)

        tensor = torch.randn(shape, dtype=dtype, device=self.device)
        tensor1 = torch.randn(shape, dtype=dtype, device=self.device)

        matmul = torch.matmul(tensor, tensor1)

        shmem_tensor = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor.copy_(tensor)
        shmem_tensor1 = symm_mem.empty(shape, dtype=dtype, device=self.device)
        shmem_tensor1.copy_(tensor1)

        shmem_matmul = torch.matmul(shmem_tensor, shmem_tensor1)
        self.assertEqual(shmem_matmul, matmul)

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_put(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_put(tensor, 1)
            dist.barrier(device_ids=[self.rank])
        elif self.rank == 1:
            dist.barrier(device_ids=[self.rank])
            torch.testing.assert_close(
                tensor, torch.zeros(numel, dtype=dtype, device=self.device)
            )

    @skipIfUnsupportMultiNPU(2)
    def test_shmem_get(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_get(tensor, 1)
            dist.barrier(device_ids=[self.rank])
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        elif self.rank == 1:
            dist.barrier(device_ids=[self.rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer(self) -> None:
        """Validate mapped buffers, explicit offsets and bounds checking."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()

        self.assertEqual(handle.buffer_size, tensor.numel() * tensor.element_size())

        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)
        self.assertEqual(local.data_ptr(), tensor.data_ptr())
        self.assertEqual(remote.data_ptr(), handle.buffer_ptrs[peer])
        self.assertEqual(local.untyped_storage().size(), handle.buffer_size)
        torch.testing.assert_close(local, tensor)
        torch.testing.assert_close(
            remote, torch.arange(16, dtype=torch.int32, device=self.device) + peer * 100
        )

        # storage_offset=4 表示跳过 4 个 int32，即 16 字节。
        remote_slice = handle.get_buffer(peer, (4,), torch.int32, 4)
        self.assertEqual(
            remote_slice.data_ptr(), handle.buffer_ptrs[peer] + 4 * tensor.element_size()
        )
        torch.testing.assert_close(
            remote_slice, torch.arange(4, 8, dtype=torch.int32, device=self.device) + peer * 100
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds the allocated size"):
            handle.get_buffer(peer, (4,), torch.int32, 14)
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer_alias(self) -> None:
        """Validate local aliasing and writes through a peer mapping."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()
        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)

        local[3:4].fill_(1000 + rank)
        torch.npu.synchronize()
        self.assertEqual(tensor[3].item(), 1000 + rank)
        dist.barrier(device_ids=[rank])

        if rank == 0:
            remote[7:8].fill_(12345)
            torch.npu.synchronize()
        dist.barrier(device_ids=[rank])
        if rank == 1:
            self.assertEqual(tensor[7].item(), 12345)
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_get_buffer_allocation_lifetime(self) -> None:
        """Validate that a live handle keeps the SHMEM allocation alive."""
        self._init_device()
        rank, peer, tensor, handle = self._make_get_buffer_fixture()
        local_ptr = handle.buffer_ptrs[rank]
        del tensor
        dist.barrier(device_ids=[rank])

        local = handle.get_buffer(rank, (16,), torch.int32)
        remote = handle.get_buffer(peer, (16,), torch.int32)
        self.assertEqual(local.data_ptr(), local_ptr)
        torch.testing.assert_close(
            local, torch.arange(16, dtype=torch.int32, device=self.device) + rank * 100
        )
        torch.testing.assert_close(
            remote, torch.arange(16, dtype=torch.int32, device=self.device) + peer * 100
        )
        dist.barrier(device_ids=[rank])

    @skipIfUnsupportMultiNPU(2)
    def test_dev_ptr_arrays_valid(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        dtype = torch.float
        numel = 1024

        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)

        host_buf_ptrs = [int(x) for x in hdl.buffer_ptrs]
        host_sp_ptrs = [int(x) for x in hdl.signal_pad_ptrs]
        self.assertEqual(len(host_buf_ptrs), self.world_size)
        self.assertEqual(len(host_sp_ptrs), self.world_size)

        dev_buf_ptrs = self._read_dev_ptr_array(int(hdl.buffer_ptrs_dev), self.world_size)
        dev_sp_ptrs = self._read_dev_ptr_array(int(hdl.signal_pad_ptrs_dev), self.world_size)

        self.assertEqual(dev_buf_ptrs, host_buf_ptrs, "buffers_dev_ content must match host buffers_")
        self.assertEqual(dev_sp_ptrs, host_sp_ptrs, "signal_pads_dev_ content must match host signal_pads_")

    @skipIfUnsupportMultiNPU(2)
    @parametrize("numel", [1, 17, 129, 4096])
    def test_signal_pad_offset_consistency(self, numel: int) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        # Byte-sized buffers cover allocations not aligned to 16 or 128 bytes.
        dtype = torch.uint8
        signal_pad_size = symm_mem.get_signal_pad_size()
        buffer_offset = (signal_pad_size + 127) // 128 * 128

        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)

        buf_ptrs = [int(x) for x in hdl.buffer_ptrs]
        sp_ptrs = [int(x) for x in hdl.signal_pad_ptrs]
        self.assertEqual(len(buf_ptrs), self.world_size)
        self.assertEqual(len(sp_ptrs), self.world_size)

        # The pad occupies a fixed prefix, independent of the requested buffer size.
        offsets = {buf_ptrs[r] - sp_ptrs[r] for r in range(self.world_size)}
        self.assertEqual(offsets, {buffer_offset})
        self.assertGreater(buffer_offset, 0, "signal pad must sit before the data region")
        self.assertEqual(buffer_offset % 128, 0, "buffer_offset must be 128-byte aligned")
        self.assertEqual(buf_ptrs[hdl.rank], t.data_ptr())
        for buffer_ptr in buf_ptrs:
            self.assertEqual(buffer_ptr % 16, 0, "data buffers must remain 16-byte aligned")

        signal_pad = hdl.get_signal_pad(hdl.rank, dtype=torch.uint8)
        self.assertEqual(signal_pad.data_ptr(), sp_ptrs[hdl.rank])
        self.assertEqual(signal_pad.numel(), signal_pad_size)
        torch.testing.assert_close(signal_pad, torch.zeros_like(signal_pad))

        # Buffer views expose only the user data, not the prefix or allocation padding.
        self.assertEqual(hdl.buffer_size, t.numel() * t.element_size())
        self.assertEqual(hdl.offset, 0)
        local = hdl.get_buffer(hdl.rank, (numel,), dtype)
        self.assertEqual(local.data_ptr(), t.data_ptr())
        self.assertEqual(local.untyped_storage().size(), numel)
        with self.assertRaisesRegex(RuntimeError, "exceeds the allocated size"):
            hdl.get_buffer(hdl.rank, (numel + 1,), dtype)

        # Writing the full data buffer must leave the preceding signal pad untouched.
        t.fill_(self.rank + 1)
        torch.testing.assert_close(local, torch.full_like(t, self.rank + 1))
        torch.testing.assert_close(signal_pad, torch.zeros_like(signal_pad))
        dist.barrier(device_ids=[self.rank])

    @skipIfUnsupportMultiNPU(2)
    def test_signal_pad_zeroed_after_free_and_realloc(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        numel, dtype = 1024, torch.float
        t1 = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl1 = symm_mem.rendezvous(t1, group=group_name)
        pad1 = hdl1.get_signal_pad(hdl1.rank, dtype=torch.uint8)
        pad1.fill_(0xFF)
        torch.npu.synchronize()
        del pad1, hdl1, t1
        dist.barrier(device_ids=[self.rank])

        t2 = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl2 = symm_mem.rendezvous(t2, group=group_name)
        pad2 = hdl2.get_signal_pad(hdl2.rank, dtype=torch.uint8)
        self.assertEqual(pad2.numel(), symm_mem.get_signal_pad_size())
        torch.testing.assert_close(pad2, torch.zeros_like(pad2))
        dist.barrier(device_ids=[self.rank])


if __name__ == "__main__":
    run_tests()
