import unittest

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.distributed_c10d import (
    _register_process_group,
    _unregister_process_group,
)
from torch_npu.testing.testcase import TestCase, run_tests

# Upstream #188548 added all_gather_single / reduce_scatter_single trampoline
# overrides to PyProcessGroup (landed in torch 2.14). On torch < 2.14 the
# bridge does not exist and calls initiated by the C++ kernel fall through to
# the base class implementation and fail, so the real-dispatch test is gated
# by version.
_TORCH_GE_2_14 = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 14)


class NpuDummyProcessGroup(dist.ProcessGroup):
    """Pure Python dummy ProcessGroup (no actual communication).

    Mimics DummyProcessGroup from upstream test/distributed/test_c10d_common.py:
    each collective leaves a recognizable local "fingerprint" and is recorded
    in collectives_called, so the test can assert that a call initiated by the
    C++ kernel was really routed to the Python implementation.
    Only the subset needed by this file is implemented.
    """

    def __init__(self, rank, size):
        super().__init__(rank, size)
        self.global_rank = rank
        self.group_size = size
        # Records the names of collectives dispatched here; used to assert
        # that calls actually reach the Python override
        self.collectives_called = set()

    def rank(self):
        return self.global_rank

    def size(self):
        return self.group_size

    def getBackendName(self):
        return "npu_dummy"

    def all_gather_single(self, output_tensor, input_tensor, opts=None):
        self.collectives_called.add("all_gather_single")
        for chunk in output_tensor.chunk(self.size()):
            chunk.copy_(input_tensor)
        return DummyWork()

    def reduce_scatter_single(self, output_tensor, input_tensor, opts=None):
        self.collectives_called.add("reduce_scatter_single")
        output_tensor.copy_(input_tensor.chunk(self.size())[self.rank()])
        return DummyWork()


class DummyWork(dist._Work):
    def wait(self, timeout=5.0):
        if torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()
        return True


class PyPGTrampolineNpuTestCase(TestCase):
    """Verify that upstream #188548 (all_gather_single / reduce_scatter_single
    trampoline overrides added to PyProcessGroup) works and has no side
    effects in the torch_npu environment.

    Runs on a single card: the dummy PG does not communicate, so no hccl
    initialization is needed.
    """

    @unittest.skipIf(
        not _TORCH_GE_2_14,
        "PyProcessGroup all_gather_single trampoline landed in torch 2.14 "
        "(upstream #188548)",
    )
    def test_all_gather_single_trampoline_npu(self):
        # torch.ops._c10d_functional.all_gather_into_tensor_out is a
        # device-agnostic op whose C++ implementation calls the
        # group->all_gather_single virtual function directly. On NPU tensors
        # the call goes through the PyProcessGroup trampoline into the Python
        # override; on torch < 2.14 the bridge is missing and the call falls
        # through to the base class and fails, hence the version gate on this
        # test.
        pg = NpuDummyProcessGroup(0, 1)
        _register_process_group("test_all_gather_single_npu", pg)
        try:
            input_tensor = torch.ones(2, device="npu")
            output = torch.empty(2, device="npu")
            torch.ops._c10d_functional.all_gather_into_tensor_out(
                input_tensor, 1, "test_all_gather_single_npu", out=output
            )
            torch.ops._c10d_functional.wait_tensor(output)
            self.assertIn("all_gather_single", pg.collectives_called)
            self.assertEqual(output.cpu(), torch.ones(2))
        finally:
            _unregister_process_group("test_all_gather_single_npu")

    def test_reduce_scatter_single_npu(self):
        # Sanity check: call the Python method of the same name directly (no
        # trampoline involved) to verify the dummy implementation itself is
        # correct on NPU; should pass on both 2.13 and 2.14.
        pg = NpuDummyProcessGroup(0, 1)
        input_tensor = torch.ones(2, device="npu")
        output = torch.zeros(2, device="npu")
        pg.reduce_scatter_single(output, input_tensor).wait()
        self.assertIn("reduce_scatter_single", pg.collectives_called)
        self.assertEqual(output.cpu(), torch.ones(2))


if __name__ == "__main__":
    run_tests()
