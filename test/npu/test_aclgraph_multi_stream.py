import os
import torch
import torch_npu
from contextlib import nullcontext
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

def _npu_stream_switch(target_stream, enabled=True):
    if not enabled:
        return nullcontext()
    return torch.npu.stream(target_stream)


class TestAclgraphMultiStream(TestCase):
    """
    Tests for ACLGraph multi-stream capture correctness.

    Covers two scenarios:
      1. Shared-expert secondary stream pattern: a single side stream performs
         computation in parallel with the main capture stream, joined via
         wait_stream on both sides (fork / join).
      2. Parallel-branch fork-join: two streams each compute a branch
         simultaneously and merge the results; verifies numerical correctness
         across multiple replays with updated inputs.
    """

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_shared_expert_stream_capture_replay(self):
        device = torch.device("npu:0")
        torch.npu.set_device(device)

        shared_stream = torch.npu.Stream()
        positions = torch.ones(4, 4, device=device, dtype=torch.bfloat16)

        def _eager(x, side):
            default = torch.npu.current_stream()
            a = x + x
            side.wait_stream(default)
            with torch.npu.stream(side):
                b = x / x
                b = b + x
            default.wait_stream(side)
            return a, b

        ref_a, ref_b = _eager(positions, torch.npu.Stream())
        torch.npu.synchronize()

        out_a = out_b = None
        g = torch.npu.NPUGraph()

        with torch.npu.graph(g):
            stream_default = torch.npu.current_stream()
            out_a = positions + positions
            shared_stream.wait_stream(stream_default)
            with _npu_stream_switch(shared_stream):
                out_b = positions / positions
                out_b = out_b + positions
            stream_default.wait_stream(shared_stream)

        torch.npu.synchronize()
        torch.npu.empty_cache()

        g.replay()
        torch.npu.synchronize()

        self.assertRtolEqual(out_a.cpu(), ref_a.cpu())
        self.assertRtolEqual(out_b.cpu(), ref_b.cpu())

        positions.fill_(2.0)
        ref_a2, ref_b2 = _eager(positions, torch.npu.Stream())
        torch.npu.synchronize()

        g.replay()
        torch.npu.synchronize()

        self.assertRtolEqual(out_a.cpu(), ref_a2.cpu())
        self.assertRtolEqual(out_b.cpu(), ref_b2.cpu())

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_parallel_branch_capture_replay(self):
        device = torch.device("npu:0")
        torch.npu.set_device(device)

        M, K = 32, 32
        dtype = torch.float16

        s_inp = torch.randn(M, K, dtype=dtype, device=device)
        wa = torch.randn(K, K, dtype=dtype, device=device)
        wb = torch.randn(K, K, dtype=dtype, device=device)

        capture_stream = torch.npu.Stream()
        side_stream = torch.npu.Stream()
        g = torch.npu.NPUGraph()

        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            for _ in range(3):
                side_stream.wait_stream(capture_stream)
                with torch.npu.stream(side_stream):
                    _b = torch.matmul(s_inp, wb)
                _a = torch.matmul(s_inp, wa)
                capture_stream.wait_stream(side_stream)
                _ = _a + _b
        torch.npu.current_stream().wait_stream(capture_stream)
        torch.npu.synchronize()

        with torch.npu.stream(capture_stream):
            with torch.npu.graph(g, stream=capture_stream):
                side_stream.wait_stream(capture_stream)
                with torch.npu.stream(side_stream):
                    branch_b = torch.matmul(s_inp, wb)
                branch_a = torch.matmul(s_inp, wa)
                capture_stream.wait_stream(side_stream)
                output = branch_a + branch_b

        torch.npu.synchronize()
        torch.npu.empty_cache()

        ref1 = torch.matmul(s_inp, wa) + torch.matmul(s_inp, wb)
        g.replay()
        torch.npu.synchronize()
        self.assertRtolEqual(output.cpu(), ref1.cpu(), prec16=0.01)

        inp2 = torch.randn(M, K, dtype=dtype, device=device)
        ref2 = torch.matmul(inp2, wa) + torch.matmul(inp2, wb)
        s_inp.copy_(inp2)
        g.replay()
        torch.npu.synchronize()
        self.assertRtolEqual(output.cpu(), ref2.cpu(), prec16=0.01)


if __name__ == "__main__":
    run_tests()
