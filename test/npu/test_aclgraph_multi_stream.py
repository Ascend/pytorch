import os

import torch
import torch_npu
from contextlib import nullcontext
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

# Device families on which the multi-stream capture path is supported.
_SUPPORTED_DEVICES = ['Ascend910B', 'Ascend910_93', 'Ascend950']


def _npu_stream_switch(target_stream, enabled=True):
    if not enabled:
        return nullcontext()
    return torch.npu.stream(target_stream)


def _segment_pool_id_of(tensor):
    """Return the ``segment_pool_id`` of the allocator segment backing ``tensor``.

    ``(0, 0)`` is the default pool. Everything allocated while a graph's private
    pool is active must be tagged with ``NPUGraph.pool()`` instead: the default
    pool is returned to the driver by ``empty_cache()`` while a captured graph
    may still be replayed, which turns such an allocation into a dangling
    address on the next replay.
    """
    storage = tensor.untyped_storage()
    begin = storage.data_ptr()
    end = begin + storage.nbytes()
    for segment in torch_npu.npu.memory_snapshot():
        seg_begin = segment["address"]
        seg_end = seg_begin + segment["total_size"]
        if seg_begin <= begin and end <= seg_end:
            return segment["segment_pool_id"]
        if seg_begin <= begin < seg_end:
            return segment["segment_pool_id"]
    return None


class TestAclgraphMultiStream(TestCase):
    """
    Tests for ACLGraph multi-stream capture correctness.

    Covers three scenarios:
      1. Shared-expert secondary stream pattern: a single side stream performs
         computation in parallel with the main capture stream, joined via
         wait_stream on both sides (fork / join).
      2. Parallel-branch fork-join: two streams each compute a branch
         simultaneously and merge the results; verifies numerical correctness
         across multiple replays with updated inputs.
      3. Fork-join pool routing: a stream that joins the capture from a
         side-stream context must have its allocations served by the graph's
         private pool, and must stay valid across del + empty_cache + replay.
    """

    @SupportedDevices(_SUPPORTED_DEVICES)
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
            with _npu_stream_switch(shared_stream):
                # Fork from *inside* the side-stream context: the event is
                # recorded on the capture stream, but the wait is issued while
                # `shared_stream` is the current stream. This ordering is what
                # makes the host-side "am I in a capture" check and the
                # device-side capture membership of `shared_stream` diverge, so
                # the allocation below used to be routed into the default pool.
                shared_stream.wait_stream(stream_default)
                out_b = positions / positions
                out_b = out_b + positions
            stream_default.wait_stream(shared_stream)

        torch.npu.synchronize()
        torch.npu.empty_cache()

        graph_pool = g.pool()
        self.assertEqual(
            _segment_pool_id_of(out_a),
            graph_pool,
            "tensor allocated on the capture stream must live in the graph pool",
        )
        self.assertEqual(
            _segment_pool_id_of(out_b),
            graph_pool,
            "tensor allocated on the side stream right after the fork must live in the graph pool",
        )

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

    @SupportedDevices(_SUPPORTED_DEVICES)
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

        graph_pool = g.pool()
        self.assertEqual(
            _segment_pool_id_of(branch_b),
            graph_pool,
            "branch computed on the side stream must be allocated in the graph pool",
        )
        self.assertEqual(
            _segment_pool_id_of(output),
            graph_pool,
            "tensor allocated on the capture stream must live in the graph pool",
        )

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

    @SupportedDevices(_SUPPORTED_DEVICES)
    def test_side_stream_fork_capture_pool_routing(self):
        """Regression test: a side stream that joins a capture keeps its pool.

        A fork is a two-step handshake: record an event on the capture stream,
        then make the side stream wait for it. When the wait is issued from
        inside the side-stream context, the host-side TLS current stream is the
        side stream, which is not yet part of the capture -- the WAIT that adds
        it is still sitting in the task queue. Any allocation made by the very
        next op therefore observes "not capturing" and lands in the default
        pool (0, 0). ``empty_cache()`` later unmaps that memory while replay
        keeps writing to it.

        Both placements of the fork are exercised: the pool a tensor ends up in
        must not depend on which context the handshake happened to be issued
        from.
        """
        device = torch.device("npu:0")
        torch.npu.set_device(device)

        for fork_inside_side_context in (True, False):
            with self.subTest(fork_inside_side_context=fork_inside_side_context):
                self._check_fork_capture_pool_routing(device, fork_inside_side_context)

    def _check_fork_capture_pool_routing(self, device, fork_inside_side_context):
        capture_stream = torch.npu.Stream()
        side_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())

        g = torch.npu.NPUGraph()
        with torch.npu.stream(capture_stream):
            with torch.npu.graph(g, stream=capture_stream):
                main_buf = torch.full((256, 256), 2.0, device=device)
                # Backlog the capture stream so the fork's WAIT is still pending
                # when the next allocation asks the allocator about its pool.
                # Otherwise the consumer thread can execute the WAIT first and
                # the regression is masked (relu(2.0) == 2.0 keeps values equal).
                prepared = main_buf
                for _ in range(16):
                    prepared = torch.relu(prepared)
                ready = capture_stream.record_event()
                if fork_inside_side_context:
                    with torch.npu.stream(side_stream):
                        side_stream.wait_event(ready)
                        # Allocation-only op: nothing is enqueued between the
                        # WAIT and the allocation, so the routing decision is
                        # taken while the WAIT may still be pending.
                        side_buf = torch.empty(256, 256, device=device)
                        side_buf.fill_(3.0)
                else:
                    side_stream.wait_event(ready)
                    with torch.npu.stream(side_stream):
                        side_buf = torch.empty(256, 256, device=device)
                        side_buf.fill_(3.0)
                capture_stream.wait_stream(side_stream)
                output = prepared + side_buf

        torch.npu.synchronize()

        graph_pool = g.pool()
        self.assertIsNotNone(_segment_pool_id_of(side_buf), "side buffer has no owning segment")
        self.assertEqual(
            _segment_pool_id_of(side_buf),
            graph_pool,
            "side-stream allocation after the fork must live in the graph pool "
            f"(fork_inside_side_context={fork_inside_side_context})",
        )
        self.assertEqual(
            _segment_pool_id_of(output),
            graph_pool,
            "capture-stream allocation must live in the graph pool",
        )

        # The captured graph keeps writing to side_buf's address on every
        # replay, so releasing the cache must not unmap it.
        del side_buf
        torch.npu.empty_cache()

        g.replay()
        torch.npu.synchronize()

        expected = torch.full((256, 256), 5.0, device=device)
        self.assertRtolEqual(output.cpu(), expected.cpu())

        del output, prepared, main_buf, g
        torch.npu.synchronize()
        torch.npu.empty_cache()


if __name__ == "__main__":
    run_tests()
