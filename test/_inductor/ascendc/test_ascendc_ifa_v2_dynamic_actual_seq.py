"""AscendC backend: IFA v2 dynamic actual seq with aclgraph replay."""
import unittest

import torch
import torch_npu
from torch._inductor import config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.logging_utils import logs_to_string


B = 2
S_MAX = 128
N = 8
D = 128
SCALE = 1.0 / (D ** 0.5)
SEQ_CONFIGS = (
    [64, 64],
    [100, 50],
    [80, 90],
    [128, 30],
)


def _make_inputs():
    q = torch.randn(B, 1, N * D, dtype=torch.float16, device="npu")
    k = torch.randn(B, S_MAX, N * D, dtype=torch.float16, device="npu")
    v = torch.randn(B, S_MAX, N * D, dtype=torch.float16, device="npu")
    return q, k, v


def _ifa_v2(q, k, v, actual_seq_qlen, actual_seq_kvlen):
    return torch.ops.npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        num_query_heads=N,
        num_key_value_heads=N,
        softmax_scale=SCALE,
        input_layout="BSH",
    )


@unittest.skipIf(not torch.npu.is_available(), "requires an NPU device")
class TestAscendcIFAv2DynamicActualSeq(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._ascendc_ok = False
        try:
            x = torch.randn(4, 4, device="npu")
            torch.compile(
                lambda t: t + 1,
                backend="inductor",
                options={"npu_backend": "ascendc"},
            )(x)
            cls._ascendc_ok = True
        except Exception:
            pass

    def setUp(self):
        super().setUp()
        if not self._ascendc_ok:
            self.skipTest("ascendc backend not available")
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_dynamic_actual_seq_reuses_npugraph_key(self):
        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_force_disable_caches = config.force_disable_caches
        old_slow_path_asserts = config.triton.slow_path_cudagraph_asserts
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.force_disable_caches = True
            config.triton.slow_path_cudagraph_asserts = False

            torch.manual_seed(0)
            torch_npu.npu.manual_seed(0)
            q, k, v = _make_inputs()
            compiled_ifa_v2 = torch.compile(
                _ifa_v2,
                backend="inductor",
                dynamic=True,
                fullgraph=True,
                options={"npu_backend": "ascendc", "triton.cudagraphs": True},
            )

            log_stream, ctx = logs_to_string("torch_npu.npugraph", "cudagraphs")
            with torch.no_grad(), ctx():
                for seq in SEQ_CONFIGS:
                    # Pass the same Python list object for qlen and kvlen, matching
                    # decode-style actual seq inputs from the original reproducer.
                    seq_arg = list(seq)
                    compiled_out = compiled_ifa_v2(q, k, v, seq_arg, seq_arg)
                    eager_out = _ifa_v2(q, k, v, seq_arg, seq_arg)
                    diff = (
                        compiled_out[0].float() - eager_out[0].float()
                    ).abs().max().item()
                    self.assertLessEqual(
                        diff,
                        1e-2,
                        f"seq={seq} max_diff={diff}",
                    )
                torch.npu.synchronize()

            logs = log_stream.getvalue()
            compile_recording_lines = [
                line for line in logs.splitlines()
                if "NPUGRAPH-TREE Compile recording" in line
            ]
            compile_recording_count = len(compile_recording_lines)
            warmup_count = logs.count("NPUGRAPH-TREE Warmup Running warmup")
            record_count = logs.count("NPUGRAPH-TREE Node Record function=")
            replay_count = logs.count("NPUGRAPH Replay graph_id=")
            update_count = logs.count("NPUGraph: updating graph")

            # The first concrete list may compile statically, then Dynamo may
            # generalize to a dynamic actual seq graph. Dynamic actual seq values
            # must not create one NPUGraphTree key per sequence pair.
            self.assertGreaterEqual(compile_recording_count, 1, logs)
            self.assertLessEqual(compile_recording_count, 2, logs)
            self.assertLessEqual(warmup_count, 2, logs)
            self.assertLessEqual(record_count, 2, logs)
            self.assertGreaterEqual(replay_count, 1, logs)
            self.assertEqual(
                update_count,
                len(SEQ_CONFIGS) - warmup_count,
                logs,
            )
            self.assertIn("NPUGRAPH-TREE State state=EXECUTION", logs)
            for bad_key_fragment in ("100, 50", "80, 90", "128, 30"):
                self.assertFalse(
                    any(
                        bad_key_fragment in line
                        for line in compile_recording_lines
                    ),
                    "\n".join(compile_recording_lines),
                )
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.force_disable_caches = old_force_disable_caches
            config.triton.slow_path_cudagraph_asserts = old_slow_path_asserts


if __name__ == "__main__":
    run_tests()
