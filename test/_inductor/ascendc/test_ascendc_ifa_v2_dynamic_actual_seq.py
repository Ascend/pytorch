"""AscendC backend: IFA v2 dynamic actual seq with aclgraph replay."""
from contextlib import contextmanager
import os
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
DEFERRED_UPDATE_LOG = "NPUGRAPH-TREE ACLGraph update deferred until after replay"
UPDATE_BEFORE_REPLAY_LOG = "NPUGRAPH-TREE ACLGraph update before replay"


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


@contextmanager
def _temporary_env(name, value):
    old_value = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _run_dynamic_actual_seq_case():
    import torch_npu._inductor.ascendc.config as ascendc_config

    old_cudagraphs = config.triton.cudagraphs
    old_cudagraph_trees = config.triton.cudagraph_trees
    old_force_disable_caches = config.force_disable_caches
    old_slow_path_asserts = config.triton.slow_path_cudagraph_asserts
    old_sync_around_fuse_kernel = ascendc_config.sync_around_fuse_kernel
    try:
        config.triton.cudagraphs = True
        config.triton.cudagraph_trees = True
        config.force_disable_caches = True
        config.triton.slow_path_cudagraph_asserts = False
        if os.getenv("ASCEND_LAUNCH_BLOCKING", None) == "1":
            ascendc_config.sync_around_fuse_kernel = False

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
                if diff > 1e-2:
                    raise AssertionError(f"seq={seq} max_diff={diff}")
            torch.npu.synchronize()

        return log_stream.getvalue()
    finally:
        config.triton.cudagraphs = old_cudagraphs
        config.triton.cudagraph_trees = old_cudagraph_trees
        config.force_disable_caches = old_force_disable_caches
        config.triton.slow_path_cudagraph_asserts = old_slow_path_asserts
        ascendc_config.sync_around_fuse_kernel = old_sync_around_fuse_kernel
        torch._dynamo.reset()


def _assert_dynamic_actual_seq_key_reuse(test_case, logs):
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
    test_case.assertGreaterEqual(compile_recording_count, 1, logs)
    test_case.assertLessEqual(compile_recording_count, 2, logs)
    test_case.assertLessEqual(warmup_count, 2, logs)
    test_case.assertLessEqual(record_count, 2, logs)
    test_case.assertGreaterEqual(replay_count, 1, logs)
    test_case.assertEqual(
        update_count,
        len(SEQ_CONFIGS) - warmup_count,
        logs,
    )
    test_case.assertIn("NPUGRAPH-TREE State state=EXECUTION", logs)
    for bad_key_fragment in ("100, 50", "80, 90", "128, 30"):
        test_case.assertFalse(
            any(
                bad_key_fragment in line
                for line in compile_recording_lines
            ),
            "\n".join(compile_recording_lines),
        )


@unittest.skipIf(not torch.npu.is_available(), "requires an NPU device")
class TestAscendcIFAv2DynamicActualSeq(TestCase):

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_dynamic_actual_seq_reuses_npugraph_key(self):
        with _temporary_env("ASCEND_LAUNCH_BLOCKING", None):
            logs = _run_dynamic_actual_seq_case()
        _assert_dynamic_actual_seq_key_reuse(self, logs)
        self.assertIn(DEFERRED_UPDATE_LOG, logs)
        self.assertNotIn(UPDATE_BEFORE_REPLAY_LOG, logs)

        with _temporary_env("ASCEND_LAUNCH_BLOCKING", "1"):
            logs = _run_dynamic_actual_seq_case()
        _assert_dynamic_actual_seq_key_reuse(self, logs)
        self.assertIn(UPDATE_BEFORE_REPLAY_LOG, logs)
        self.assertNotIn(DEFERRED_UPDATE_LOG, logs)


if __name__ == "__main__":
    run_tests()
