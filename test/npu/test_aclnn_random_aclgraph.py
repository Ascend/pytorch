import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import run_tests, TestCase


class TestAclnnRandomAclgraph(TestCase):
    """ACLGraph 随机数算子下沉验证: eager vs graph 精度一致性."""

    def setUp(self):
        super().setUp()
        torch.npu.set_device(0)
        torch.npu.synchronize()

    def tearDown(self):
        torch.npu.synchronize()
        super().tearDown()

    def _assert_eager_graph_equal(self, eager_fn, graph_fn, seed=42):
        torch.npu.manual_seed(seed)
        eager_result = eager_fn()

        torch.npu.manual_seed(seed)
        g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g):
                graph_result = graph_fn()
            g.replay()
            torch.npu.synchronize()
        finally:
            del g

        if isinstance(eager_result, tuple):
            for ge, ee in zip(graph_result, eager_result):
                self.assertTrue(torch.equal(ge, ee), "eager vs graph mismatch")
        elif isinstance(eager_result, torch.Tensor):
            self.assertTrue(torch.equal(graph_result, eager_result), "eager vs graph mismatch")

    # Verify aclnnInplaceUniformTensor: eager vs graph produces identical results.
    def test_uniform(self):
        def fn():
            return torch.empty(256, device="npu").uniform_(-1.0, 1.0)
        self._assert_eager_graph_equal(fn, fn)

    # Verify aclnnInplaceNormalTensor: eager vs graph produces identical results.
    def test_normal(self):
        def fn():
            return torch.empty(256, device="npu").normal_(0.0, 1.0)
        self._assert_eager_graph_equal(fn, fn)

    # Verify aclnnInplaceRandomTensor with args: eager vs graph match.
    def test_random_from_to(self):
        def fn():
            return torch.empty(256, device="npu", dtype=torch.float32).random_(0, 100)
        self._assert_eager_graph_equal(fn, fn)

    # Verify aclnnInplaceRandomTensor without args: eager vs graph match.
    def test_random_no_args(self):
        def fn():
            return torch.empty(256, device="npu", dtype=torch.float32).random_()
        self._assert_eager_graph_equal(fn, fn, seed=77)

    # Verify aclnnMultinomialTensor: eager vs graph produces identical results.
    def test_multinomial(self):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
        def fn():
            return torch.multinomial(weights, num_samples=100, replacement=True)
        self._assert_eager_graph_equal(fn, fn)

    # Verify aclnnDropoutGenMaskV2: eager vs graph produces identical results.
    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_native_dropout(self):
        x = torch.randn(256, 256, device="npu")
        def fn():
            return torch.nn.functional.dropout(x, p=0.5, training=True)
        self._assert_eager_graph_equal(fn, fn)

    # Verify lazy registration: no register_generator_state() call, stream capture_begin+uniform_.
    # The first RNG op auto-creates per-capture state via get_capture_state(capture_id, true),
    # consecutive replays produce different results, proving RNG offset advances.
    def test_lazy_registration(self):
        x = torch.empty(256, device="npu")
        g = torch.npu.NPUGraph()
        s = torch.npu.Stream()
        try:
            with torch.npu.stream(s):
                g.capture_begin()
                y = x.uniform_(-1.0, 1.0)
                g.capture_end()
            torch.npu.current_stream().wait_stream(s)
            results = []
            for _ in range(3):
                g.replay()
                results.append(y.clone())
                torch.npu.synchronize()
        finally:
            del g
        self.assertFalse(torch.equal(results[0], results[1]), "Consecutive replays should produce different results")

    # Verify replay == eager: same seed, stream capture_begin+uniform_.
    # capture_epilogue/replay_prologue correctly record and advance wholegraph_increment,
    # replay produces bitwise-identical results to eager.
    def test_replay_consistency(self):
        x = torch.empty(256, device="npu")
        seed = 12345

        torch.npu.manual_seed(seed)
        g = torch.npu.NPUGraph()
        s = torch.npu.Stream()
        try:
            with torch.npu.stream(s):
                g.capture_begin()
                y_graph = x.uniform_(-1.0, 1.0)
                g.capture_end()
            torch.npu.current_stream().wait_stream(s)

            torch.npu.manual_seed(seed)
            g.replay()
            graph_result = y_graph.clone()
            torch.npu.synchronize()
        finally:
            del g

        torch.npu.manual_seed(seed)
        eager_result = torch.empty(256, device="npu").uniform_(-1.0, 1.0)

        self.assertTrue(torch.equal(graph_result, eager_result), "Graph replay should be identical to eager with same seed")


if __name__ == "__main__":
    run_tests()
