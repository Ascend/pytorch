import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import run_tests, TestCase

import torch


def _to_cumsum(seq_lens):
    """将 seq_len 列表转为累加和 Tensor (用于 actual_seq_qlen/kvlen 参数)."""
    return torch.cumsum(torch.tensor(seq_lens, dtype=torch.int64), dim=0)


class _FA3TestBase(TestCase):
    """FA3 测试基类: 每个 setUp 清理 NPU device 状态, 防止 graph capture 残留污染."""

    def setUp(self):
        super().setUp()
        torch.npu.set_device(0)
        torch.npu.synchronize()

    def tearDown(self):
        torch.npu.synchronize()
        super().tearDown()


class TestFA3ForwardAclgraphUpdate(TestCase):
    @staticmethod
    def _to_cumsum(seq_lens):
        return torch.cumsum(torch.tensor(seq_lens, dtype=torch.int64), dim=0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_forward_update_with_non_out_and_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [128, 128]
        seq_qlen_new = [64, 192]
        seq_kvlen_old = [128, 128]
        seq_kvlen_new = [192, 64]
        T = sum(seq_qlen_old)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        qlen_old = self._to_cumsum(seq_qlen_old)
        kvlen_old = self._to_cumsum(seq_kvlen_old)
        qlen_new = self._to_cumsum(seq_qlen_new)
        kvlen_new = self._to_cumsum(seq_kvlen_new)

        res_src = torch_npu.npu_fusion_attention_v3(
            query,
            key,
            value,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale=scale,
            actual_seq_qlen=qlen_new,
            actual_seq_kvlen=kvlen_new,
        )

        g = torch_npu.npu.NPUGraph()
        attention_score = None
        with torch_npu.npu.graph(g, auto_dispatch_capture=True):
            attention_score, softmax_max, softmax_sum, softmax_out, seed, offset = (
                torch_npu.npu_fusion_attention_v3(
                    query,
                    key,
                    value,
                    head_num=N,
                    input_layout="TND",
                    keep_prob=1.0,
                    scale=scale,
                    actual_seq_qlen=qlen_old,
                    actual_seq_kvlen=kvlen_old,
                )
            )

        g.update(
            cpu_update_input=[
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                }
            ]
        )
        g.replay()
        self.assertEqual(attention_score.cpu(), res_src[0].cpu(), prec=1)


class TestFA3BackwardAclgraphUpdate(TestCase):
    @staticmethod
    def _to_cumsum(seq_lens):
        return torch.cumsum(torch.tensor(seq_lens, dtype=torch.int64), dim=0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_backward_update_with_non_out_and_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [128, 128]
        seq_qlen_new = [64, 192]
        seq_kvlen_old = [128, 128]
        seq_kvlen_new = [192, 64]
        T = sum(seq_qlen_old)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        dy = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        qlen_old = self._to_cumsum(seq_qlen_old)
        kvlen_old = self._to_cumsum(seq_kvlen_old)
        qlen_new = self._to_cumsum(seq_qlen_new)
        kvlen_new = self._to_cumsum(seq_kvlen_new)

        # Run forward first to get real softmax_max/softmax_sum/attention_in
        with torch.no_grad():
            fw_result = torch_npu.npu_fusion_attention_v3(
                query,
                key,
                value,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale=scale,
                actual_seq_qlen=qlen_new,
                actual_seq_kvlen=kvlen_new,
            )
        attention_in = fw_result[0]
        softmax_max = fw_result[1]
        softmax_sum = fw_result[2]

        res_src = torch_npu.npu_fusion_attention_grad_v3(
            query,
            key,
            value,
            dy,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale_value=scale,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attention_in,
            actual_seq_qlen=qlen_new,
            actual_seq_kvlen=kvlen_new,
        )

        g = torch_npu.npu.NPUGraph()
        dq = None
        with torch_npu.npu.graph(g, auto_dispatch_capture=True):
            dq, dk, dv, dpse, dsink = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                attention_in=attention_in,
                actual_seq_qlen=qlen_old,
                actual_seq_kvlen=kvlen_old,
            )

        g.update(
            cpu_update_input=[
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                }
            ]
        )
        g.replay()
        self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=0.1)


class TestFA3Compile(_FA3TestBase):
    """torch.compile(backend='inductor', mode='reduce-overhead') 测试."""

    @staticmethod
    def _to_cumsum(seq_lens):
        return _to_cumsum(seq_lens)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_forward_compile_reduce_overhead(self):
        from torch_npu._inductor import config as npu_config

        npu_config.npugraph_trees.disable_cpu_input_check = True
        """TND 前向, compile + reduce-overhead 模式.

        qlen/kvlen 作为闭包常量捕获, 不作为 compile 输入.
        compile 只处理 NPU tensor (q/k/v), FA3 handler 负责 ACLGraph 内的 graph capture.
        """
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen = [64, 64]
        T = sum(seq_qlen)
        qlen = self._to_cumsum(seq_qlen)
        kvlen = self._to_cumsum(seq_qlen)

        query = torch.randn(
            T, N, D, dtype=torch.float16, device="npu", requires_grad=True
        )
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")

        # qlen/kvlen 闭包捕获, 不入 compile 输入签名
        def fa3_fn(q, k, v, actual_seq_qlen, actual_seq_kvlen):
            result = torch_npu.npu_fusion_attention_v3(
                q,
                k,
                v,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale=scale,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            return result[0]

        eager_out = fa3_fn(query, key, value, qlen, kvlen)
        compiled_fn = torch.compile(fa3_fn, backend="inductor", mode="reduce-overhead")
        compiled_out = compiled_fn(query, key, value, qlen, kvlen)
        self.assertEqual(compiled_out.cpu(), eager_out.cpu(), prec=1.0)
        compiled_out.sum().backward()
        self.assertEqual(compiled_out.cpu(), eager_out.cpu(), prec=1.0)


if __name__ == "__main__":
    run_tests()
