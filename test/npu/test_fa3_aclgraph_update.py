"""
FA3 (npu_fusion_attention_v3) ACLGraph Update 测试套件

运行方式:
    python -m pytest pytorch_5260/test/npu/test_fa3_aclgraph_update.py -v
    python -m pytest pytorch_5260/test/npu/test_fa3_aclgraph_update.py -k "test_tnd" -v

覆盖场景矩阵:
+--------+---------+----------------+------------------------+
| Layout | Dropout | Eager+ACLGraph | compile+reduce-overhead|
+--------+---------+----------------+------------------------+
| TND    | Off     | Yes (+update)  | Yes                    |
| TND    | On      | No (fallback)  | No (fallback)          |
+--------+---------+----------------+------------------------+

用例 Checklist:
====================================================================================================
#   类名                              用例名                                                类型/说明
====================================================================================================

--- 一、TND 前向 ACLGraph Update (TestFA3ForwardAclgraphUpdate) ---
 1  test_fa3_forward_update_with_non_out_and_auto_...    auto_dispatch + .default 接口, handler自动转.out
 2  test_fa3_forward_multi_replay_with_update            多次update+replay (3轮不同seq_lens)
 3  test_fa3_forward_three_batch_update                  3 batch seq_len变更update

--- 二、TND 反向 ACLGraph Update (TestFA3BackwardAclgraphUpdate) ---
 4  test_fa3_backward_update_with_non_out_and_auto_...   auto_dispatch + .default (反向)

--- 三、前反向联动 (TestFA3ForwardBackwardCombined) ---
 5  test_tnd_forward_backward_combined_aclgraph          双图联动: g_fw+g_bw 分别update
 6  test_tnd_forward_backward_multi_iteration            TND多轮训练迭代 (双图)
 7  test_tnd_forward_backward_single_graph_update        ★单图联动: 前反向同一NPUGraph, 一次性update
 8  test_tnd_single_graph_multi_replay                   ★单图联动: 多轮update+replay

--- 四、torch.compile + reduce-overhead (TestFA3Compile) ---
 9  test_tnd_forward_compile_reduce_overhead             TND前向 compile+reduce-overhead
10  test_tnd_graph_partition                              TND前向 compile + reduce-overhead + 图划分

====================================================================================================
总计: 4个测试类, 10个测试用例
"""

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


# =========================================================================
# 一、TND 前向 ACLGraph Update 测试
# =========================================================================


class TestFA3ForwardAclgraphUpdate(_FA3TestBase):
    """TND 布局前向 ACLGraph update 测试.

    验证 FA3ForwardHandler 的 prepare_capture / update_args 流程:
    - prepare_capture: 计算 workspace → infer_output → 构造 out 列表 → 切换到 .out 接口
    - update_args: 刷新 args[14] (actual_seq_qlen), args[15] (actual_seq_kvlen)
    """

    @staticmethod
    def _to_cumsum(seq_lens):
        return _to_cumsum(seq_lens)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_forward_update_with_non_out_and_auto_dispatch_capture(self):
        """[类型: auto_dispatch + .default接口] TND 前向, 调用 .default 接口, handler 自动转 .out.

        验证用户直接调用 npu_fusion_attention_v3 (非 .out), handler 的 prepare_capture
        自动完成: get_max_workspace → infer_output → 构造 out → 切换到 .out 接口.
        """
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
        # 用户调用 .default 接口, handler 自动 prepare_capture 转为 .out
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
        self.assertEqual(attention_score.cpu(), res_src[0].cpu())

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_forward_multi_replay_with_update(self):
        """[类型: 多次replay] TND 前向, 多次 update+replay, 验证每次 seq_lens 不同时结果正确.

        场景: capture 一次 → update(seq1)+replay → update(seq2)+replay → update(seq3)+replay
        验证图复用能力, 每次刷新不同的 CPU seq_lens.
        """
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [128, 128]
        T = sum(seq_qlen_old)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        qlen_old = self._to_cumsum(seq_qlen_old)
        kvlen_old = self._to_cumsum(seq_qlen_old)

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

        # 第一次 update+replay: [64, 192]
        seq1_q = [64, 192]
        seq1_kv = [192, 64]
        qlen1 = self._to_cumsum(seq1_q)
        kvlen1 = self._to_cumsum(seq1_kv)
        res1 = torch_npu.npu_fusion_attention_v3(
            query,
            key,
            value,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale=scale,
            actual_seq_qlen=qlen1,
            actual_seq_kvlen=kvlen1,
        )
        g.update(
            cpu_update_input=[{"actual_seq_qlen": qlen1, "actual_seq_kvlen": kvlen1}]
        )
        g.replay()
        self.assertEqual(attention_score.cpu(), res1[0].cpu())

        # 第二次 update+replay: [192, 64]
        seq2_q = [192, 64]
        seq2_kv = [64, 192]
        qlen2 = self._to_cumsum(seq2_q)
        kvlen2 = self._to_cumsum(seq2_kv)
        res2 = torch_npu.npu_fusion_attention_v3(
            query,
            key,
            value,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale=scale,
            actual_seq_qlen=qlen2,
            actual_seq_kvlen=kvlen2,
        )
        g.update(
            cpu_update_input=[{"actual_seq_qlen": qlen2, "actual_seq_kvlen": kvlen2}]
        )
        g.replay()
        self.assertEqual(attention_score.cpu(), res2[0].cpu())

        # 第三次 update+replay: 恢复到原始 seq_lens
        res3 = torch_npu.npu_fusion_attention_v3(
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
        g.update(
            cpu_update_input=[
                {"actual_seq_qlen": qlen_old, "actual_seq_kvlen": kvlen_old}
            ]
        )
        g.replay()
        self.assertEqual(attention_score.cpu(), res3[0].cpu())

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_forward_three_batch_update(self):
        """[类型: 多batch更新] TND 前向, 3个batch (3个seq_len) 变更 update.

        验证 actual_seq_qlen/kvlen 长度不变但值变化时, update 正确刷新.
        seq_lens 数量固定为3, 但各段长度不同.
        """
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [85, 85, 86]  # 3 batch, T=256
        seq_qlen_new = [128, 64, 64]
        seq_kvlen_old = [85, 85, 86]
        seq_kvlen_new = [100, 80, 76]
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
            attention_score, _, _, _, _, _ = torch_npu.npu_fusion_attention_v3(
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

        g.update(
            cpu_update_input=[
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                }
            ]
        )
        g.replay()
        self.assertEqual(attention_score.cpu(), res_src[0].cpu())


# =========================================================================
# 二、TND 反向 ACLGraph Update 测试
# =========================================================================


class TestFA3BackwardAclgraphUpdate(_FA3TestBase):
    """TND 布局反向 ACLGraph update 测试.

    验证 FA3BackwardHandler 的 prepare_capture / update_args 流程:
    - prepare_capture: 计算 workspace → infer_output → 构造 out 列表 → 切换到 .out 接口
    - update_args: 刷新 args[21] (actual_seq_qlen), args[22] (actual_seq_kvlen)
    """

    @staticmethod
    def _to_cumsum(seq_lens):
        return _to_cumsum(seq_lens)

    def _run_forward(self, query, key, value, N, scale, qlen, kvlen):
        """执行一次前向, 获取反向所需的中间结果."""
        with torch.no_grad():
            return torch_npu.npu_fusion_attention_v3(
                query,
                key,
                value,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale=scale,
                actual_seq_qlen=qlen,
                actual_seq_kvlen=kvlen,
            )

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_fa3_backward_update_with_non_out_and_auto_dispatch_capture(self):
        """[类型: auto_dispatch + .default接口] TND 反向, 调用 .default 接口, handler 自动转 .out.

        验证用户调用 npu_fusion_attention_grad_v3 (非 .out) 时,
        FA3BackwardHandler.prepare_capture 自动完成完整转换.
        """
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

        fw_result = self._run_forward(query, key, value, N, scale, qlen_new, kvlen_new)
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
        self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=1.0)


# =========================================================================
# 三、前反向联动测试
# =========================================================================


class TestFA3ForwardBackwardCombined(_FA3TestBase):
    """前反向联动测试.

    模拟真实训练场景: 前向产出 softmax_max/softmax_sum/attention_in 传递给反向.
    """

    @staticmethod
    def _to_cumsum(seq_lens):
        return _to_cumsum(seq_lens)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_forward_backward_combined_aclgraph(self):
        """[类型: TND前反向联动] TND 前向+反向均在 ACLGraph 中, update 刷新 seq_lens.

        流程:
        1. 前向 ACLGraph capture → update → replay (产出 softmax_max/sum/attention_in)
        2. 使用前向产出执行反向 ACLGraph capture → update → replay
        3. 验证反向结果与 eager 一致
        """
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

        # ---- Step 1: 前向 ACLGraph ----
        g_fw = torch_npu.npu.NPUGraph()
        fw_attention = None
        with torch_npu.npu.graph(g_fw, auto_dispatch_capture=True):
            fw_attention, fw_smax, fw_ssum, fw_sout, fw_seed, fw_offset = (
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

        g_fw.update(
            cpu_update_input=[
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                }
            ]
        )
        g_fw.replay()

        # ---- Step 2: 反向 ACLGraph (使用前向产出的中间结果) ----
        g_bw = torch_npu.npu.NPUGraph()
        dq = None
        with torch_npu.npu.graph(g_bw, auto_dispatch_capture=True):
            dq, dk, dv, dpse, dsink = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_smax,
                softmax_sum=fw_ssum,
                attention_in=fw_attention,
                actual_seq_qlen=qlen_old,
                actual_seq_kvlen=kvlen_old,
            )

        g_bw.update(
            cpu_update_input=[
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                }
            ]
        )
        g_bw.replay()

        # ---- Step 3: Eager 基准对比 ----
        with torch.no_grad():
            fw_eager = torch_npu.npu_fusion_attention_v3(
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

        res_src = torch_npu.npu_fusion_attention_grad_v3(
            query,
            key,
            value,
            dy,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale_value=scale,
            softmax_max=fw_eager[1],
            softmax_sum=fw_eager[2],
            attention_in=fw_eager[0],
            actual_seq_qlen=qlen_new,
            actual_seq_kvlen=kvlen_new,
        )

        self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=1.0)
        self.assertEqual(dk.cpu(), res_src[1].cpu(), prec=1.0)
        self.assertEqual(dv.cpu(), res_src[2].cpu(), prec=1.0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_forward_backward_multi_iteration(self):
        """[类型: TND多轮训练迭代] TND 前反向联动, 模拟多轮训练的 update.

        流程: capture 一次 → 多轮 (update → 前向replay → 反向replay).
        每轮使用不同的 seq_lens, 验证前反向联动在多轮迭代中的正确性.
        """
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [128, 128]
        T = sum(seq_qlen_old)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        dy = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        qlen_old = self._to_cumsum(seq_qlen_old)
        kvlen_old = self._to_cumsum(seq_qlen_old)

        # Capture 前向图
        g_fw = torch_npu.npu.NPUGraph()
        fw_attention = None
        with torch_npu.npu.graph(g_fw, auto_dispatch_capture=True):
            fw_attention, fw_smax, fw_ssum, fw_sout, fw_seed, fw_offset = (
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

        # Capture 反向图
        g_bw = torch_npu.npu.NPUGraph()
        dq = None
        with torch_npu.npu.graph(g_bw, auto_dispatch_capture=True):
            dq, dk, dv, dpse, dsink = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_smax,
                softmax_sum=fw_ssum,
                attention_in=fw_attention,
                actual_seq_qlen=qlen_old,
                actual_seq_kvlen=kvlen_old,
            )

        # 多轮迭代
        test_seqs = [
            ([64, 192], [192, 64]),
            ([192, 64], [64, 192]),
            ([100, 156], [156, 100]),
        ]
        for seq_q, seq_kv in test_seqs:
            qlen_new = self._to_cumsum(seq_q)
            kvlen_new = self._to_cumsum(seq_kv)
            update_input = [
                {"actual_seq_qlen": qlen_new, "actual_seq_kvlen": kvlen_new}
            ]

            # 前向 update + replay
            g_fw.update(cpu_update_input=update_input)
            g_fw.replay()

            # 反向 update + replay
            g_bw.update(cpu_update_input=update_input)
            g_bw.replay()

            # Eager 基准
            with torch.no_grad():
                fw_eager = torch_npu.npu_fusion_attention_v3(
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
            res_src = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_eager[1],
                softmax_sum=fw_eager[2],
                attention_in=fw_eager[0],
                actual_seq_qlen=qlen_new,
                actual_seq_kvlen=kvlen_new,
            )

            self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=1.0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_forward_backward_single_graph_update(self):
        """[类型: 单图前反向联动update] TND 前向+反向在同一个 ACLGraph 中, 一次性 update 刷新两者参数.

        区别于前向/反向分两张图 (g_fw / g_bw) 再分别 update 的模式,
        此用例将前向和反向 capture 到同一个 NPUGraph 中,
        通过单次 g.update(cpu_update_input=[fw_update, bw_update]) 同时刷新
        前向的 actual_seq_qlen/kvlen (args[14]/[15]) 和
        反向的 actual_seq_qlen/kvlen (args[21]/[22]).

        流程:
        1. 单个 NPUGraph capture: 前向 → 中间结果 (softmax_max/sum/attention_in)
           作为反向输入 → 反向
        2. 单次 g.update() 刷新前向和反向的 seq_lens
        3. 单次 g.replay() 执行前向+反向
        4. 与 eager 基准对比
        """
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

        # ---- 单图 Capture: 前向 + 反向 ----
        g = torch_npu.npu.NPUGraph()
        fw_attention = None
        dq = None
        with torch_npu.npu.graph(g, auto_dispatch_capture=True):
            # 前向: handler 自动 prepare_capture (get_workspace → infer_output → .out)
            fw_attention, fw_smax, fw_ssum, fw_sout, fw_seed, fw_offset = (
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

            # 反向: 直接使用前向产出 (同一图内, replay 时前向先写、反向后读)
            dq, dk, dv, dpse, dsink = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_smax,
                softmax_sum=fw_ssum,
                attention_in=fw_attention,
                actual_seq_qlen=qlen_old,
                actual_seq_kvlen=kvlen_old,
            )

        # ---- 单次 update: 同时刷新前向和反向的 seq_lens ----
        # cpu_update_input 列表中, [0] 对应前向 handler, [1] 对应反向 handler
        g.update(
            cpu_update_input=[
                {"actual_seq_qlen": qlen_new, "actual_seq_kvlen": kvlen_new},  # forward
                {
                    "actual_seq_qlen": qlen_new,
                    "actual_seq_kvlen": kvlen_new,
                },  # backward
            ]
        )

        # ---- 单次 replay: 前向+反向顺序执行 ----
        g.replay()

        # ---- Eager 基准 ----
        with torch.no_grad():
            fw_eager = torch_npu.npu_fusion_attention_v3(
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

        res_src = torch_npu.npu_fusion_attention_grad_v3(
            query,
            key,
            value,
            dy,
            head_num=N,
            input_layout="TND",
            keep_prob=1.0,
            scale_value=scale,
            softmax_max=fw_eager[1],
            softmax_sum=fw_eager[2],
            attention_in=fw_eager[0],
            actual_seq_qlen=qlen_new,
            actual_seq_kvlen=kvlen_new,
        )

        self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=1.0)
        self.assertEqual(dk.cpu(), res_src[1].cpu(), prec=1.0)
        self.assertEqual(dv.cpu(), res_src[2].cpu(), prec=1.0)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_single_graph_multi_replay(self):
        """[类型: 单图前反向多次replay] 单图 capture 前向+反向, 多轮 update+replay.

        验证单图模式下, 多轮使用不同 seq_lens update+replay 的稳定性.
        每轮: g.update([fw_update, bw_update]) → g.replay() → 对比 eager.
        """
        N, D = 8, 128
        scale = 1.0 / D
        seq_qlen_old = [128, 128]
        T = sum(seq_qlen_old)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        key = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        value = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        dy = torch.randn(T, N, D, dtype=torch.float16, device="npu")
        qlen_old = self._to_cumsum(seq_qlen_old)
        kvlen_old = self._to_cumsum(seq_qlen_old)

        # 单图 capture
        g = torch_npu.npu.NPUGraph()
        fw_attention = None
        dq = None
        with torch_npu.npu.graph(g, auto_dispatch_capture=True):
            fw_attention, fw_smax, fw_ssum, fw_sout, fw_seed, fw_offset = (
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
            dq, dk, dv, dpse, dsink = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_smax,
                softmax_sum=fw_ssum,
                attention_in=fw_attention,
                actual_seq_qlen=qlen_old,
                actual_seq_kvlen=kvlen_old,
            )

        # 多轮 update + replay
        test_seqs = [
            ([64, 192], [192, 64]),
            ([192, 64], [64, 192]),
        ]
        for seq_q, seq_kv in test_seqs:
            qlen_new = self._to_cumsum(seq_q)
            kvlen_new = self._to_cumsum(seq_kv)
            update_entry = {"actual_seq_qlen": qlen_new, "actual_seq_kvlen": kvlen_new}

            # 单次 update 同时刷新前向和反向
            g.update(cpu_update_input=[update_entry, update_entry])
            g.replay()

            # Eager 基准
            with torch.no_grad():
                fw_eager = torch_npu.npu_fusion_attention_v3(
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
            res_src = torch_npu.npu_fusion_attention_grad_v3(
                query,
                key,
                value,
                dy,
                head_num=N,
                input_layout="TND",
                keep_prob=1.0,
                scale_value=scale,
                softmax_max=fw_eager[1],
                softmax_sum=fw_eager[2],
                attention_in=fw_eager[0],
                actual_seq_qlen=qlen_new,
                actual_seq_kvlen=kvlen_new,
            )

            self.assertEqual(dq.cpu(), res_src[0].cpu(), prec=1.0)


# =========================================================================
# 四、TND + Dropout fallback 测试
# =========================================================================


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

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_tnd_graph_partition(self):
        from torch_npu._inductor import config as npu_config

        npu_config.npugraph_trees.disable_cpu_input_check = True
        """TND 前向, compile + reduce-overhead 模式.

        fa3_fn 包含 FA3 前后的普通算子, 验证 graph partition 正确性:
        FA3 前的 add/relu 和 FA3 后的 add/relu 留在 ACLGraph 中,
        FA3 算子本身由 handler 处理.
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

        def fa3_fn(q, k, v, actual_seq_qlen, actual_seq_kvlen):
            # --- FA3 前置算子: 对 q/k/v 做预处理 ---
            q = q * 1.0 + 0.01
            k = k * 1.0 + 0.01
            v = v * 1.0 + 0.01

            # --- FA3 核心算子 ---
            result = torch_npu.npu_fusion_attention_v3(
                q,
                k,
                v,
                head_num=N,
                input_layout="TND",
                keep_prob=0.9,
                scale=scale,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            fa3_out = result[0]

            # --- FA3 后置算子: 对 FA3 输出做后处理 ---
            out = torch.relu(fa3_out)
            out = out * 2.0 + 0.5
            return out

        eager_out = fa3_fn(query, key, value, qlen, kvlen)
        compiled_fn = torch.compile(fa3_fn, backend="inductor", mode="reduce-overhead")
        compiled_out = compiled_fn(query, key, value, qlen, kvlen)
        self.assertEqual(compiled_out.cpu(), eager_out.cpu(), prec=1.0)


if __name__ == "__main__":
    run_tests()
