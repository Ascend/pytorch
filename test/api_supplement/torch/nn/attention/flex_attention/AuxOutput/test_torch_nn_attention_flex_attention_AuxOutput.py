# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.flex_attention.AuxOutput 接口功能正确性
API 名称：torch.nn.attention.flex_attention.AuxOutput
API 签名：AuxOutput(lse: Tensor | None = None, max_scores: Tensor | None = None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认字段为 None 与显式传入 Tensor 两分支 | 已覆盖（auxoutput_default_fields_none/
auxoutput_construct_with_values） |
| 枚举选项 | return_aux 字段组合：仅 lse、仅 max_scores、双请求、均不请求 | 已覆盖（auxoutput_from_flex_attention_lse_on_device/
auxoutput_from_flex_attention_max_scores_on_device/
auxoutput_from_flex_attention_both_fields_on_device/
auxoutput_unrequested_fields_none_on_device） |
| 参数类型 | 字段类型 Tensor | None，NamedTuple 元组语义（_fields/_asdict/相等性） | 已覆盖（auxoutput_namedtuple_semantics） |
| 传参与不传参 | 默认构造（全 None）与经 flex_attention return_aux 填充 | 已覆盖（auxoutput_default_fields_none/
auxoutput_from_flex_attention_both_fields_on_device） |
| 等价类/边界值 | return_aux=AuxRequest() 未请求任何字段时输出仍为二元组且字段均为 None | 已覆盖（auxoutput_unrequested_fields_none_on_device） |
| 正常传参场景 | 设备端 flex_attention 返回的 lse/max_scores 数值与参考实现一致（(B,H,Mq) 形状） | 已覆盖（auxoutput_from_flex_attention_*_on_device） |
| 异常传参场景 | query/key/value 设备不一致时抛 ValueError，AuxOutput 不产出 | 已覆盖（auxoutput_mixed_device_input_rejected） |
| 混合设备类型 | q/k/v 混合设备/ CPU）输入被设备校验拒绝 | 已覆盖（auxoutput_mixed_device_input_rejected） |

未覆盖项及原因：
- max_scores 的 CPU flex Golden：CPU 端 flex_attention 返回 max_scores 抛 NotImplementedError
（Returning max scores is not supported on CPU.），max_scores 数值采用 CPU 手工参考实现对比。
- AuxOutput 仅由 flex_attention 在 return_aux 非 None 时产出，无独立设备执行路径；其数值正确性
通过 flex_attention 组合场景在设备端与参考实现对比完成。
- 设备门控：用例仅在 NPU 或 CUDA 可用时执行；两者都不可用时 DEVICE_NAME 置 None，整类在
类初始化前显式跳过（skipIf，基类 setUpClass 会先于 per-test setUp 选卡），不回退 cpu，避免
混合设备断言失去设备差异、设备端组合用例退化为 CPU 自比对。
"""
import math
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement tests actually run. Plain CI
# collection skips the class before any torch_npu import or device init.
# When neither NPU nor CUDA is available, DEVICE_NAME stays None: the device
# cases are then skipped explicitly in setUp instead of silently degrading to
# CPU-only self-comparisons or losing the mixed-device contract.
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else None

DEVICE_SKIP_REASON = (
    "no NPU/CUDA device available; these supplement cases require an accelerator"
)

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


import torch.nn.attention.flex_attention as fa  # noqa: E402


def _make_qkv(batch, heads, seq, dim, device, dtype=torch.float32, seed=0):
    """Create identical q/k/v on CPU and the target device from one seed."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    tensors = []
    for _ in range(3):
        cpu_tensor = torch.randn(batch, heads, seq, dim, generator=gen, dtype=dtype)
        tensors.append((cpu_tensor, cpu_tensor.to(device)))
    return tensors


def _reference_scores(q_cpu, k_cpu, scale):
    """Scaled attention scores as an fp32 CPU reference."""
    return (q_cpu.float() @ k_cpu.float().transpose(-2, -1)) * scale


def _default_scale(dim):
    return 1.0 / math.sqrt(dim)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
# Class-level guard: when torch_npu is installed but no accelerator exists,
# the torch_npu TestCase base class calls torch.npu.set_device in setUpClass
# before any per-test setUp could skip, so the whole class must be skipped
# before base-class initialization runs.
@unittest.skipIf(DEVICE_NAME is None, DEVICE_SKIP_REASON)
class TestFlexAttentionAuxOutput(TestCase):
    """Test cases for torch.nn.attention.flex_attention.AuxOutput (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        if self.device_name is None:
            # Without an accelerator there is no device path to exercise: skip
            # rather than run on CPU, where mixed-device inputs would not differ
            # and device-filled AuxOutput fields would only self-compare.
            self.skipTest(DEVICE_SKIP_REASON)
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # NamedTuple structural contract (device independent)
    # ------------------------------------------------------------------

    def test_auxoutput_default_fields_none(self):
        """AuxOutput() leaves both fields None and keeps NamedTuple semantics."""
        aux = fa.AuxOutput()
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsInstance(aux, tuple)
        self.assertIsNone(aux.lse)
        self.assertIsNone(aux.max_scores)
        self.assertEqual(aux._fields, ('lse', 'max_scores'))
        self.assertEqual(aux, fa.AuxOutput(None, None))
        self.assertEqual(aux._asdict(), {'lse': None, 'max_scores': None})

    def test_auxoutput_construct_with_values(self):
        """Explicit construction keeps the given tensors in field order."""
        lse = torch.tensor([1.0, 2.0])
        max_scores = torch.tensor([0.5, 0.25])
        aux = fa.AuxOutput(lse=lse, max_scores=max_scores)
        self.assertIs(aux.lse, lse)
        self.assertIs(aux.max_scores, max_scores)
        self.assertIs(aux[0], lse)
        self.assertIs(aux[1], max_scores)
        replaced = aux._replace(lse=None)
        self.assertIsNone(replaced.lse)
        self.assertIs(replaced.max_scores, max_scores)

    def test_auxoutput_namedtuple_semantics(self):
        """AuxOutput is a distinct NamedTuple from the request class."""
        self.assertTrue(issubclass(fa.AuxOutput, tuple))
        self.assertIsNot(fa.AuxOutput, fa.AuxRequest)
        aux = fa.AuxOutput(lse=torch.zeros(1))
        self.assertIsInstance(aux, tuple)
        self.assertEqual(len(aux), 2)

    # ------------------------------------------------------------------
    # Integration with flex_attention on the target device
    # ------------------------------------------------------------------

    def test_auxoutput_from_flex_attention_lse_on_device(self):
        """return_aux lse=True fills aux.lse matching the CPU flex golden."""
        batch, heads, seq, dim = 2, 2, 16, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device)
        out, aux = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(lse=True),
        )
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsNotNone(aux.lse)
        self.assertIsNone(aux.max_scores)
        self.assertEqual(aux.lse.shape, (batch, heads, seq))
        cpu_out, cpu_aux = fa.flex_attention(
            qkv[0][0],
            qkv[1][0],
            qkv[2][0],
            return_aux=fa.AuxRequest(lse=True),
        )
        torch.testing.assert_close(out.cpu(), cpu_out)
        torch.testing.assert_close(aux.lse.cpu(), cpu_aux.lse)

    def test_auxoutput_from_flex_attention_max_scores_on_device(self):
        """return_aux max_scores=True fills aux.max_scores vs manual reference."""
        batch, heads, seq, dim = 2, 2, 16, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device)
        out, aux = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(max_scores=True),
        )
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsNone(aux.lse)
        self.assertIsNotNone(aux.max_scores)
        self.assertEqual(aux.max_scores.shape, (batch, heads, seq))
        scores = _reference_scores(qkv[0][0], qkv[1][0], _default_scale(dim))
        torch.testing.assert_close(out.cpu(), scores.softmax(-1) @ qkv[2][0].float())
        torch.testing.assert_close(aux.max_scores.cpu(), scores.max(dim=-1).values)

    def test_auxoutput_from_flex_attention_both_fields_on_device(self):
        """Requesting both aux fields returns both populated tensors."""
        batch, heads, seq, dim = 1, 2, 8, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device, seed=3)
        out, aux = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(lse=True, max_scores=True),
        )
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsNotNone(aux.lse)
        self.assertIsNotNone(aux.max_scores)
        scores = _reference_scores(qkv[0][0], qkv[1][0], _default_scale(dim))
        torch.testing.assert_close(out.cpu(), scores.softmax(-1) @ qkv[2][0].float())
        torch.testing.assert_close(aux.lse.cpu(), scores.logsumexp(dim=-1))
        torch.testing.assert_close(aux.max_scores.cpu(), scores.max(dim=-1).values)

    def test_auxoutput_unrequested_fields_none_on_device(self):
        """AuxRequest() without flags still returns the (out, AuxOutput) pair."""
        batch, heads, seq, dim = 1, 2, 8, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device, seed=4)
        result = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(),
        )
        self.assertIsInstance(result, tuple)
        out, aux = result
        self.assertIsInstance(out, torch.Tensor)
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsNone(aux.lse)
        self.assertIsNone(aux.max_scores)

    def test_auxoutput_mixed_device_input_rejected(self):
        """Mixed-device q/k/v inputs are rejected before any AuxOutput is built."""
        batch, heads, seq, dim = 1, 2, 8, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device, seed=5)
        with self.assertRaises(ValueError):
            fa.flex_attention(
                qkv[0][1],
                qkv[1][0],
                qkv[2][0],
                return_aux=fa.AuxRequest(lse=True),
            )


if __name__ == "__main__":
    run_tests()
