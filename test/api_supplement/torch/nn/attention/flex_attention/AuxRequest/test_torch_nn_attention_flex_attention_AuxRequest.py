# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.flex_attention.AuxRequest 接口功能正确性
API 名称：torch.nn.attention.flex_attention.AuxRequest
API 签名：AuxRequest(lse: bool = False, max_scores: bool = False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认全 False 与显式请求字段两分支 | 已覆盖（auxrequest_defaults_false/auxrequest_construct_flags） |
| 枚举选项 | lse/max_scores 布尔组合：仅 lse、仅 max_scores、双 True | 已覆盖（auxrequest_construct_flags/
auxrequest_requests_aux_on_device） |
| 参数类型 | 字段为 bool，NamedTuple 元组语义（_fields/_replace/_asdict/相等性） | 已覆盖（auxrequest_namedtuple_semantics） |
| 传参与不传参 | 默认构造（全 False）与逐字段显式传参 | 已覆盖（auxrequest_defaults_false/auxrequest_construct_flags） |
| 等价类/边界值 | 未知关键字抛 TypeError | 已覆盖（auxrequest_invalid_keyword_raises） |
| 正常传参场景 | 设备端 flex_attention 按 AuxRequest 请求填充 AuxOutput 字段 | 已覆盖（auxrequest_requests_aux_on_device/
auxrequest_unrequested_fields_none_on_device） |
| 异常传参场景 | 非法字段名构造抛 TypeError | 已覆盖（auxrequest_invalid_keyword_raises） |
| 混合设备类型 | AuxRequest 为纯 Python 值对象，不涉及设备输入；设备行为由 flex_attention 组合场景验证 | 未覆盖（无设备输入路径，原因见文件头说明） |

未覆盖项及原因：
- 字段值不限于 bool 时（如传字符串）NamedTuple 不做类型校验，行为依赖调用方真值判断，无稳定
异常契约，不作断言。
- 混合设备类型：AuxRequest 本身不接收 Tensor，无混合设备输入路径；其驱动的 flex_attention
设备行为在本文件组合用例与 flex_attention 用例文件中验证。
- 设备门控：用例仅在 NPU 或 CUDA 可用时执行；两者都不可用时 DEVICE_NAME 置 None，整类在
类初始化前显式跳过（skipIf，基类 setUpClass 会先于 per-test setUp 选卡），不回退 cpu，避免
max_scores 请求用例触发 CPU 端 NotImplementedError、其余设备用例退化为 CPU 自比对。
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
# CPU execution, where requesting max_scores raises NotImplementedError and
# the device tensors would only self-compare.
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


def _default_scale(dim):
    return 1.0 / math.sqrt(dim)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
# Class-level guard: when torch_npu is installed but no accelerator exists,
# the torch_npu TestCase base class calls torch.npu.set_device in setUpClass
# before any per-test setUp could skip, so the whole class must be skipped
# before base-class initialization runs.
@unittest.skipIf(DEVICE_NAME is None, DEVICE_SKIP_REASON)
class TestFlexAttentionAuxRequest(TestCase):
    """Test cases for torch.nn.attention.flex_attention.AuxRequest (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        if self.device_name is None:
            # Without an accelerator there is no device path to exercise: skip
            # rather than run on CPU, where requesting max_scores raises
            # NotImplementedError and device tensors would only self-compare.
            self.skipTest(DEVICE_SKIP_REASON)
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # NamedTuple structural contract (device independent)
    # ------------------------------------------------------------------

    def test_auxrequest_defaults_false(self):
        """AuxRequest() defaults both flags to False."""
        request = fa.AuxRequest()
        self.assertIsInstance(request, fa.AuxRequest)
        self.assertIsInstance(request, tuple)
        self.assertFalse(request.lse)
        self.assertFalse(request.max_scores)
        self.assertEqual(request._fields, ('lse', 'max_scores'))
        self.assertEqual(request, (False, False))

    def test_auxrequest_construct_flags(self):
        """Explicit flag combinations are kept in field order."""
        self.assertEqual(fa.AuxRequest(lse=True), (True, False))
        self.assertEqual(fa.AuxRequest(max_scores=True), (False, True))
        both = fa.AuxRequest(lse=True, max_scores=True)
        self.assertEqual(both, (True, True))
        self.assertEqual(both._asdict(), {'lse': True, 'max_scores': True})

    def test_auxrequest_namedtuple_semantics(self):
        """_replace/_asdict/equality follow the NamedTuple contract."""
        request = fa.AuxRequest(lse=True)
        replaced = request._replace(lse=False, max_scores=True)
        self.assertEqual(replaced, (False, True))
        self.assertEqual(request, (True, False))
        self.assertEqual(len(request), 2)
        self.assertTrue(request[0])
        self.assertFalse(request[1])
        self.assertTrue(issubclass(fa.AuxRequest, tuple))

    def test_auxrequest_invalid_keyword_raises(self):
        """Unknown fields are rejected with TypeError."""
        with self.assertRaises(TypeError):
            fa.AuxRequest(bogus=True)

    # ------------------------------------------------------------------
    # Integration with flex_attention on the target device
    # ------------------------------------------------------------------

    def test_auxrequest_requests_aux_on_device(self):
        """Both flags drive flex_attention to fill both AuxOutput fields."""
        batch, heads, seq, dim = 1, 2, 8, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device, seed=7)
        out, aux = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(lse=True, max_scores=True),
        )
        self.assertIsInstance(aux, fa.AuxOutput)
        self.assertIsNotNone(aux.lse)
        self.assertIsNotNone(aux.max_scores)
        scores = (qkv[0][0].float() @ qkv[1][0].float().transpose(-2, -1)) * _default_scale(dim)
        torch.testing.assert_close(out.cpu(), scores.softmax(-1) @ qkv[2][0].float())
        torch.testing.assert_close(aux.lse.cpu(), scores.logsumexp(dim=-1))
        torch.testing.assert_close(aux.max_scores.cpu(), scores.max(dim=-1).values)

    def test_auxrequest_unrequested_fields_none_on_device(self):
        """A default AuxRequest keeps both AuxOutput fields None."""
        batch, heads, seq, dim = 1, 2, 8, 8
        qkv = _make_qkv(batch, heads, seq, dim, self.device, seed=8)
        out, aux = fa.flex_attention(
            qkv[0][1],
            qkv[1][1],
            qkv[2][1],
            return_aux=fa.AuxRequest(),
        )
        self.assertIsInstance(out, torch.Tensor)
        self.assertIsNone(aux.lse)
        self.assertIsNone(aux.max_scores)


if __name__ == "__main__":
    run_tests()
