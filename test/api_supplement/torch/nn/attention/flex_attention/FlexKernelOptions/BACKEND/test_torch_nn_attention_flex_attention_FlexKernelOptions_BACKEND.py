# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.flex_attention.FlexKernelOptions.BACKEND 键功能正确性
API 名称：torch.nn.attention.flex_attention.FlexKernelOptions.BACKEND
API 签名：BACKEND: NotRequired[Literal["AUTO", "TRITON", "FLASH", "TRITON_DECODE"]]
（FlexKernelOptions TypedDict 的 NotRequired 键注解，经 kernel_options dict 传入 flex_attention，不是运行时类属性）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 键缺省（默认 AUTO 语义） | 键缺省路径属 FlexKernelOptions 本体（主文件 none_default_forward） |
| 枚举选项 | AUTO/TRITON/TRITON_DECODE/FLASH 逐值 | 已覆盖（forward_auto/triton/triton_decode/flash_eager_path） |
| 参数类型 | 非法取值 BOGUS 抛 ValueError | 已覆盖（invalid_value_raises） |
| 传参与不传参 | 单独传 BACKEND、与 FORCE_USE_FLEX_ATTENTION 组合 | 已覆盖（forward_*、conflicts_force_flag_raises） |
| 等价类/边界值 | 合法集合外的取值 | 已覆盖（invalid_value_raises） |
| 正常传参场景 | 各合法值前向输出对照 CPU Golden | 已覆盖（forward_auto/triton/triton_decode/flash_eager_path） |
| 异常传参场景 | BACKEND+FORCE 组合抛 RuntimeError | 已覆盖（conflicts_force_flag_raises） |
| 混合设备类型 | q/k/v 需同设备 | 未覆盖（无 CPU/NPU 混合输入合法场景） |

未覆盖项及原因：
- BACKEND="FLASH" 的 CUTE fused kernel 与各 BACKEND 的 kernel 分派差异：fused inductor 路径需 triton-ascend（本环境未安装），无法观察分派差异。
- eager 分解路径下各值被接受且数值正确（forward_flash_eager_path）。
- NPU 执行方式：本环境未安装 triton-ascend，fused inductor 路径不可用；用例经公开 eager 入口（dynamo eager backend 分解执行，全部计算在设备上完成）。
- 已验证内容为键的接受性、约束契约与数值正确性；kernel 级调参效果属 fused 路径行为，本环境不可观察。
"""
import math
import os
import typing
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Device probing happens only when the supplement tests really run. Plain CI
# collection skips these tests before any device import or init happens.
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = 'cuda'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

from torch.nn.attention.flex_attention import FlexKernelOptions, flex_attention


def _golden_attention(query, key, value, mask=None):
    """Reference attention computed with fp32 math (device-independent)."""
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value.float())


def _golden_backward(query, key, value, grad_out):
    """Reference gradients from a manual CPU autograd graph."""
    q = query.clone().requires_grad_(True)
    k = key.clone().requires_grad_(True)
    v = value.clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    out.backward(grad_out.float())
    return q.grad, k.grad, v.grad


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFlexKernelOptionsBackend(TestCase):
    """Cases for the documented FlexKernelOptions contract."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)
        torch.manual_seed(0)
        self.q_cpu = torch.randn(1, 2, 64, 32, dtype=torch.float32)
        self.k_cpu = torch.randn(1, 2, 64, 32, dtype=torch.float32)
        self.v_cpu = torch.randn(1, 2, 64, 32, dtype=torch.float32)

    def _q(self):
        return self.q_cpu.to(self.device_name)

    def _k(self):
        return self.k_cpu.to(self.device_name)

    def _v(self):
        return self.v_cpu.to(self.device_name)

    def _device_inputs(self, requires_grad=False):
        q = self.q_cpu.to(self.device_name).requires_grad_(requires_grad)
        k = self.k_cpu.to(self.device_name).requires_grad_(requires_grad)
        v = self.v_cpu.to(self.device_name).requires_grad_(requires_grad)
        return q, k, v

    def test_backend_key_metadata(self):
        self.assertIn("BACKEND", FlexKernelOptions.__annotations__)
        self.assertIn("BACKEND", FlexKernelOptions.__optional_keys__)
        self.assertNotIn("BACKEND", FlexKernelOptions.__required_keys__)
        hints = typing.get_type_hints(FlexKernelOptions)
        self.assertIs(typing.Literal, typing.get_origin(hints["BACKEND"]))
        self.assertEqual(("AUTO", "TRITON", "FLASH", "TRITON_DECODE"),
                         typing.get_args(hints["BACKEND"]))

    def test_backend_forward_auto(self):
        out = flex_attention(self._q(), self._k(), self._v(),
                             kernel_options={"BACKEND": "AUTO"})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_backend_forward_triton(self):
        out = flex_attention(self._q(), self._k(), self._v(),
                             kernel_options={"BACKEND": "TRITON"})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_backend_forward_triton_decode(self):
        out = flex_attention(self._q(), self._k(), self._v(),
                             kernel_options={"BACKEND": "TRITON_DECODE"})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_backend_forward_flash_eager_path(self):
        out = flex_attention(self._q(), self._k(), self._v(),
                             kernel_options={"BACKEND": "FLASH"})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_backend_invalid_value_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            flex_attention(self._q(), self._k(), self._v(),
                           kernel_options={"BACKEND": "BOGUS"})

    def test_backend_conflicts_force_flag_raises(self):
        with self.assertRaisesRegex(RuntimeError, "BACKEND cannot be combined"):
            flex_attention(self._q(), self._k(), self._v(),
                           kernel_options={"BACKEND": "AUTO",
                                           "FORCE_USE_FLEX_ATTENTION": True})


if __name__ == "__main__":
    run_tests()
