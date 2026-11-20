# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.flex_attention.FlexKernelOptions 接口功能正确性
API 名称：torch.nn.attention.flex_attention.FlexKernelOptions
API 签名：class FlexKernelOptions(TypedDict, total=False)；18 个 NotRequired 键注解，作为 flex_attention(kernel_options=...) 的参数类型

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | kernel_options=None、{}、非空 TypedDict 实例 | 已覆盖（none_default/empty_dict/typeddict_instance_forward） |
| 枚举选项 | BACKEND 字面量 AUTO/TRITON/FLASH/TRITON_DECODE | 已覆盖（resolved_hints 断言字面量集合；逐值前向见 BACKEND 专项文件） |
| 参数类型 | 11 个 int 键、6 个 bool 键、1 个 Literal 键 | 已覆盖（resolved_hints） |
| 传参与不传参 | 缺省不传、空 dict、显式键集、fwd_/bwd_ 前缀键 | 已覆盖（none_default/empty_dict/typeddict_instance/fwd_bwd_prefixed） |
| 等价类/边界值 | total=False、无必选键、键不是类属性 | 已覆盖（typeddict_identity/keys_not_class_attributes） |
| 正常传参场景 | TypedDict/dict 传入与 fused 路径消费均对照 CPU Golden | 已覆盖（typeddict_instance_forward/inductor_compile_forward） |
| 异常传参场景 | FlexKernelOptions 本体不校验键值 | 未覆盖（异常契约在各键专项文件，如 BACKEND 的 ValueError/RuntimeError） |
| 混合设备类型 | q/k/v 需同设备 | 未覆盖（无 CPU/NPU 混合输入合法场景） |

未覆盖项及原因：
- 全 dtype 矩阵不覆盖：kernel_options 与 dtype 无关，仅补 bf16 前向（bf16_forward，容差为 bf16 表示误差）。
- forward 用例经公开 eager 入口执行（dynamo eager backend 分解，计算全部在设备上完成）；eager 分解不消费 kernel_options。
- kernel_options 的 fused kernel 消费由 inductor_compile_forward 覆盖：inductor 编译生成 fused flex kernel，
  数值对照 CPU Golden 与目标设备 eager 输出；该路径需 triton（NPU 为 triton-ascend），缺失时按缺依赖跳过。
- fused 路径的 shape 适配面不覆盖：NPU 上特定 shape（实测 (1,2,64,32) 编译失败、(1,2,128,64) 可用）的
  inductor 调度融合触发 SympifyError（torch_npu/_inductor 调度器限制，与 FlexKernelOptions 绑定无关）；
  inductor_compile_forward 取已验证可用的 (1,2,128,64)。
- kernel 级调参的性能效果不覆盖：BLOCK_M 等调参键不改变数值语义，性能对比不属功能用例范围。
- 内存维度不适用：TypedDict 参数无资源分配语义；确定性已覆盖（deterministic_fixed_seed）。
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
import typing_extensions


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
class TestFlexKernelOptions(TestCase):
    """Cases for the documented FlexKernelOptions contract."""

    INT_KEYS = (
        "BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2",
        "num_warps", "num_stages", "kpack", "matrix_instr_nonkdim", "waves_per_eu",
    )
    BOOL_KEYS = (
        "PRESCALE_QK", "ROWS_GUARANTEED_SAFE", "BLOCKS_ARE_CONTIGUOUS",
        "WRITE_DQ", "FORCE_USE_FLEX_ATTENTION", "USE_TMA",
    )
    BACKEND_ARGS = ("AUTO", "TRITON", "FLASH", "TRITON_DECODE")
    ALL_KEYS = INT_KEYS + BOOL_KEYS + ("BACKEND",)

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

    def test_flexkerneloptions_typeddict_identity(self):
        self.assertTrue(typing_extensions.is_typeddict(FlexKernelOptions))
        self.assertFalse(FlexKernelOptions.__total__)
        self.assertEqual(frozenset(), FlexKernelOptions.__required_keys__)
        self.assertEqual(set(self.ALL_KEYS), set(FlexKernelOptions.__optional_keys__))

    def test_flexkerneloptions_resolved_hints(self):
        hints = typing.get_type_hints(FlexKernelOptions)
        for key in self.INT_KEYS:
            self.assertIs(int, hints[key])
        for key in self.BOOL_KEYS:
            self.assertIs(bool, hints[key])
        self.assertIs(typing.Literal, typing.get_origin(hints["BACKEND"]))
        self.assertEqual(self.BACKEND_ARGS, typing.get_args(hints["BACKEND"]))
        self.assertEqual(set(self.ALL_KEYS), set(hints))

    def test_flexkerneloptions_keys_not_class_attributes(self):
        for key in self.ALL_KEYS:
            self.assertFalse(hasattr(FlexKernelOptions, key),
                             "TypedDict keys must not become class attributes")

    def test_flexkerneloptions_none_default_forward(self):
        out = flex_attention(self._q(), self._k(), self._v())
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_flexkerneloptions_empty_dict_forward(self):
        out = flex_attention(self._q(), self._k(), self._v(), kernel_options={})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_flexkerneloptions_typeddict_instance_forward(self):
        opts = FlexKernelOptions(BLOCK_M=64, BLOCK_N=64)
        self.assertIsInstance(opts, dict)
        self.assertEqual({"BLOCK_M": 64, "BLOCK_N": 64}, opts)
        out = flex_attention(self._q(), self._k(), self._v(), kernel_options=opts)
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden)

    def test_flexkerneloptions_fwd_bwd_prefixed_keys(self):
        q, k, v = self._device_inputs(requires_grad=True)
        grad_out = torch.randn(1, 2, 64, 32, dtype=torch.float32)
        opts = {"fwd_BLOCK_M": 64, "bwd_BLOCK_M1": 32, "PRESCALE_QK": False}
        out = flex_attention(q, k, v, kernel_options=opts)
        out.backward(grad_out.to(self.device_name))
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        qg, kg, vg = _golden_backward(self.q_cpu, self.k_cpu, self.v_cpu, grad_out)
        torch.testing.assert_close(out.float().cpu(), golden)
        torch.testing.assert_close(q.grad.float().cpu(), qg)
        torch.testing.assert_close(k.grad.float().cpu(), kg)
        torch.testing.assert_close(v.grad.float().cpu(), vg)

    def test_flexkerneloptions_bf16_forward(self):
        q = self.q_cpu.to(torch.bfloat16).to(self.device_name)
        k = self.k_cpu.to(torch.bfloat16).to(self.device_name)
        v = self.v_cpu.to(torch.bfloat16).to(self.device_name)
        out = flex_attention(q, k, v, kernel_options={"BLOCK_M": 32, "BLOCK_N": 32})
        golden = _golden_attention(self.q_cpu, self.k_cpu, self.v_cpu)
        torch.testing.assert_close(out.float().cpu(), golden, rtol=1.6e-2, atol=2e-2)

    def test_flexkerneloptions_deterministic_fixed_seed(self):
        q, k, v = self._device_inputs()
        out1 = flex_attention(q, k, v, kernel_options={"BLOCK_M": 32})
        out2 = flex_attention(q, k, v, kernel_options={"BLOCK_M": 32})
        self.assertTrue(torch.equal(out1, out2))

    def test_flexkerneloptions_inductor_compile_forward(self):
        # The fused inductor path consumes kernel_options when generating the
        # flex kernel; the eager decomposition used by the forward cases above
        # does not. This case uses its own shape (1, 2, 128, 64): on NPU the
        # fused path is currently shape-sensitive and the shared (1, 2, 64, 32)
        # fixture hits an inductor scheduler SympifyError during epilogue
        # fusion (inductor/torch_npu scheduler limitation, unrelated to the
        # FlexKernelOptions binding itself).
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton is not available (missing dependency)")
        q_cpu = torch.randn(1, 2, 128, 64, dtype=torch.float32)
        k_cpu = torch.randn(1, 2, 128, 64, dtype=torch.float32)
        v_cpu = torch.randn(1, 2, 128, 64, dtype=torch.float32)
        q = q_cpu.to(torch.bfloat16).to(self.device_name)
        k = k_cpu.to(torch.bfloat16).to(self.device_name)
        v = v_cpu.to(torch.bfloat16).to(self.device_name)
        opts = FlexKernelOptions(BLOCK_M=64, BLOCK_N=64, num_warps=4)
        out_eager = flex_attention(q, k, v, kernel_options=opts)
        compiled = torch.compile(
            flex_attention, backend="inductor", fullgraph=True, dynamic=False
        )
        out_fused = compiled(q, k, v, kernel_options=opts)
        golden = _golden_attention(q_cpu, k_cpu, v_cpu)
        torch.testing.assert_close(
            out_fused.float().cpu(), golden, rtol=1.6e-2, atol=2e-2
        )
        torch.testing.assert_close(
            out_fused.float(), out_eager.float(), rtol=1.6e-2, atol=2e-2
        )


if __name__ == "__main__":
    run_tests()
