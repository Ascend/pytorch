# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.checkpoint.create_selective_checkpoint_contexts 接口功能正确性
API 名称：torch.utils.checkpoint.create_selective_checkpoint_contexts
API 签名：create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空算子列表与非空算子列表             | 已覆盖（empty_op_list_and_allow_mutation、op_list_checkpoint_matches_eager） |
| 枚举选项         | CheckpointPolicy 枚举返回策略        | 已覆盖（callable_policy_checkpoint）               |
| 参数类型         | OpOverload 列表、可调用策略与非法整型 | 已覆盖（op_list_checkpoint_matches_eager、callable_policy_checkpoint 等） |
| 传参与不传参     | 可选布尔参数默认与显式传入           | 已覆盖（op_list_checkpoint_matches_eager、empty_op_list_and_allow_mutation） |
| 等价类/边界值    | 空列表、OpOverloadPacket 非法元素    | 已覆盖（empty_op_list_and_allow_mutation、invalid_args_raise） |
| 正常传参场景     | 配合 checkpoint 前反向与 eager 对齐  | 已覆盖（op_list_checkpoint_matches_eager、callable_policy_checkpoint） |
| 异常传参场景     | 非法列表元素、错误类型及缺参报错     | 已覆盖（invalid_args_raise）                       |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：接口仅构造选择性重计算上下文管理器，不直接接收多设备 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import functools
import os
import unittest

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy, SelectiveCheckpointContext, checkpoint, create_selective_checkpoint_contexts,
)

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _sample_fn(x, y):
    return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateSelectiveCheckpointContexts(TestCase):
    """Test cases for torch.utils.checkpoint.create_selective_checkpoint_contexts."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_op_list_checkpoint_matches_eager(self):
        """Selective checkpoint with an op list matches eager forward output and gradients."""
        torch.manual_seed(0)
        x_base = torch.randn(4, 4, device=self.device)
        y_base = torch.randn(4, 4, device=self.device)
        x_e, y_e = x_base.clone().requires_grad_(True), y_base.clone().requires_grad_(True)
        out_e = _sample_fn(x_e, y_e)
        out_e.sum().backward()
        ops_to_save = [torch.ops.aten.mm.default]
        ctx_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        x_c, y_c = x_base.clone().requires_grad_(True), y_base.clone().requires_grad_(True)
        out_c = checkpoint(_sample_fn, x_c, y_c, use_reentrant=False, context_fn=ctx_fn)
        out_c.sum().backward()
        torch.testing.assert_close(out_c.cpu(), out_e.cpu())
        torch.testing.assert_close(x_c.grad.cpu(), x_e.grad.cpu())
        torch.testing.assert_close(y_c.grad.cpu(), y_e.grad.cpu())

    def test_callable_policy_checkpoint(self):
        """Callable policy receives SelectiveCheckpointContext and produces valid gradients."""
        seen_flags = []

        def policy_fn(ctx, op, *args, **kwargs):
            self.assertIsInstance(ctx, SelectiveCheckpointContext)
            seen_flags.append(ctx.is_recompute)
            return CheckpointPolicy.MUST_SAVE if op == torch.ops.aten.mm.default else CheckpointPolicy.PREFER_RECOMPUTE
        ctx_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        x = torch.randn(4, 4, device=self.device, requires_grad=True)
        y = torch.randn(4, 4, device=self.device, requires_grad=True)
        out = checkpoint(lambda a, b: torch.sin(torch.sin(torch.mm(a, b))), x, y,
                         use_reentrant=False, context_fn=ctx_fn)
        out.sum().backward()
        self.assertIn(False, seen_flags)
        self.assertGreaterEqual(len(seen_flags), 2)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_empty_op_list_and_allow_mutation(self):
        """Empty op list with allow_cache_entry_mutation=True configures recompute context."""
        pair = create_selective_checkpoint_contexts([], allow_cache_entry_mutation=True)
        self.assertIsInstance(pair, tuple)
        self.assertEqual(len(pair), 2)
        self.assertTrue(pair[1].allow_cache_entry_mutation)

    def test_invalid_args_raise(self):
        """Invalid op packet, non-callable/non-list arg, or missing arg raises errors."""
        with self.assertRaises(ValueError):
            create_selective_checkpoint_contexts([torch.ops.aten.mm])
        with self.assertRaises(TypeError):
            create_selective_checkpoint_contexts(123)
        with self.assertRaises(TypeError):
            create_selective_checkpoint_contexts()


if __name__ == "__main__":
    run_tests()
