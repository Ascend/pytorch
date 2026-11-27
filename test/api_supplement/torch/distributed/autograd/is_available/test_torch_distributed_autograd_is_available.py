# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.autograd.is_available 接口功能正确性
API 名称：torch.distributed.autograd.is_available
API 签名：torch.distributed.autograd.is_available() -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 空/非空        | 无参函数调用 | 已覆盖（test_return_bool_and_module_exports） |
| 枚举选项       | 无枚举入参 | 未覆盖，无参查询接口，不接收枚举参数 |
| 参数类型       | 无参调用与非法多余参数 | 已覆盖（test_return_bool_and_module_exports、test_unexpected_args_raises） |
| 传参与不传参   | 无参调用与传入多余位置/关键字参数 | 已覆盖（test_return_bool_and_module_exports、test_unexpected_args_raises） |
| 等价类/边界值  | 返回值与 torch._C._dist_autograd_init 及模块导出一致性 | 已覆盖（test_return_bool_and_module_exports） |
| 正常传参场景   | 精确布尔返回值、幂等性、不改变分布式初始化状态 | 已覆盖（test_return_bool_and_module_exports、test_idempotent_and_no_side_effect） |
| 异常传参场景   | 传入多余位置参数或关键字参数触发 TypeError | 已覆盖（test_unexpected_args_raises） |
| 混合设备类型   | 多 Tensor 输入场景 | 未覆盖，纯编译能力查询接口，不接收 Tensor 输入 |

未覆盖项及原因：
- 枚举选项：torch.distributed.autograd.is_available 为无参布尔查询函数。
- 混合设备类型：接口不接收 Tensor 参数，不涉及设备迁移。

注意：本接口查询当前构建是否启用分布式自动求导（hasattr(torch._C, "_dist_autograd_init")），
     单进程测试断言其精确布尔取值与 C++ 底层标志及模块导出符号严格一致。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        DEVICE_NAME = "npu" if torch.npu.is_available() else None
    except (ImportError, AttributeError):
        DEVICE_NAME = None
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributedAutogradIsAvailable(TestCase):

    def test_return_bool_and_module_exports(self):
        result = dist_autograd.is_available()
        self.assertIsInstance(result, bool)
        self.assertEqual(result, hasattr(torch._C, "_dist_autograd_init"))
        for sym in ("DistAutogradContext", "backward", "get_gradients", "_new_context"):
            self.assertEqual(
                hasattr(dist_autograd, sym),
                result,
                f"Expected hasattr(dist_autograd, '{sym}') == {result}",
            )
        self.assertTrue(hasattr(dist_autograd, "context"))

    def test_idempotent_and_no_side_effect(self):
        before_init = dist.is_initialized()
        first = dist_autograd.is_available()
        second = dist_autograd.is_available()
        self.assertIs(first, second)
        self.assertEqual(before_init, dist.is_initialized())

    def test_unexpected_args_raises(self):
        with self.assertRaises(TypeError):
            dist_autograd.is_available(1)
        with self.assertRaises(TypeError):
            dist_autograd.is_available(flag=True)


if __name__ == "__main__":
    run_tests()
