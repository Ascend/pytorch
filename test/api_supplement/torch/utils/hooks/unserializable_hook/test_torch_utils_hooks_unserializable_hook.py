# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hooks.unserializable_hook 接口功能正确性
API 名称：torch.utils.hooks.unserializable_hook
API 签名：torch.utils.hooks.unserializable_hook(f)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空容器/空输入场景                    | 未覆盖（装饰器接收单个可调用对象，无容器输入）     |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 函数、lambda、可调用类实例与不可设属 | 已覆盖（basic_decorator、callable_types、          |
|                  | 性对象                               | invalid_object_raises）                            |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_decorator、missing_arg_raises）      |
| 等价类/边界值    | 幂等重复装饰与屏蔽 hook 序列化警告   | 已覆盖（idempotent_redecoration、                  |
|                  |                                      | suppresses_hook_warning）                          |
| 正常传参场景     | 标记 __torch_unserializable__ 属性   | 已覆盖（basic_decorator、callable_types、          |
|                  |                                      | suppresses_hook_warning）                          |
| 异常传参场景     | 缺参与不可设置属性对象异常           | 已覆盖（missing_arg_raises、invalid_object_raises）|
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（hook 装饰器不涉及多 Tensor 设备混合）      |

未覆盖项及原因：
- 空/非空：该接口为接收单个可调用对象 f 的装饰器，不存在空容器或空字符串概念。
- 枚举选项：该接口仅接收单个目标对象参数，无枚举类参数。
- 混合设备类型：该接口仅在 Python 对象上设置属性标记，不涉及多 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
import warnings

import torch
from torch.utils.hooks import unserializable_hook, warn_if_has_hooks

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestUnserializableHook(TestCase):
    """Test cases for torch.utils.hooks.unserializable_hook."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_decorator(self):
        """Marks a regular function with __torch_unserializable__ and preserves behavior."""
        def my_hook(grad):
            return grad * 2

        decorated = unserializable_hook(my_hook)
        self.assertIs(decorated, my_hook)
        self.assertTrue(getattr(decorated, "__torch_unserializable__", False))
        self.assertEqual(decorated(3), 6)

    def test_callable_types(self):
        """Supports decorating both lambdas and callable class instances."""
        lam = unserializable_hook(lambda x: x + 1)
        self.assertTrue(getattr(lam, "__torch_unserializable__", False))
        self.assertEqual(lam(4), 5)

        class CallableHook:
            def __call__(self, x):
                return x - 1

        obj = CallableHook()
        decorated_obj = unserializable_hook(obj)
        self.assertIs(decorated_obj, obj)
        self.assertTrue(getattr(decorated_obj, "__torch_unserializable__", False))
        self.assertEqual(decorated_obj(4), 3)

    def test_idempotent_redecoration(self):
        """Applying unserializable_hook repeatedly returns the same marked function."""
        def fn(x):
            return x

        first = unserializable_hook(fn)
        second = unserializable_hook(first)
        self.assertIs(second, fn)
        self.assertTrue(getattr(second, "__torch_unserializable__", False))

    def test_suppresses_hook_warning(self):
        """Decorated backward hook suppresses warnings emitted by warn_if_has_hooks."""
        t_plain = torch.randn(2, requires_grad=True)
        t_plain.register_hook(lambda g: g)
        with warnings.catch_warnings(record=True) as caught_plain:
            warnings.simplefilter("always")
            warn_if_has_hooks(t_plain)
        self.assertGreaterEqual(len(caught_plain), 1)

        t_marked = torch.randn(2, requires_grad=True)
        t_marked.register_hook(unserializable_hook(lambda g: g))
        with warnings.catch_warnings(record=True) as caught_marked:
            warnings.simplefilter("always")
            warn_if_has_hooks(t_marked)
        self.assertEqual(len(caught_marked), 0)

    def test_missing_arg_raises(self):
        """Calling unserializable_hook without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            unserializable_hook()

    def test_invalid_object_raises(self):
        """Passing an object that cannot accept attributes raises AttributeError."""
        with self.assertRaises(AttributeError):
            unserializable_hook(123)


if __name__ == "__main__":
    run_tests()
