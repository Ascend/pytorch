# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Unflatten.extra_repr 接口功能正确性
API 名称：torch.nn.modules.flatten.Unflatten.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；tuple/list/torch.Size 三种存储形态精确断言 | 已覆盖（tuple_size/list_size/size_obj） |
| 枚举选项 | dim 取正数、0、负数多个合法值组合 | 已覆盖（test_extra_repr_tuple_size/test_extra_repr_list_size/test_extra_repr_negative_dim） |
| 参数类型 | 返回值必须为 str 类型；容器类型决定输出格式 | 已覆盖（test_extra_repr_returns_str/test_extra_repr_size_obj） |
| 传参与不传参 | extra_repr 无参；字符串读取构造时保存的 dim/unflattened_size | 已覆盖（test_extra_repr_tuple_size/test_extra_repr_list_size） |
| 等价类/边界值 | 属性被修改后字符串同步反映；torch.Size 输出带类型前缀 | 已覆盖（test_extra_repr_reflects_mutated_dim/test_extra_repr_size_obj） |
| 正常传参场景 | repr(module) 以 "Unflatten(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 无入参校验路径，无稳定异常契约 | 未覆盖（无参方法不存在非法传参路径） |
| 混合设备类型 | 设备侧构造的 Unflatten 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- 异常传参场景不适用：extra_repr 无参数、无校验分支，不存在可稳定断言的异常路径。
- 版本行为说明：extra_repr 输出 f"dim={dim}, unflattened_size={size}"；
  torch.Size 的 repr 带 torch.Size([...]) 前缀，list 带 [] 括号，tuple 带 () 括号。
- 本文件看护 torch.nn.modules.flatten.Unflatten.extra_repr（正名路径）；
  Unflatten 模块对象与 forward 契约分别在同名模块对象/forward 用例文件中覆盖。
- torch.nn.Unflatten 与 torch.nn.modules.flatten.Unflatten 为同一类对象，模块对象文件断言。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     设备侧仅验证构造与表示行为，无浮点数值输出，不涉及浮点 Golden 对照。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

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
class TestUnflattenExtraRepr(TestCase):
    """Cases for torch.nn.modules.flatten.Unflatten.extra_repr (NPU/CUDA)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for tuple/list/torch.Size storage forms."""
        self.assertIsInstance(
            torch.nn.modules.flatten.Unflatten(1, (2, 5, 5)).extra_repr(), str
        )
        self.assertIsInstance(
            torch.nn.modules.flatten.Unflatten(0, [4, 3]).extra_repr(), str
        )
        self.assertIsInstance(
            torch.nn.modules.flatten.Unflatten(
                1, torch.Size([2, 5, 5])
            ).extra_repr(),
            str,
        )

    def test_extra_repr_tuple_size(self):
        """Tuple storage reports dim and the tuple repr."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        self.assertEqual(m.extra_repr(), "dim=1, unflattened_size=(2, 5, 5)")

    def test_extra_repr_list_size(self):
        """List storage reports dim and the list repr."""
        m = torch.nn.modules.flatten.Unflatten(0, [4, 3])
        self.assertEqual(m.extra_repr(), "dim=0, unflattened_size=[4, 3]")

    def test_extra_repr_size_obj(self):
        """torch.Size storage reports the torch.Size([...]) repr verbatim."""
        m = torch.nn.modules.flatten.Unflatten(-1, torch.Size([2, 5, 5]))
        self.assertEqual(
            m.extra_repr(), "dim=-1, unflattened_size=torch.Size([2, 5, 5])"
        )

    def test_extra_repr_negative_dim(self):
        """Negative dim is reported as stored."""
        m = torch.nn.modules.flatten.Unflatten(-1, (2, 25))
        self.assertEqual(m.extra_repr(), "dim=-1, unflattened_size=(2, 25)")

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        self.assertEqual(
            repr(m), "Unflatten(dim=1, unflattened_size=(2, 5, 5))"
        )

    def test_extra_repr_reflects_mutated_dim(self):
        """extra_repr reads the current attribute values at call time."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        m.dim = 3
        self.assertEqual(
            m.extra_repr(), "dim=3, unflattened_size=(2, 5, 5)"
        )

    def test_extra_repr_device_consistency(self):
        """Device-constructed Unflatten has an identical representation string."""
        cpu_m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        device_m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5)).to(
            self.device
        )
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(
            device_m.extra_repr(), "dim=1, unflattened_size=(2, 5, 5)"
        )


if __name__ == "__main__":
    run_tests()
