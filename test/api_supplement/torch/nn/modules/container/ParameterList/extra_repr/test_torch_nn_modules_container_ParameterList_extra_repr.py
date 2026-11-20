# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.container.ParameterList.extra_repr 接口功能正确性
API 名称：torch.nn.modules.container.ParameterList.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度         | 说明                                        | 覆盖情况                            |
|------------------|---------------------------------------------|-------------------------------------|
| 空/非空          | 空列表返回空字符串；非空列表逐项生成条目行  | 已覆盖（extra_repr_empty_list/extra_repr_parameter_entries） |
| 枚举选项         | 条目形态枚举：Parameter/原始 Tensor/None/非 Tensor 对象 | 已覆盖（extra_repr_raw_tensor_wrapped/extra_repr_non_tensor_entry） |
| 参数类型         | 返回 str；原始 Tensor 存入时被 __setitem__ 包装为 Parameter 后按 Parameter 行显示 | 已覆盖（extra_repr_raw_tensor_wrapped） |
| 传参与不传参     | 无参方法                                    | 已覆盖（全部用例无参调用）          |
| 等价类/边界值     | 空列表为空串；切片取子列表后重新编号渲染    | 已覆盖（extra_repr_empty_list/extra_repr_slice_renumbers） |
| 正常传参场景     | 条目格式为 "  (k): Parameter containing: [dtype of size ...]"；
|                  | repr(ParameterList) 内嵌这些行且渲染稳定 | 已覆盖（extra_repr_parameter_entries） |
| 异常传参场景     | 无参方法，无异常入参路径                    | 未覆盖（无参方法，无异常入参路径）  |
| 混合设备类型     | NPU 驻留参数条目附带 (npu:0) 设备串；CPU 驻留不带设备串 | 已覆盖（extra_repr_device_string）  |

未覆盖项及原因：
- 无。本接口为纯字符串表示方法，无设备计算路径；设备驻留条目的设备串在
  extra_repr_device_string 中覆盖（设备串来自 p.device，CUDA 环境下为 cuda
  索引串）。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
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
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestParameterListextra_repr(TestCase):
    """Test cases for torch.nn.modules.container.ParameterList.extra_repr.

    ParameterList.extra_repr renders one line per entry (torch 2.15.0,
    torch/nn/modules/container.py). Empirical contract (torch 2.15.0+cpu +
    torch_npu 2.15.0.dev20261118, probed on Ascend910B4):
      - an empty ParameterList yields '';
      - tensor entries render as "  (k): Parameter containing:
        [torch.float32 of size WxH]"; raw tensors appended via append() are
        wrapped into Parameters by ParameterList.__setitem__, so they show
        as Parameter lines too;
      - non-tensor entries render as "  (k): Object of type: <TypeName>";
      - entries resident on cuda/privateuse1 devices append " (device)";
        CPU-resident entries carry no device suffix;
      - the rendering feeds Module.__repr__ and is stable across calls;
      - torch.nn.modules.container.ParameterList is the public alias class.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_extra_repr_empty_list(self):
        """An empty ParameterList has an empty extra_repr."""
        pl = torch.nn.modules.container.ParameterList()
        self.assertEqual(pl.extra_repr(), '',
                         "empty ParameterList must render an empty string")

    def test_extra_repr_parameter_entries(self):
        """Parameter entries render with index, dtype and size, and feed repr()."""
        pl = torch.nn.modules.container.ParameterList()
        pl.append(torch.nn.Parameter(torch.randn(2, 3)))
        pl.append(torch.nn.Parameter(torch.randn(4)))
        rendered = pl.extra_repr()
        expected = ('  (0): Parameter containing: [torch.float32 of size 2x3]\n'
                    '  (1): Parameter containing: [torch.float32 of size 4]')
        self.assertEqual(rendered, expected,
                         "entry lines must match the documented format")
        self.assertIn('  (0): Parameter containing: [torch.float32 of size 2x3]',
                      repr(pl), "repr(ParameterList) must embed the extra_repr lines")
        self.assertEqual(pl.extra_repr(), pl.extra_repr(),
                         "extra_repr must be stable across calls")

    def test_extra_repr_raw_tensor_wrapped(self):
        """A raw tensor appended via append() is wrapped into a Parameter."""
        pl = torch.nn.modules.container.ParameterList()
        pl.append(torch.randn(4))
        self.assertIsInstance(pl[0], torch.nn.Parameter,
                              "append() must wrap raw tensors into Parameters")
        self.assertEqual(pl.extra_repr(),
                         '  (0): Parameter containing: [torch.float32 of size 4]',
                         "the wrapped tensor must render as a Parameter line")

    def test_extra_repr_non_tensor_entry(self):
        """A non-tensor entry renders as an Object-of-type line."""
        pl = torch.nn.modules.container.ParameterList()
        pl.append(torch.nn.Parameter(torch.randn(2)))
        pl.append(None)
        pl.append(42)
        rendered = pl.extra_repr()
        expected = ('  (0): Parameter containing: [torch.float32 of size 2]\n'
                    '  (1): Object of type: NoneType\n'
                    '  (2): Object of type: int')
        self.assertEqual(rendered, expected,
                         "non-tensor entries must render their type name")

    def test_extra_repr_slice_renumbers(self):
        """A sliced sub-list re-renders with fresh zero-based indices."""
        pl = torch.nn.modules.container.ParameterList()
        pl.append(torch.nn.Parameter(torch.randn(2)))
        pl.append(torch.nn.Parameter(torch.randn(3)))
        pl.append(torch.nn.Parameter(torch.randn(4)))
        sub = pl[1:3]
        self.assertEqual(
            sub.extra_repr(),
            '  (0): Parameter containing: [torch.float32 of size 3]\n'
            '  (1): Parameter containing: [torch.float32 of size 4]',
            "sliced ParameterList must renumber entries from zero")

    def test_extra_repr_device_string(self):
        """Device-resident entries carry a device suffix; CPU entries do not."""
        pl = torch.nn.modules.container.ParameterList()
        pl.append(torch.nn.Parameter(torch.randn(3, device=self.device)))
        rendered = pl.extra_repr()
        base = '  (0): Parameter containing: [torch.float32 of size 3]'
        if self.device_name == 'cpu':
            self.assertEqual(rendered, base,
                             "CPU-resident entries must not carry a device suffix")
        else:
            # cuda/privateuse1 devices append " (device)" inside the brackets
            # per the 2.15 source ("[{dtype} of size {size}{device_str}]").
            self.assertEqual(
                rendered, base[:-1] + ' (%s:0)]' % self.device_name,
                "device-resident entries must append the device string")

    def test_extra_repr_alias_is_container_parameterlist(self):
        """The container path is the same class exposed as torch.nn.ParameterList."""
        self.assertIs(torch.nn.modules.container.ParameterList,
                      torch.nn.ParameterList,
                      "container.ParameterList must be the public alias target")


if __name__ == "__main__":
    run_tests()
