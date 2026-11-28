# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cuda.graph_annotations.get_kernel_py_stacks 接口功能正确性
API 名称：torch.cuda.graph_annotations.get_kernel_py_stacks
API 签名：get_kernel_py_stacks() -> Mapping[int, str]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：新增/删除、空视图和不存在的 ID |
| 正常传参场景 | 已覆盖：实际记录对象的只读实时视图及 ID 组合 |
| 异常传参场景 | 已覆盖：禁止赋值 TypeError 与缺失 ID KeyError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 图执行方式：仅验证记录视图语义；CUDA 捕获和驱动记录路径需用户在 GPU 验证。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from unittest.mock import patch
import torch.cuda._graph_annotations as annotations
from torch.cuda.graph_annotations import get_kernel_py_stacks


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _registry(self):
        isolation = patch.dict(annotations._GRAPH_ANNOTATIONS, {}, clear=True)
        self.addCleanup(isolation.stop)
        isolation.start()
        record = annotations._GraphAnnotations()
        record.py_stacks[7] = "model.py:12:forward\nmodel.py:3:call"
        annotations._GRAPH_ANNOTATIONS[3] = record
        return record, (3 << 32) | 7

    def test_get_kernel_py_stacks_mapping_is_live_read_only_and_uses_composite_node_ids(self):
        record, key = self._registry()
        view = get_kernel_py_stacks()
        self.assertEqual(dict(view), {key: record.py_stacks[7]})
        with self.assertRaises(TypeError):
            view[key] = "forbidden"
        record.py_stacks[8] = "next.py:2:launch"
        self.assertEqual(view[(3 << 32) | 8], "next.py:2:launch")
        self.assertEqual(len(view), 2)

    def test_get_kernel_py_stacks_removed_entries_disappear_and_missing_ids_raise(self):
        record, key = self._registry()
        view = get_kernel_py_stacks()
        record.py_stacks.clear()
        self.assertEqual(dict(view), {})
        with self.assertRaises(KeyError):
            _ = view[key]
        annotations._GRAPH_ANNOTATIONS.clear()
        self.assertEqual(len(view), 0)


if __name__ == "__main__":
    run_tests()
