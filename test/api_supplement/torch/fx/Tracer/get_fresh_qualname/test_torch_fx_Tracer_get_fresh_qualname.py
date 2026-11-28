# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.Tracer.get_fresh_qualname 接口功能正确性
API 名称：torch.fx.Tracer.get_fresh_qualname
API 签名：Tracer.get_fresh_qualname(self, prefix) -> str

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空集合、缺失元数据或无依赖节点 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Graph/Node 和相关结构参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：独立前缀和零号名字释放后重置 |
| 正常传参场景 | 已覆盖：避让现有属性及新保留名字 |
| 异常传参场景 | 不适用：使用有效图或辅助对象 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
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




@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_get_fresh_qualname_skips_existing_attributes_and_reservations(self):
        tracer = torch.fx.Tracer()
        tracer.trace(torch.nn.Identity())
        tracer.root.generated0 = "occupied"
        tracer.root.generated1 = "occupied"
        first = tracer.get_fresh_qualname("generated")
        self.assertEqual(first, "generated2")
        setattr(tracer.root, first, "new reservation")
        self.assertEqual(tracer.get_fresh_qualname("generated"), "generated3")
        self.assertEqual(tracer.root.generated0, "occupied")

    def test_get_fresh_qualname_independent_prefix_and_reset_when_zero_is_free(self):
        tracer = torch.fx.Tracer()
        tracer.trace(torch.nn.Identity())
        self.assertEqual(tracer.get_fresh_qualname("temporary"), "temporary0")
        tracer.root.temporary0 = 1
        self.assertEqual(tracer.get_fresh_qualname("temporary"), "temporary1")
        self.assertEqual(tracer.get_fresh_qualname("other"), "other0")
        del tracer.root.temporary0
        self.assertEqual(tracer.get_fresh_qualname("temporary"), "temporary0")


if __name__ == "__main__":
    run_tests()
