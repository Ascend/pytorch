# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.log_lru_cache_stats 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.log_lru_cache_stats
API 签名：log_lru_cache_stats(wrapped_f) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：清空后的累计统计不丢失 |
| 正常传参场景 | 已覆盖：DEBUG 日志中的函数名及命中统计 |
| 异常传参场景 | 不适用：有效符号输入的确定结果 |
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

from torch.fx.experimental.symbolic_shapes import log_lru_cache_stats
from torch.fx.experimental import symbolic_shapes
from torch.fx.experimental.symbolic_shapes import lru_cache


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_log_lru_cache_stats_debug_log_contains_name_and_cache_counts(self):
        @lru_cache(2)
        def compute(value):
            return value + 1

        compute(3)
        compute(3)
        with self.assertLogs(symbolic_shapes.log, level="DEBUG") as output:
            log_lru_cache_stats(compute)
        message = output.records[0].getMessage()
        self.assertIn("compute", message)
        self.assertIn("hits=1", message)
        self.assertIn("misses=1", message)

    def test_log_lru_cache_stats_log_retains_counts_across_clear(self):
        @lru_cache(1)
        def compute(value):
            return value

        compute(1)
        compute.cache_clear()
        compute(2)
        with self.assertLogs(symbolic_shapes.log, level="DEBUG") as output:
            log_lru_cache_stats(compute)
        self.assertIn("misses=2", output.records[0].getMessage())
        self.assertEqual(compute.cache_info().misses, 1)


if __name__ == "__main__":
    run_tests()
