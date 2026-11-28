# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.lru_cache 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.lru_cache
API 签名：lru_cache(maxsize) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空缓存，maxsize=None 和实际整数容量 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | None/int 容量及实际计算回调 |
| 传参与不传参 | 不适用：maxsize 必传，显式取 None/整数 |
| 等价类/边界值 | 容量 -1/0/1/2/None、禁用/淘汰/无上限及累计统计 |
| 正常传参场景 | 缓存复用、清空后的累计命中/未命中 |
| 异常传参场景 | 非法容量类型在应用装饰器时抛 TypeError |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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

from torch.fx.experimental.symbolic_shapes import lru_cache


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_lru_cache_reuse_eviction_and_cumulative_counts_after_clear(self):
        calls = []

        @lru_cache(2)
        def compute(value):
            calls.append(value)
            return value * value + 1

        self.assertEqual([compute(2), compute(2), compute(3)], [5, 5, 10])
        self.assertEqual(calls, [2, 3])
        self.assertEqual(compute.cumulative_cache_info()[:2], (1, 2))
        compute.cache_clear()
        self.assertEqual(compute.cache_info()[:2], (0, 0))
        self.assertEqual(compute.cumulative_cache_info()[:2], (1, 2))
        self.assertEqual(compute(2), 5)
        self.assertEqual(compute.cumulative_cache_info()[:2], (1, 3))

    def test_lru_cache_disabled_cache_and_bound_capacity(self):
        calls = []

        @lru_cache(0)
        def uncached(value):
            calls.append(value)
            return value + 1

        self.assertEqual([uncached(2), uncached(2)], [3, 3])
        self.assertEqual(calls, [2, 2])

        @lru_cache(1)
        def bounded(value):
            return value + 1

        for value in (1, 2, 1):
            bounded(value)
        self.assertEqual(bounded.cache_info().currsize, 1)
        self.assertEqual(bounded.cumulative_cache_info().misses, 3)


    def test_parameter_none_capacity_retains_all_keys_and_negative_capacity_disables_cache(self):
        for capacity in (None, -1):
            calls = []

            @lru_cache(capacity)
            def compute(value):
                calls.append(value)
                return value + 1

            self.assertEqual([compute(value) for value in (1, 2, 3, 1)], [2, 3, 4, 2])
            self.assertEqual(calls, [1, 2, 3] if capacity is None else [1, 2, 3, 1])
            self.assertEqual(compute.cache_info().currsize, 3 if capacity is None else 0)
            compute.cache_clear()
        with self.assertRaises(TypeError):
            lru_cache("invalid")(lambda value: value)


if __name__ == "__main__":
    run_tests()
