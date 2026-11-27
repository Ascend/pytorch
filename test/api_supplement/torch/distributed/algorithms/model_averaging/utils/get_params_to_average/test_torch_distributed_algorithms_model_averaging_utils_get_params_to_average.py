# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.model_averaging.utils.get_params_to_average 接口功能正确性
API 名称：torch.distributed.algorithms.model_averaging.utils.get_params_to_average
API 签名：get_params_to_average(params) -> list[Parameter]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|-------------------------------------|--------------------------------------------------|
| 空/非空        | 空列表、空迭代器、空 param_groups 字典 | 已覆盖（test_model_parameters_and_empty_inputs） |
| 枚举选项       | 无枚举入参 | 未覆盖，函数仅接收参数或参数组可迭代对象，无枚举参数 |
| 参数类型       | parameters 与 param_groups | 已覆盖（test_optimizer_param_groups_and_mixed_devices 等） |
| 传参与不传参   | 单必选参数 params | 已覆盖（全部正常用例） |
| 等价类/边界值  | 全/部分/无 grad、多组展平 | 已覆盖（test_model_parameters_and_empty_inputs 等） |
| 正常传参场景   | 过滤保留对象身份与顺序 | 已覆盖（test_model_parameters_and_empty_inputs 等） |
| 异常传参场景   | 非法元素类型 NotImplementedError、缺键 KeyError | 已覆盖（test_invalid_inputs_raises） |
| 混合设备类型   | NPU/CUDA 与 CPU 混合设备参数列表 | 已覆盖（test_optimizer_param_groups_and_mixed_devices） |

未覆盖项及原因：
- 枚举选项：get_params_to_average 仅接收 params 可迭代对象，不包含枚举参数。

注意：本接口为纯 Python 参数过滤工具，不涉及跨进程通信，采用单进程测试；
     所有测试张量均使用 self.device_name（及混合设备场景中的 CPU）构造，
     并断言返回列表的长度、对象身份（is）与设备一致性。
"""
import os
import unittest

import torch
import torch.nn as nn
from torch.distributed.algorithms.model_averaging.utils import get_params_to_average

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
class TestGetParamsToAverage(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_model_parameters_and_empty_inputs(self):
        for inp in ([], iter([]), [{"params": []}]):
            self.assertEqual(get_params_to_average(inp), [])

        layer1 = nn.Linear(4, 3, bias=True).to(self.device_name)
        layer2 = nn.Linear(3, 2, bias=True).to(self.device_name)
        all_params = list(layer1.parameters()) + list(layer2.parameters())
        self.assertEqual(get_params_to_average(all_params), [])

        layer1(torch.randn(2, 4, device=self.device_name)).sum().backward()
        filtered = get_params_to_average(all_params)
        self.assertEqual(len(filtered), 2)
        self.assertIs(filtered[0], layer1.weight)
        self.assertIs(filtered[1], layer1.bias)

        layer2(torch.randn(2, 3, device=self.device_name)).sum().backward()
        filtered_all = get_params_to_average(iter(all_params))
        self.assertEqual(len(filtered_all), 4)
        for actual, expected in zip(filtered_all, all_params):
            self.assertIs(actual, expected)

    def test_optimizer_param_groups_and_mixed_devices(self):
        p1 = nn.Parameter(torch.randn(3, 3, device=self.device_name))
        p2 = nn.Parameter(torch.randn(3, device=self.device_name))
        p3 = nn.Parameter(torch.randn(2, 2, device=self.device_name))
        p4 = nn.Parameter(torch.randn(2, device=self.device_name))
        p1.grad = torch.ones_like(p1)
        p4.grad = torch.ones_like(p4)
        optimizer = torch.optim.SGD([
            {"params": [p1, p2], "lr": 0.01},
            {"params": [p3, p4], "lr": 0.05},
        ])
        filtered = get_params_to_average(optimizer.param_groups)
        self.assertEqual(len(filtered), 2)
        self.assertIs(filtered[0], p1)
        self.assertIs(filtered[1], p4)

        p_dev = nn.Parameter(torch.randn(4, device=self.device_name))
        p_cpu = nn.Parameter(torch.randn(4, device="cpu"))
        p_dev_no_grad = nn.Parameter(torch.randn(4, device=self.device_name))
        p_dev.grad = torch.ones_like(p_dev)
        p_cpu.grad = torch.ones_like(p_cpu)
        mixed = get_params_to_average([p_dev, p_dev_no_grad, p_cpu])
        self.assertEqual(len(mixed), 2)
        self.assertIs(mixed[0], p_dev)
        self.assertIs(mixed[1], p_cpu)
        self.assertEqual(mixed[0].device.type, self.device_name)
        self.assertEqual(mixed[1].device.type, "cpu")

    def test_invalid_inputs_raises(self):
        for invalid_item in (torch.randn(3, device=self.device_name), 123, "param", None):
            with self.assertRaises(NotImplementedError):
                get_params_to_average([invalid_item])
        with self.assertRaises(KeyError):
            get_params_to_average([{"lr": 0.01}])


if __name__ == "__main__":
    run_tests()
