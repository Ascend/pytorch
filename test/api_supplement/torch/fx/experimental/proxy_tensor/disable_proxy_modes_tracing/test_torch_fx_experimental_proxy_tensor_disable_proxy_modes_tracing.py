# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.disable_proxy_modes_tracing 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.disable_proxy_modes_tracing
API 签名：disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode, None, None]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：嵌套禁用和异常退出 |
| 正常传参场景 | 已覆盖：实际调度 mode 临时禁用与恢复 |
| 异常传参场景 | 已覆盖：受控异常后 mode 恢复 |
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

from torch.fx.experimental import proxy_tensor as pt


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _tracer(self):
        tracer = pt.PythonKeyTracer()
        tracer.root = torch.nn.Module()
        tracer.graph = torch.fx.Graph()
        return tracer

    def _mode(self):
        return pt.ProxyTorchDispatchMode(self._tracer(), "real")

    def test_disable_proxy_modes_tracing_active_mode_is_removed_temporarily_and_restored(self):
        original = pt.get_proxy_mode()
        mode = self._mode()
        with mode:
            self.assertIs(pt.get_proxy_mode(), mode)
            with pt.disable_proxy_modes_tracing():
                self.assertIsNone(pt.get_proxy_mode())
                with pt.disable_proxy_modes_tracing():
                    self.assertIsNone(pt.get_proxy_mode())
            self.assertIs(pt.get_proxy_mode(), mode)
        self.assertIs(pt.get_proxy_mode(), original)

    def test_disable_proxy_modes_tracing_exception_exit_restores_mode(self):
        original = pt.get_proxy_mode()
        mode = self._mode()
        with mode:
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                with pt.disable_proxy_modes_tracing():
                    self.assertIsNone(pt.get_proxy_mode())
                    raise RuntimeError("controlled")
            self.assertIs(pt.get_proxy_mode(), mode)
        self.assertIs(pt.get_proxy_mode(), original)


if __name__ == "__main__":
    run_tests()
