# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.graph_transform_observer.GraphTransformObserver 接口功能正确性
API 名称：torch.fx.passes.graph_transform_observer.GraphTransformObserver
API 签名：GraphTransformObserver(gm, passname, subsystem=None, log_url=None)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | subsystem=None/空串/名字，log_url=None/空串/目录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | GraphModule、pass 名字、字符串路径、bisector 决策回调和 DOT 文件 |
| 传参与不传参 | subsystem/log_url 省略及显式值；路径推断和显式覆盖 |
| 等价类/边界值 | 激活/未激活、禁用/执行、正常/异常退出及 hook 恢复 |
| 正常传参场景 | 创建/替换/删除来源、子系统禁用、DOT 内容和变换后图执行 |
| 异常传参场景 | 受控异常传播及 hook 恢复 |
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

import operator
from torch._inductor import config
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import NodeSourceAction


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _fixture(self):
        module = torch.fx.symbolic_trace(lambda value: -value)
        source = next(node for node in module.graph.nodes if node.op == "placeholder")
        old = next(node for node in module.graph.nodes if node.op == "call_function")
        return module, source, old

    def test_graphtransformobserver_replacement_provenance_and_hook_removal(self):
        original = config.trace.provenance_tracking_level
        module, source, old = self._fixture()
        with config.patch({"trace.provenance_tracking_level": 1, "trace.log_url_for_graph_xform": None}):
            observer = GraphTransformObserver(module, "ut_replace")
            self.assertTrue(observer.active)
            with observer:
                with module.graph.inserting_before(old):
                    new = module.graph.call_function(operator.abs, (source,))
                old.replace_all_uses_with(new)
                module.graph.erase_node(old)
            self.assertEqual(observer.created_nodes, {new.name})
            self.assertEqual(observer.erased_nodes, {old.name})
            provenance = next(item for item in new.meta["from_node"] if item.name == old.name)
            self.assertEqual(provenance.pass_name, "ut_replace")
            self.assertIn(NodeSourceAction.REPLACE, provenance.action)
            module.recompile()
            self.assertEqual(module(-4), 4)
            with module.graph.inserting_before(next(node for node in module.graph.nodes if node.op == "output")):
                extra = module.graph.call_function(operator.pos, (source,))
            self.assertNotIn(extra.name, observer.created_nodes)
        self.assertEqual(config.trace.provenance_tracking_level, original)

    def test_graphtransformobserver_exception_exit_removes_hooks_and_inactive_pass_executes(self):
        original = config.trace.provenance_tracking_level
        module, source, _ = self._fixture()
        with config.patch({"trace.provenance_tracking_level": 1, "trace.log_url_for_graph_xform": None}):
            observer = GraphTransformObserver(module, "ut_exception")
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                with observer:
                    raise RuntimeError("controlled")
            with module.graph.inserting_before(next(node for node in module.graph.nodes if node.op == "output")):
                extra = module.graph.call_function(operator.abs, (source,))
            self.assertNotIn(extra.name, observer.created_nodes)
        self.assertEqual(config.trace.provenance_tracking_level, original)
        with config.patch({"trace.provenance_tracking_level": 0, "trace.provenance_tracking_to_timeline": False,
                           "trace.log_url_for_graph_xform": None, "disabled_passes": ""}):
            inactive = GraphTransformObserver(module, "ut_inactive")
            self.assertFalse(inactive.active)
            self.assertEqual(inactive.apply_gm_pass(lambda gm: gm(-3)), 3)


    def test_parameter_subsystem_none_empty_and_named_disable_decisions(self):
        from unittest.mock import patch
        from torch._inductor.compiler_bisector import CompilerBisector

        with config.patch({"trace.provenance_tracking_level": 0, "trace.provenance_tracking_to_timeline": False,
                           "trace.log_url_for_graph_xform": None, "disabled_passes": ""}):
            for subsystem in (None, "", "lowering"):
                for disabled in (False, True):
                    with self.subTest(subsystem=subsystem, disabled=disabled):
                        module, _, _ = self._fixture()
                        calls = []

                        def decision(backend, actual_subsystem, name):
                            calls.append((backend, actual_subsystem, name()))
                            return disabled

                        ran = []
                        with patch.object(CompilerBisector, "disable_subsystem", side_effect=decision):
                            observer = GraphTransformObserver(module, "parameter_pass", subsystem=subsystem,
                                                              log_url=None)
                            result = observer.apply_gm_pass(lambda gm: (ran.append(gm), gm(-3))[1])
                        if subsystem is None:
                            self.assertEqual(calls, [])
                        else:
                            self.assertEqual(calls, [("inductor", subsystem, "parameter_pass")])
                        if subsystem is not None and disabled:
                            self.assertIsNone(result)
                            self.assertEqual(ran, [])
                        else:
                            self.assertEqual(result, 3)
                            self.assertEqual(ran, [module])

    def test_parameter_log_directory_inference_override_and_empty_path_write_dot_graphs(self):
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import pydot

        count_name = "_GraphTransformObserver__pass_count"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inferred = root / "inferred"
            explicit = root / "explicit"
            inferred.mkdir()
            explicit.mkdir()
            with config.patch({"trace.provenance_tracking_level": 0, "trace.provenance_tracking_to_timeline": False,
                               "trace.log_url_for_graph_xform": str(inferred)}):
                with patch.object(GraphTransformObserver, count_name, GraphTransformObserver.get_current_pass_count()):
                    for supplied, expected_dir in ((None, inferred), (str(explicit), explicit), ("", root)):
                        with self.subTest(log_url=supplied):
                            module, source, old = self._fixture()
                            previous_cwd = os.getcwd()
                            try:
                                os.chdir(root)
                                observer = GraphTransformObserver(module, "parameter_log", log_url=supplied)
                                self.assertTrue(observer.active)
                                self.assertEqual(observer.log_url, str(inferred) if supplied is None else supplied)
                                with observer:
                                    with module.graph.inserting_before(old):
                                        new = module.graph.call_function(operator.abs, (source,))
                                    old.replace_all_uses_with(new)
                                    module.graph.erase_node(old)
                            finally:
                                os.chdir(previous_cwd)
                            paths = list(expected_dir.glob("*.dot"))
                            self.assertEqual(len(paths), 2)
                            for path in paths:
                                diagrams = pydot.graph_from_dot_file(str(path))
                                self.assertEqual(len(diagrams), 1)
                                self.assertTrue(diagrams[0].get_nodes())
                            self.assertEqual(observer.created_nodes, {new.name})
                            self.assertEqual(observer.erased_nodes, {old.name})
                            module.recompile()
                            self.assertEqual(module(-3), 3)


if __name__ == "__main__":
    run_tests()
