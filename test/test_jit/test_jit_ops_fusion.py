import os
import shutil

import torch
import torch_npu.jit
from torch_npu.testing.testcase import TestCase, run_tests


class TestJitOpsFusion(TestCase):
    test_jit_model_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "test_jit_fusion")

    @classmethod
    def setUpClass(cls):
        os.makedirs(TestJitOpsFusion.test_jit_model_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        assert os.path.exists(TestJitOpsFusion.test_jit_model_path)
        shutil.rmtree(TestJitOpsFusion.test_jit_model_path, ignore_errors=True)

    def test_func_fast_gelu(self):
        def ori_func(x):
            x = x**2
            x = torch.nn.functional.gelu(x)
            return x

        x = torch.rand(3, 3).npu()
        jit_model = torch.jit.trace(ori_func, x)
        torch_npu.jit.optimize(jit_model)
        match_kinds = [n.kind() == 'npu::fast_gelu' for n in jit_model.graph.nodes()]
        self.assertEqual(any(match_kinds), True)

    def test_module_fast_gelu(self):
        class Ori_Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, x):
                x = x**2
                x = torch.nn.functional.gelu(x)
                return x

        x = torch.rand(3, 3).npu()
        jit_model = torch.jit.trace(Ori_Module(), x)
        torch_npu.jit.optimize(jit_model)
        match_kinds = [n.kind() == 'npu::fast_gelu' for n in jit_model.graph.nodes()]
        self.assertEqual(any(match_kinds), True)

    def test_fast_gelu_result_check(self):
        def ori_func(x):
            x = x**2
            x = torch.nn.functional.gelu(x)
            return x

        x = torch.rand(3, 3).npu()
        jit_model = torch.jit.trace(ori_func, x)
        pre_result = jit_model(x)

        torch_npu.jit.optimize(jit_model)

        model_path = os.path.join(TestJitOpsFusion.test_jit_model_path, 'rewrite.pt')
        torch.jit.save(jit_model, model_path)
        assert os.path.isfile(model_path)
        jit_model = torch.jit.load(model_path)
        post_result = jit_model(x)
        self.assertAlmostEqual(pre_result, post_result, delta=0.05)


if __name__ == '__main__':
    run_tests()
