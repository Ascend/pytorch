import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FoldSqueezeModel(torch.nn.Module):
    def forward(self, t1, t2,):
        squeeze_1 = torch.ops.aten.squeeze.default(t1)
        squeeze_2 = torch.ops.aten.squeeze.default(squeeze_1)
        
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(t2, 1)
        squeeze_3 = torch.ops.aten.squeeze.dim(unsqueeze_1, 1)

        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


class TestFoldSqueezePass(TestUtils):
    def op_calc(self, t1, t2):
        squeeze_1 = torch.squeeze(t1)
        squeeze_2 = torch.squeeze(squeeze_1)
        unsqueeze_1 = torch.unsqueeze(t2, 1)
        squeeze_3 = torch.squeeze(unsqueeze_1, 1)
        return {"squeeze_2": squeeze_2, "squeeze_3": squeeze_3}


    def test_compile_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        std_result = self.op_calc(t1, t2)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        t1 = torch.randn(2, 4)
        t2 = torch.randn(2, 1, 1, 4)
        model = FoldSqueezeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_squeeze
        fold_squeeze(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFoldSqueezePass)


if __name__ == "__main__":
    run_tests()