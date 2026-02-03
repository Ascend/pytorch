import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FoldViewModel(torch.nn.Module):
    def forward(self, t1, t2):
        squeeze_1 = torch.ops.aten.squeeze.dim(t1, 2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(squeeze_1, 0)
        view_1 = torch.ops.aten.view.default(unsqueeze_1, [1, -1])
        
        output = torch.ops.aten.view.default(t2, [128, 64])
        return {"view_1": view_1, "output": output}


class TestFoldViewPass(TestUtils):
    def op_calc(self, t1, t2):
        squeeze_1 = t1.squeeze(2) # 去掉第3维（长度为1），y.shape: (1,3,5)
        unsqueeze_1 = squeeze_1.unsqueeze(0) # 在第1维增加一个长度为1的维度，z.shape: (1,1,3,5)
        view_1 = unsqueeze_1.view(1, -1) # 展平成 (1, 15)，w.shape: (1,15)
        
        output = t2.view(128, 64)
        return {"view_1": view_1, "output": output}


    def test_compile_cases(self):
        t1 = torch.randn(1, 3, 1, 5)
        t2 = torch.randn(128, 64)
        std_result = self.op_calc(t1, t2)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        t1 = torch.randn(1, 3, 1, 5)
        t2 = torch.randn(128, 64)
        model = FoldViewModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import view_fold_pass
        view_fold_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFoldViewPass)


if __name__ == "__main__":
    run_tests()