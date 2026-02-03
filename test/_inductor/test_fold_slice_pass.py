import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FoldSliceModel(torch.nn.Module):
    def forward(self, base, view, t1, t2, t3):
        end = 16
        slice_1 = torch.ops.aten.slice_scatter.default(base, view, 1, 0, end)
        result = view + slice_1
        slice_2 = torch.ops.aten.slice_scatter.default(t1, t2, 1, 0, 3)
        
        b = torch.ops.aten.slice.Tensor(t3, 1, 0, None)
        result_c = torch.ops.aten.add.Tensor(b, b)
        return {"result": result, "slice_2": slice_2, "result_c": result_c}


class TestFoldSlicePass(TestUtils):
    def op_calc(self, base, view, t1, t2, t3):
        end = view.shape[1]
        result = torch.slice_scatter(base, view, 1, 0, end)
        result = result + view
        data = t1.slice_scatter(t2, dim=1, start=0, end=t2.shape[1])
        
        b = t3[:, 0:]
        result_c = b + b
        return {"result": result, "data": data, "result_c": result_c}


    def test_compile_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        std_result = self.op_calc(base, view, t1, t2, t3)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(base, view, t1, t2, t3)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        model = FoldSliceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(base, view, t1, t2, t3)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_slice
        fold_slice(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(base, view, t1, t2, t3)
        inductor_result = graph_module(base, view, t1, t2, t3)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFoldSlicePass)


if __name__ == "__main__":
    run_tests()