import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FoldDetachModel(torch.nn.Module):
    def forward(self, t1):
        detach_x = torch.ops.aten.detach.default(t1)
        output = torch.ops.aten.relu.default(detach_x)
        return output


class TestFoldDetachPass(TestUtils):
    def op_calc(self, t1):
        detach_x = torch.ops.aten.detach.default(t1)
        output = torch.ops.aten.relu.default(detach_x)
        return output



    @parametrize('shape', [(3, 3)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(3, 3)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldDetachModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_detach
        fold_detach(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFoldDetachPass)


if __name__ == "__main__":
    run_tests()