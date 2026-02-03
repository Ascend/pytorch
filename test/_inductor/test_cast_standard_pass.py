import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class CastStandardModel(torch.nn.Module):
    def forward(self, arg1_1):
        cast_1 = torch.ops.npu._npu_dtype_cast.default(arg1_1, torch.int64)
        output = torch.ops.aten.relu.default(cast_1)
        return output


class TestCastStandardPass(TestUtils):
    def op_calc(self, first_element):
        cast_1 = torch.ops.npu._npu_dtype_cast.default(first_element, torch.int64)
        output = torch.ops.aten.relu.default(cast_1)
        return output

    @parametrize('shape', [(256, 5)])
    @parametrize('dtype', ['int64'])
    def test_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(first_element)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        

    @parametrize('shape', [(256, 5)])
    @parametrize('dtype', ['int64'])
    def test_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = CastStandardModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_cast
        fold_cast(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(first_element)
        inductor_result = graph_module(first_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestCastStandardPass)


if __name__ == "__main__":
    run_tests()