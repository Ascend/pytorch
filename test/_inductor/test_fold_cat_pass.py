import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FoldCatModel(torch.nn.Module):
    def forward(self, t1, t2, t3, t4, t5):
        cat1 = torch.ops.aten.cat.default([t1, t2], 1)
        cat2 = torch.ops.aten.cat.default([cat1, t3], 1)
        cat3 = torch.ops.aten.cat.default([cat2, t4], 1)
        cat4 = torch.ops.aten.cat.default([cat3, t5], 1)
        return cat4


class TestFoldCatPass(TestUtils):
    def op_calc(self, t1, t2, t3, t4, t5):
        cat1 = torch.cat([t1, t2], dim=1)
        cat2 = torch.cat([cat1, t3], dim=1)
        cat3 = torch.cat([cat2, t4], dim=1)
        cat4 = torch.cat([cat3, t5], dim=1)
        return cat4


    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        t4 = self._generate_tensor(shape, dtype)
        t5 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1, t2, t3, t4, t5)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1, t2, t3, t4, t5)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        t2 = self._generate_tensor(shape, dtype)
        t3 = self._generate_tensor(shape, dtype)
        t4 = self._generate_tensor(shape, dtype)
        t5 = self._generate_tensor(shape, dtype)
        model = FoldCatModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1, t2, t3, t4, t5)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_cat
        fold_cat(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2, t3, t4, t5)
        inductor_result = graph_module(t1, t2, t3, t4, t5)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFoldCatPass)


if __name__ == "__main__":
    run_tests()