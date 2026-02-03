import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class CatSliceCatModel(torch.nn.Module):
    def forward(self, first_element):
        # 第一次 cat：在 dim=1 上拼接 3 个部分 → [1, 3, 64, 1024]
        cat1 = torch.cat(
            [
                first_element[:, 0:1, :, :],   # 第0个channel
                first_element[:, 1:2, :, :],   # 第1个channel
                first_element[:, 2:3, :, :],   # 第2个channel
            ],
            dim=1,
        )
        # 第二次 cat：在同一个 dim=1 上，对 cat1 的连续切片再做一次 cat
        # → 完全等价于 cat1，Inductor 应该直接 erase 掉第二次 cat 和所有 getitem
        cat2 = torch.cat(
            [
                cat1[:, 0:1, :, :],   # 0~1
                cat1[:, 1:2, :, :],   # 1~2
                cat1[:, 2:3, :, :],   # 2~3
            ],
            dim=1,
        )
        return cat2


class TestCatSliceCatPass(TestUtils):
    def op_calc(self, first_element):
        # 第一次 cat：在 dim=1 上拼接 3 个部分 → [1, 3, 64, 1024]
        cat1 = torch.cat(
            [
                first_element[:, 0:1, :, :],   # 第0个channel
                first_element[:, 1:2, :, :],   # 第1个channel
                first_element[:, 2:3, :, :],   # 第2个channel
            ],
            dim=1,
        )
        # 第二次 cat：在同一个 dim=1 上，对 cat1 的连续切片再做一次 cat
        # → 完全等价于 cat1，Inductor 应该直接 erase 掉第二次 cat 和所有 getitem
        cat2 = torch.cat(
            [
                cat1[:, 0:1, :, :],   # 0~1
                cat1[:, 1:2, :, :],   # 1~2
                cat1[:, 2:3, :, :],   # 2~3
            ],
            dim=1,
        )
        return cat2


    @parametrize('shape', [(1, 3, 64, 1024)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(first_element)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 3, 64, 1024)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = CatSliceCatModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import cat_slice_cat_fold_pass
        cat_slice_cat_fold_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(first_element)
        inductor_result = graph_module(first_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestCatSliceCatPass)


if __name__ == "__main__":
    run_tests()