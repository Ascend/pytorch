import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_cat


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
        fold_cat(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2, t3, t4, t5)
        inductor_result = graph_module(t1, t2, t3, t4, t5)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8, 16)])
    @parametrize('dtype', ['float32'])
    def test_deep_nested_same_axis_fold(self, shape, dtype):
        """多层嵌套同轴 cat，应完全折叠为单层 cat"""
        class M(torch.nn.Module):
            def forward(self, a, b, c, d, e):
                c1 = torch.ops.aten.cat.default([a, b], dim=1)
                c2 = torch.ops.aten.cat.default([c1, c], dim=1)
                c3 = torch.ops.aten.cat.default([c2, d], dim=1)
                c4 = torch.ops.aten.cat.default([c3, e], dim=1)
                return c4

        tensors = [self._generate_tensor(shape, dtype) for _ in range(5)]
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(*tensors)
        gm.graph.print_tabular()
        fold_cat(gm.graph)
        gm.graph.print_tabular()
        gm.recompile()

        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 1, "应折叠为单层 cat")
        self.assertEqual(len(cat_nodes[0].args[0]), 5, "输入应包含所有 5 个原始 tensor")

        self.assertEqual(model(*tensors), gm(*tensors))


    @parametrize('shape', [(3, 5)])
    @parametrize('dtype', ['float32'])
    def test_different_axis_no_fold(self, shape, dtype):
        """不同轴的 cat 不应折叠"""
        class M(torch.nn.Module):
            def forward(self, a, b, c):
                c1 = torch.ops.aten.cat.default([a, b], 0)   # dim=0
                c2 = torch.ops.aten.cat.default([c1, c], 1)  # dim=1 → 不同
                return c2

        t1, t2 = [self._generate_tensor(shape, dtype) for _ in range(2)]
        shape1 = (shape[0] * 2, shape[1])
        t3 = self._generate_tensor(shape1, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(t1, t2, t3)

        fold_cat(gm.graph)

        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 2, "不同轴不应折叠")


    @parametrize('shape', [(2, 6)])
    @parametrize('dtype', ['float32'])
    def test_input_cat_has_multiple_users_no_fold(self, shape, dtype):
        """被多个下游使用的 cat 输入不应折叠"""
        class M(torch.nn.Module):
            def forward(self, a, b, c):
                inner = torch.ops.aten.cat.default([a, b], 1)
                out1 = inner + 1.0
                out2 = inner * 2.0
                out3 = torch.ops.aten.cat.default([inner, c], 1)
                return out1, out2, out3

        t1, t2, t3 = [self._generate_tensor(shape, dtype) for _ in range(3)]
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(t1, t2, t3)

        fold_cat(gm.graph)

        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 2, "有多个用户，不应折叠 inner cat")


    @parametrize('shape', [(4, 3, 5)])
    @parametrize('dtype', ['float32'])
    def test_last_dim_negative_and_positive_axis(self, shape, dtype):
        """最后一维使用 -1 和正数轴应视为相同"""
        class M(torch.nn.Module):
            def forward(self, a, b, c):
                inner = torch.ops.aten.cat.default([a, b], -1)   # -1
                outer = torch.ops.aten.cat.default([inner, c], 2)  # 2
                return outer

        t1, t2, t3 = [self._generate_tensor(shape, dtype) for _ in range(3)]
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(t1, t2, t3)
        fold_cat(gm.graph)
        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 1, "最后一维 -1 和 2 应折叠")
        self.assertEqual(cat_nodes[0].args[1], -1)


    @parametrize('shape', [(2, 7)])
    @parametrize('dtype', ['float32'])
    def test_mixed_cat_and_non_cat_inputs(self, shape, dtype):
        """cat 输入中混有非 cat 节点"""
        class M(torch.nn.Module):
            def forward(self, a, b, c):
                inner = torch.ops.aten.cat.default([a, b], 0)
                outer = torch.ops.aten.cat.default([inner, c * 1.5, torch.ops.aten.relu.default(c)], 0)
                return outer

        t1, t2, t3 = [self._generate_tensor(shape, dtype) for _ in range(3)]
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(t1, t2, t3)

        fold_cat(gm.graph)

        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 1, "混有非 cat 输入，应保留一层 cat")
        self.assertEqual(len(cat_nodes[0].args[0]), 4, "应包含 inner + c*1.5 + relu(c)")


    def test_single_input_cat_no_fold(self):
        """只有一个输入的 cat 不应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.cat.default([x], dim=0)

        t = self._generate_tensor((4, 5), 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(t)

        fold_cat(gm.graph)

        cat_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default]
        self.assertEqual(len(cat_nodes), 1, "单输入 cat 不应被移除")


instantiate_parametrized_tests(TestFoldCatPass)


if __name__ == "__main__":
    run_tests()