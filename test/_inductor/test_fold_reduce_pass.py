import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_reduce


class FoldReduceModel(torch.nn.Module):
    def forward(self, t1):
        sum_1 = torch.ops.aten.sum.dim_IntList(t1, [1, 3])
        return sum_1


class TestFoldReducePass(TestUtils):
    def op_calc(self, t1):
        output = torch.sum(t1, dim=(1, 3), keepdim=False)
        return output



    @parametrize('shape', [(128, 1, 64, 1)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(128, 1, 64, 1)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldReduceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_reduce(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 1, 1)])
    @parametrize('dtype', ['float32'])
    def test_sum_all_dims_size_1_should_fold(self, shape, dtype):
        """所有 reduce 维度都是 1 → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, [1, 2], keepdim=False)
                return reduced + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        gm.recompile()

        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0, "所有 dim=1 的 sum 应被折叠删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(4, 1, 5)])
    @parametrize('dtype', ['float32'])
    def test_sum_partial_dims_not_1_no_fold(self, shape, dtype):
        """部分 dim 不是 1 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, [1], keepdim=False)
                return reduced.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0, "所有 dim=1 的 sum 应被折叠删除")
        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(2, 1, 1, 8)])
    def test_sum_with_keepdim_true_should_fold(self, shape):
        """keepdim=True 且所有 dim=1 → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, [1, 2], keepdim=True)
                return reduced.squeeze()

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        gm.recompile()

        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0)


    @parametrize('shape', [(3, 1)])
    def test_sum_scalar_dim_should_fold(self, shape):
        """dim 是 scalar（int）而不是 list → 正确处理"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, 1, keepdim=False)
                return reduced

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0)


    @parametrize('shape', [(2, 1, 1)])
    def test_multi_sum_chain_should_all_fold(self, shape):
        """多层连续 sum → 应全部折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s1 = torch.ops.aten.sum.dim_IntList(x, [1], keepdim=False)
                s2 = torch.ops.aten.sum.dim_IntList(s1, [1], keepdim=False)
                return s2 * 2.0

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        gm.recompile()

        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0)


    @parametrize('shape', [(1, 64)])
    def test_sum_with_multiple_downstream_users(self, shape):
        """sum 被折叠后，下游多个用户正确指向原输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, [0], keepdim=False)
                a = reduced * 3.0
                b = reduced + 5.0
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_reduce(gm.graph)
        gm.recompile()

        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 0)

        self.assertEqual(model(x), gm(x))


    def test_no_sum_node_no_change(self):
        """无 sum 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = torch.randn(8, 8)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_reduce(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(4, 4)])
    def test_sum_without_shape_info_no_fold(self, shape):
        """输入没有 shape → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                reduced = torch.ops.aten.sum.dim_IntList(x, [0, 1], keepdim=False)
                return reduced

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape 信息
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]

        fold_reduce(gm.graph)

        sum_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList]
        self.assertEqual(len(sum_nodes), 1, "无 shape 信息不应折叠")


instantiate_parametrized_tests(TestFoldReducePass)


if __name__ == "__main__":
    run_tests()