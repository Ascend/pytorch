import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_where


class FoldWhereModel(torch.nn.Module):
    def forward(self, t1):
        mask = t1 > 0
        return torch.ops.aten.where.self(mask, t1, t1)


class TestFoldWherePass(TestUtils):
    def op_calc(self, t1):
        mask = t1 > 0
        return torch.where(mask, t1, t1) # 两个分支完全相同


    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldWhereModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_where(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32'])
    def test_where_same_branch_fold(self, shape, dtype):
        """inp == other → 应折叠为 inp"""
        class M(torch.nn.Module):
            def forward(self, cond, x):
                result = torch.ops.aten.where.self(cond, x, x)
                return result + 1.0

        cond = torch.rand(shape).npu() > 0.5
        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(cond, x)

        fold_where(gm.graph)
        gm.recompile()

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 0, "相同分支的 where 应被折叠删除")

        self.assertEqual(model(cond, x), gm(cond, x))


    @parametrize('shape', [(5, 5)])
    @parametrize('dtype', ['float32'])
    def test_where_both_one_like_fold(self, shape, dtype):
        """inp 和 other 都是 1-like → 应折叠为 inp（或 one）"""
        class M(torch.nn.Module):
            def forward(self, cond):
                ones1 = torch.ones_like(cond, dtype=torch.float32)
                ones2 = torch.ones_like(cond, dtype=torch.float32)
                result = torch.ops.aten.where.self(cond, ones1, ones2)
                return result.sum()

        cond = torch.rand(shape) > 0.5
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(cond)

        fold_where(gm.graph)
        gm.recompile()

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 0, "两个 1-like 分支应折叠")


    @parametrize('shape', [(3, 7)])
    @parametrize('dtype', ['float32'])
    def test_where_both_zero_like_fold(self, shape, dtype):
        """inp 和 other 都是 0-like → 应折叠为 inp（或 zero）"""
        class M(torch.nn.Module):
            def forward(self, cond):
                zeros1 = torch.zeros_like(cond, dtype=torch.float32)
                zeros2 = torch.zeros_like(cond, dtype=torch.float32)
                result = torch.ops.aten.where.self(cond, zeros1, zeros2)
                return result.mean()

        cond = torch.rand(shape) > 0.5
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(cond)

        fold_where(gm.graph)
        gm.recompile()

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 0, "两个 0-like 分支应折叠")


    @parametrize('shape', [(2, 16)])
    @parametrize('dtype', ['float32'])
    def test_where_different_branches_no_fold(self, shape, dtype):
        """分支不同 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, cond, x, y):
                result = torch.ops.aten.where.self(cond, x, y)
                return result.sum()

        cond = torch.rand(shape).npu() > 0.5
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype) + 1.0
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(cond, x, y)

        fold_where(gm.graph)

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 1, "分支不同不应折叠")


    @parametrize('shape', [(1, 64)])
    def test_where_with_multiple_downstream_users(self, shape):
        """where 被折叠后，下游多个用户正确指向 inp"""
        class M(torch.nn.Module):
            def forward(self, cond, x):
                result = torch.ops.aten.where.self(cond, x, x)
                a = result * 3.0
                b = result + 5.0
                return a, b

        cond = torch.rand(shape).npu() > 0.5
        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(cond, x)

        fold_where(gm.graph)
        gm.recompile()

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 0)

        self.assertEqual(model(cond, x), gm(cond, x))


    def test_no_where_node_no_change(self):
        """无 where 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = torch.randn(8, 8).npu()
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_where(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(4, 4)])
    def test_where_without_meta_no_fold(self, shape):
        """缺少 meta 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, cond, x):
                result = torch.ops.aten.where.self(cond, x, x)
                return result

        cond = torch.rand(shape).npu() > 0.5
        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 meta
        for node in gm.graph.nodes:
            if "tensor_meta" in node.meta:
                del node.meta["tensor_meta"]

        fold_where(gm.graph)

        where_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.where.self]
        self.assertEqual(len(where_nodes), 1, "缺少 meta 不应折叠")


instantiate_parametrized_tests(TestFoldWherePass)


if __name__ == "__main__":
    run_tests()