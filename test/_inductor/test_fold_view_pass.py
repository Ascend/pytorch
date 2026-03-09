import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import view_fold_pass


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
        view_fold_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1, t2)
        inductor_result = graph_module(t1, t2)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8, 16)])
    @parametrize('dtype', ['float32'])
    def test_consecutive_view_like_fold(self, shape, dtype):
        """连续 view/reshape → 应合并为单层或直接消除"""
        class M(torch.nn.Module):
            def forward(self, x):
                v1 = torch.ops.aten.view.default(x, [-1, 16])
                v2 = torch.ops.aten.reshape.default(v1, [4, -1])
                return v2 + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        view_fold_pass(gm.graph)
        gm.recompile()

        view_like_nodes = []
        for n in gm.graph.nodes:
            if n.target in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            ):
                view_like_nodes.append(n)
        self.assertEqual(len(view_like_nodes), 1, "连续 view-like 应全部折叠")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_view_equivalent_shape_fold(self, shape, dtype):
        """view 到完全相同的 shape → 应直接替换为输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [2, 3, 4])
                return torch.relu(v)

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        view_fold_pass(gm.graph)
        gm.recompile()

        view_nodes = []
        for n in gm.graph.nodes:
            if n.target == torch.ops.aten.view.default:
                view_nodes.append(n)
        self.assertEqual(len(view_nodes), 0, "等价 shape 的 view 应被折叠删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(4, 5)])
    @parametrize('dtype', ['float32'])
    def test_view_different_shape_no_fold(self, shape, dtype):
        """view 到不同 shape → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [20])
                return v.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        view_fold_pass(gm.graph)

        view_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.view.default]
        self.assertEqual(len(view_nodes), 1, "shape 不等价不应折叠")


    @parametrize('shape', [(1, 1, 8)])
    def test_squeeze_unsqueeze_view_chain(self, shape):
        """squeeze → unsqueeze → view 的组合（部分可折叠）"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.squeeze.default(x)
                u = torch.ops.aten.unsqueeze.default(s, 0)
                v = torch.ops.aten.view.default(u, [1, 8])
                return v.sum()

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        view_fold_pass(gm.graph)
        gm.recompile()
        view_nodes = []
        for n in gm.graph.nodes:
            if n.target in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            ):
                view_nodes.append(n)
        self.assertLess(len(view_nodes), 3, "部分 view-like 应被折叠")


    @parametrize('shape', [(8, 16)])
    def test_view_with_multiple_users_no_fold(self, shape):
        """view 有多个下游用户 → 不折叠（虽然代码没显式检查 users，但实际替换会影响）"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [-1])
                a = torch.relu(v)
                b = v + 1.0
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        view_fold_pass(gm.graph)

        view_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.view.default]
        self.assertEqual(len(view_nodes), 1, "多用户 view 不应折叠")


    @parametrize('shape', [(4, 5)])
    def test_no_view_node_no_change(self, shape):
        """无 view-like 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        view_fold_pass(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(2, 3, 4)])
    def test_view_without_shape_info_no_fold(self, shape):
        """输入缺少 shape 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, [6, 4])
                return v.sum()

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape 信息
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]

        view_fold_pass(gm.graph)

        view_nodes = []
        for n in gm.graph.nodes:
            if n.target in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            ):
                view_nodes.append(n)
        self.assertEqual(len(view_nodes), 1, "缺少 shape 不应折叠")


instantiate_parametrized_tests(TestFoldViewPass)


if __name__ == "__main__":
    run_tests()