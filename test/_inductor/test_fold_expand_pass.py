import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_expand


class FoldExpandModel(torch.nn.Module):
    def forward(self, t1):
        add = torch.ops.aten.expand.default(t1, [256, 128, 1])
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class TestFoldExpandPass(TestUtils):
    def op_calc(self, t1):
        expand = torch.ops.aten.expand.default(t1, [256, 128, 1])
        relu = torch.relu(expand)
        return relu



    @parametrize('shape', [(256, 128, 1)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(256, 128, 1)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldExpandModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_expand(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_expand_same_shape_should_fold(self, shape, dtype):
        """目标 shape 与输入完全相同 → 应折叠删除"""
        class M(torch.nn.Module):
            def forward(self, x):
                expanded = torch.ops.aten.expand.default(x, [2, 3, 4])
                return expanded + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)
        gm.recompile()

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 0, "相同 shape 的 expand 应被删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(1, 1, 4)])
    @parametrize('dtype', ['float32'])
    def test_expand_with_minus_one_should_fold(self, shape, dtype):
        """目标 shape 包含 -1（广播维度），实际等价 → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                expanded = torch.ops.aten.expand.default(x, [1, -1, 4])
                return torch.relu(expanded)

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 0, "-1 维度等价时应折叠")


    @parametrize('shape', [(2, 3, 1)])
    @parametrize('dtype', ['float32'])
    def test_expand_different_shape_no_fold(self, shape, dtype):
        """目标 shape 与输入不等价 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                expanded = torch.ops.aten.expand.default(x, [2, 3, 6])
                return expanded.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 1, "shape 不匹配不应折叠")


    @parametrize('shape', [(1, 8)])
    def test_multi_expand_chain_should_all_fold(self, shape):
        """多层连续 expand → 应全部折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                e1 = torch.ops.aten.expand.default(x, [1, 8])
                e2 = torch.ops.aten.expand.default(e1, [1, 8])
                e3 = torch.ops.aten.expand.default(e2, [1, 8])
                return e3 * 2.0

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)
        gm.recompile()

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 0, "多层等价 expand 应全部删除")


    @parametrize('shape', [(4, 4)])
    def test_expand_with_multiple_downstream_users(self, shape):
        """expand 被折叠后，下游多个用户正确指向原输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                expanded = torch.ops.aten.expand.default(x, [4, 4])
                a = expanded * 3.0
                b = expanded + 5.0
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)
        gm.recompile()

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 0)

        self.assertEqual(model(x), gm(x))


    def test_no_expand_node_no_change(self):
        """图中无 expand 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = torch.randn(8, 8)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_expand(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph), "无 expand 节点不应修改图")


    @parametrize('shape', [(2, 3)])
    def test_expand_without_shape_info_no_fold(self, shape):
        """输入没有 shape 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                expanded = torch.ops.aten.expand.default(x, [2, 3])
                return expanded

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape 信息
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]

        fold_expand(gm.graph)

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 1, "无 shape 信息不应折叠")


    @parametrize('shape', [(1, 5)])
    def test_expand_non_list_shape_no_fold(self, shape):
        """shape 参数不是 list → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                # 故意传入非 list（如 tuple）
                expanded = torch.ops.aten.expand.default(x, (1, 5))
                return expanded

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        fold_expand(gm.graph)

        expand_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.expand.default]
        self.assertEqual(len(expand_nodes), 1, "shape 不是 list 不应折叠")


instantiate_parametrized_tests(TestFoldExpandPass)


if __name__ == "__main__":
    run_tests()