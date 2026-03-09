import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_detach


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
        fold_detach(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_detach_should_be_removed(self, shape, dtype):
        """普通 detach → 应被完全删除"""
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.detach.default(x)
                return torch.relu(y + 1.0)

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_detach(gm.graph)
        gm.recompile()

        detach_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.detach.default]
        self.assertEqual(len(detach_nodes), 0, "detach 节点应被删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(5, 5)])
    def test_multi_detach_chain_should_all_be_removed(self, shape):
        """多层连续 detach → 应全部删除"""
        class M(torch.nn.Module):
            def forward(self, x):
                y1 = torch.ops.aten.detach.default(x)
                y2 = torch.ops.aten.detach.default(y1)
                y3 = torch.ops.aten.detach.default(y2)
                return y3 * 2.0 + x

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_detach(gm.graph)
        gm.recompile()

        detach_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.detach.default]
        self.assertEqual(len(detach_nodes), 0, "多层 detach 应全部被折叠删除")


    @parametrize('shape', [(3, 7)])
    def test_detach_with_multiple_downstream_users(self, shape):
        """detach 后有多个下游用户 → 全部替换回原始输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                detached = torch.ops.aten.detach.default(x)
                a = detached * 3.0
                b = detached + 5.0
                c = torch.relu(detached)
                return a, b, c

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_detach(gm.graph)
        gm.recompile()
        detach_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.detach.default]
        self.assertEqual(len(detach_nodes), 0)


    @parametrize('shape', [(2, 16)])
    def test_no_detach_node_no_change(self, shape):
        """图中没有 detach 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph_str = str(gm.graph)
        fold_detach(gm.graph)
        self.assertEqual(orig_graph_str, str(gm.graph), "无 detach 节点不应修改图")


    @parametrize('shape', [(1, 64)])
    def test_detach_in_complex_graph(self, shape):
        """复杂图中 detach 被折叠后不影响其他节点"""
        class M(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z_detach = torch.ops.aten.detach.default(z)
                out1 = z_detach * 2.0
                out2 = torch.cat([z_detach, y], dim=0)
                return out1, out2

        x = self._generate_tensor(shape, 'float32')
        y = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x, y)

        fold_detach(gm.graph)
        gm.recompile()

        detach_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.detach.default]
        self.assertEqual(len(detach_nodes), 0)

        self.assertEqual(model(x, y), gm(x, y))


    @parametrize('shape', [(8, 8)])
    def test_detach_with_requires_grad_input(self, shape):
        """输入 requires_grad=True → detach 折叠后数值一致（inference 阶段安全）"""
        x = self._generate_tensor(shape, 'float32')
        x.requires_grad_(True)

        class M(torch.nn.Module):
            def forward(self, x):
                return x.detach() + 1.0

        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_detach(gm.graph)
        gm.recompile()

        self.assertEqual(model(x), gm(x))


instantiate_parametrized_tests(TestFoldDetachPass)


if __name__ == "__main__":
    run_tests()