import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_sink_view


class FoldSinkViewModel(torch.nn.Module):
    def forward(self, t1):
        view_1 = torch.ops.aten.view.default(t1, [1, -1])
        output = torch.ops.aten.relu.default(view_1)
        return output


class TestFoldSinkViewPass(TestUtils):
    def op_calc(self, t1):
        x = t1.view(1, -1) # reshape
        return torch.relu(x)



    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldSinkViewModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_sink_view(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8, 16)])
    @parametrize('dtype', ['float32'])
    def test_view_before_act_should_sink(self, shape, dtype):
        """view → relu → 应下沉为 relu → view"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, (-1, 16))
                a = torch.ops.aten.relu.default(v)
                return a + 0.5

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        gm.graph.print_tabular()
        fold_sink_view(gm.graph)
        gm.graph.print_tabular()
        gm.recompile()

        view_nodes = [n for n in gm.graph.nodes if 'view' in str(n.target)]
        act_nodes = [n for n in gm.graph.nodes if 'relu' in str(n.target)]

        self.assertEqual(len(view_nodes), 1, "应保留一个 view")
        self.assertEqual(len(act_nodes), 1, "应保留一个 relu")
        # 检查顺序：relu 在 view 之前
        relu_node = act_nodes[0]
        view_node = view_nodes[0]
        self.assertIn(view_node, relu_node.users, "relu 的输出应连接到 view")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(2, 3, 4)])
    @parametrize('dtype', ['float32'])
    def test_view_before_supported_binary_op_should_sink(self, shape, dtype):
        """view(x) + y → view(x) + y"""
        class M(torch.nn.Module):
            def forward(self, x, y):
                v = torch.ops.aten.view.default(x, (-1,))
                result = torch.ops.aten.add.Tensor(v, y)
                return result

        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor((24,), dtype)  # 广播兼容
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x, y)
        fold_sink_view(gm.graph)
        gm.recompile()
        self.assertEqual(model(x, y), gm(x, y))


    @parametrize('shape', [(4, 5)])
    @parametrize('dtype', ['float32'])
    def test_view_with_multiple_users_no_sink(self, shape, dtype):
        """view 有多个用户 → 不下沉"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, (-1,))
                a = torch.ops.aten.relu.default(v)
                b = torch.ops.aten.add.Scalar(v, 1.0)
                return a, b

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_sink_view(gm.graph)
        view_nodes = [n for n in gm.graph.nodes if 'view' in str(n.target)]
        self.assertEqual(len(view_nodes), 1, "多用户时不应下沉 view")
        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(1, 8, 16)])
    def test_chain_view_act_should_sink_multiple_times(self, shape):
        """多层 view + act → 应多次下沉"""
        class M(torch.nn.Module):
            def forward(self, x):
                v1 = torch.ops.aten.view.default(x, (-1, 16))
                a1 = torch.ops.aten.relu.default(v1)
                v2 = torch.ops.aten.view.default(a1, (-1,))
                a2 = torch.ops.aten.sigmoid.default(v2)
                return a2

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_sink_view(gm.graph)
        gm.recompile()

        view_nodes = [n for n in gm.graph.nodes if 'view' in str(n.target)]
        self.assertEqual(len(view_nodes), 2, "应保留两个 view，但顺序下沉")
        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_view_before_unsupported_op_no_sink(self, shape, dtype):
        """view 前接不支持的操作 → 不下沉"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, (-1,))
                # 假设 conv 不支持下沉（根据 check_support_op 判断）
                result = torch.ops.aten.conv1d.default(v.unsqueeze(1), torch.randn(1, 12, 1).npu())
                return result.squeeze()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_sink_view(gm.graph)
        view_nodes = [n for n in gm.graph.nodes if 'view' in str(n.target)]
        self.assertEqual(len(view_nodes), 1, "不支持的操作前不应下沉")


    @parametrize('shape', [(2, 3)])
    def test_no_view_node_no_change(self, shape):
        """无 view 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.add.Tensor(x, x) * 2.0

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_sink_view(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(4, 5)])
    def test_view_without_shape_info_no_sink(self, shape):
        """输入无 shape 信息 → 不下沉"""
        class M(torch.nn.Module):
            def forward(self, x):
                v = torch.ops.aten.view.default(x, (-1, 5))
                return torch.ops.aten.add.Tensor(x, v)

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]
        fold_sink_view(gm.graph)
        view_nodes = [n for n in gm.graph.nodes if 'view' in str(n.target)]
        self.assertEqual(len(view_nodes), 1, "无 shape 信息不应下沉")


instantiate_parametrized_tests(TestFoldSinkViewPass)


if __name__ == "__main__":
    run_tests()