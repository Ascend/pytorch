import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_four_op_pass


class FoldAddModel(torch.nn.Module):
    def forward(self, first_element):
        add = torch.ops.aten.add.Tensor(first_element, 0)
        add_output = torch.ops.aten.relu.default(add)
        return add_output


class TestFoldAddPass(TestUtils):
    def op_calc(self, first_element):
        add = torch.add(first_element, 0)
        add_output = torch.relu(add)
        return add_output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(first_element)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        model = FoldAddModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(first_element)

        # 应用优化 Pass
        fold_four_op_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(first_element)
        inductor_result = graph_module(first_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_add_zero_left_fold(self, shape, dtype):
        """ 0 + x 应该折叠成 x """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.add.Tensor(0, x)   # 左边是 0 (scalar)
                return torch.relu(y + 1.0)

        t = self._generate_tensor(shape, dtype)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        # 检查 pass 是否删除了 add 节点
        self.assertFalse(
            any(n.target in (torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
                for n in gm.graph.nodes),
            "add + 0 should be folded away"
        )

        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(5, 5)])
    def test_add_scalar_zero_right_fold(self, shape):
        """ x + 0.0 (scalar) 应该折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.add(x, 0.0)           # 使用 python torch.add → 应转为 add.Scalar
                return y * 2.0

        t = torch.randn(shape)
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertFalse(
            any(n.target in (torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
                for n in gm.graph.nodes)
        )
        torch.testing.assert_close(m(t), gm(t), atol=1e-5, rtol=1e-5)


    @parametrize('shape', [(2, 3, 4)])
    def test_add_int_zero_right_fold(self, shape):
        """ x + 0 (int literal) """
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.add.Tensor(x, 0) + x

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertEqual(m(t), gm(t))


    @parametrize('shape', [(10,)])
    def test_add_zero_not_fold_nonzero(self, shape):
        """ x + 1.5  不应该折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.add(x, 1.5)
                return torch.sigmoid(y)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        before_nodes = len([n for n in gm.graph.nodes if n.op == 'call_function'])

        fold_four_op_pass(gm.graph)
        gm.recompile()

        after_nodes = len([n for n in gm.graph.nodes if n.op == 'call_function'])
        self.assertEqual(before_nodes, after_nodes, "不应折叠非零常量")


    @parametrize('shape', [(3, 4)])
    def test_add_zero_chain_fold(self, shape):
        """ 多个连续的 +0 折叠 """
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.ops.aten.add.Tensor(x, 0)
                b = torch.ops.aten.add.Tensor(a, 0.0)
                c = torch.ops.aten.add.Tensor(b, torch.zeros_like(x))
                return torch.ops.aten.relu.default(c)

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)
        fold_four_op_pass(gm.graph)
        gm.recompile()

        add_nodes = [n for n in gm.graph.nodes if 'add' in str(n.target)]
        self.assertEqual(len(add_nodes), 0, "所有 +0 都应该被折叠掉")


    def test_no_add_op_no_change(self):
        """ 图中没有 add 节点，pass 应安全通过 """
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, 2.0) + torch.pow(x, 2)

        t = torch.randn(8)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(t)

        orig_graph_str = str(gm.graph)
        fold_four_op_pass(gm.graph)
        self.assertEqual(orig_graph_str, str(gm.graph), "无 add 节点不应修改图")


    @parametrize('shape', [(1, 128)])
    def test_add_zero_with_downstream_users(self, shape):
        """ +0 后有多个下游用户，都应正确替换 """
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.add(x, 0)
                return (y * 2, y + 3, torch.relu(y))

        t = self._generate_tensor(shape, 'float32')
        m = M()
        gm = fx.symbolic_trace(m)
        ShapeProp(gm).propagate(t)

        fold_four_op_pass(gm.graph)
        gm.recompile()

        self.assertEqual(m(t), gm(t))


instantiate_parametrized_tests(TestFoldAddPass)


if __name__ == "__main__":
    run_tests()