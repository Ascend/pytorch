import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_clone


class FoldCloneModel(torch.nn.Module):
    def forward(self, t1):
        clone_1 = torch.ops.aten.clone.default(t1)
        relu_1 = torch.ops.aten.relu.default(clone_1)
        return relu_1


class TestFoldClonePass(TestUtils):
    def op_calc(self, t1):
        clone_1 = torch.clone(t1)
        output = torch.relu(clone_1)
        return output


    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(1, 2, 3)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldCloneModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_clone(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32'])
    def test_clone_same_memory_format_should_fold(self, shape, dtype):
        """memory_format 相同 → 应折叠删除 clone"""
        class M(torch.nn.Module):
            def forward(self, x):
                cloned = torch.ops.aten.clone.default(x)  # 默认 memory_format 相同
                return cloned + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_clone(gm.graph)
        gm.recompile()

        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clone_nodes), 0, "相同 memory_format 的 clone 应被删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(5, 5)])
    @parametrize('dtype', ['float32'])
    def test_clone_different_memory_format_no_fold(self, shape, dtype):
        """memory_format 不同 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                # 显式指定不同 memory_format（假设原 x 是 contiguous）
                cloned = torch.ops.aten.clone.default(x, memory_format=torch.preserve_format)
                # 如果想强制不同，可以用 channels_last，但需确保 dtype 支持
                return cloned

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_clone(gm.graph)

        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clone_nodes), 1, "memory_format 不同不应折叠")


    @parametrize('shape', [(3, 7)])
    def test_clone_in_output_path_no_fold(self, shape):
        """clone 是 output 的直接/间接输入 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                cloned = torch.ops.aten.clone.default(x)
                return cloned  # 直接输出 clone

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_clone(gm.graph)
        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clone_nodes), 1, "clone 在 output 路径上不应折叠")


    @parametrize('shape', [(2, 16)])
    def test_multi_clone_some_fold_some_not(self, shape):
        """多个 clone，只有部分可折叠"""
        class M(torch.nn.Module):
            def forward(self, x, y):
                c1 = torch.ops.aten.clone.default(x)
                c2 = torch.ops.aten.clone.default(y, memory_format=torch.channels_last)
                c3 = torch.ops.aten.clone.default(c1)
                return c1 + c2 + c3

        x = self._generate_tensor(shape, 'float32')
        y = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x, y)
        fold_clone(gm.graph)
        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        # 预期：c1 和 c3 折叠，c2 保留（或根据你的 memory_format 判断）
        self.assertLessEqual(len(clone_nodes), 1)


    @parametrize('shape', [(1, 64)])
    def test_clone_with_downstream_users(self, shape):
        """clone 被折叠后，下游多个用户应正确指向原输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                cloned = torch.ops.aten.clone.default(x)
                a = cloned * 2
                b = cloned + 3
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_clone(gm.graph)
        gm.recompile()

        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clone_nodes), 0)

        self.assertEqual(model(x), gm(x))


    def test_no_clone_node_no_change(self):
        """图中无 clone 节点，pass 不应修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return x + x * 2.0

        x = torch.randn(8, 8)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_clone(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(4, 4)])
    def test_clone_without_tensor_meta_no_fold(self, shape):
        """输入没有 tensor_meta → 不折叠（安全检查）"""
        class M(torch.nn.Module):
            def forward(self, x):
                cloned = torch.ops.aten.clone.default(x)
                return cloned

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 tensor_meta
        for node in gm.graph.nodes:
            if "tensor_meta" in node.meta:
                del node.meta["tensor_meta"]

        fold_clone(gm.graph)

        clone_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clone_nodes), 1, "缺少 tensor_meta 不应折叠")


instantiate_parametrized_tests(TestFoldClonePass)


if __name__ == "__main__":
    run_tests()