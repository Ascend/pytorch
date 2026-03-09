import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_to_copy


class FoldToCopyModel(torch.nn.Module):
    def forward(self, t1):
        copy_1 = torch.ops.aten._to_copy.default(t1)
        result = torch.ops.aten.add.Tensor(t1, copy_1)
        return result


class TestFoldToCopyPass(TestUtils):
    def op_calc(self, t1):
        copy_1 = torch.ops.aten._to_copy.default(t1)
        result = copy_1 + t1
        return result


    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(t1)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(t1)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
    
    @parametrize('shape', [(2, 4)])
    @parametrize('dtype', ['float32'])
    def test_ut_cases(self, shape, dtype):
        t1 = self._generate_tensor(shape, dtype)
        model = FoldToCopyModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(t1)
        
        # 应用优化 Pass
        fold_to_copy(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(t1)
        inductor_result = graph_module(t1)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 8)])
    @parametrize('dtype', ['float32'])
    def test_useless_to_copy_same_dtype_should_fold(self, shape, dtype):
        """dtype 相同、无其他变化 → 应折叠删除 _to_copy"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
                return copied + 1.0

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_to_copy(gm.graph)
        gm.recompile()

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 0, "无意义的 _to_copy 应被删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(5, 5)])
    @parametrize('dtype', ['float32'])
    def test_to_copy_different_dtype_no_fold(self, shape, dtype):
        """dtype 不同 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x, dtype=torch.float64)
                return copied.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_to_copy(gm.graph)

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 1, "dtype 不同不应折叠")


    @parametrize('shape', [(3, 7)])
    def test_to_copy_in_output_path_no_fold(self, shape):
        """_to_copy 是 output 的直接/间接输入 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x)
                return copied  # 直接输出

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_to_copy(gm.graph)

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 1, "在 output 路径上的 _to_copy 不应折叠")


    @parametrize('shape', [(2, 16)])
    def test_multi_to_copy_some_fold_some_not(self, shape):
        """多个 _to_copy，只有部分无意义可折叠"""
        class M(torch.nn.Module):
            def forward(self, x, y):
                c1 = torch.ops.aten._to_copy.default(x)                # 无意义，可折叠
                c2 = torch.ops.aten._to_copy.default(y, dtype=torch.float64)  # dtype 不同，不可折叠
                return c1 + c2

        x = self._generate_tensor(shape, 'float32')
        y = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x, y)
        fold_to_copy(gm.graph)
        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 1, "只有一个有意义的 _to_copy 保留")


    @parametrize('shape', [(1, 64)])
    def test_to_copy_with_multiple_downstream_users(self, shape):
        """_to_copy 被折叠后，下游多个用户正确指向原输入"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x)
                a = copied * 3.0
                b = copied + 5.0
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_to_copy(gm.graph)
        gm.recompile()

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 0)

        self.assertEqual(model(x), gm(x))


    def test_no_to_copy_node_no_change(self):
        """无 _to_copy 节点 → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = torch.randn(8, 8)
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_to_copy(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(4, 4)])
    def test_to_copy_without_meta_no_fold(self, shape):
        """缺少 meta 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x)
                return copied

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 meta
        for node in gm.graph.nodes:
            if "tensor_meta" in node.meta:
                del node.meta["tensor_meta"]

        fold_to_copy(gm.graph)

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 1, "缺少 meta 不应折叠")


    @parametrize('shape', [(8, 8)])
    def test_to_copy_with_memory_format_specified_no_fold(self, shape):
        """指定 memory_format → 不折叠（即使当前相同）"""
        class M(torch.nn.Module):
            def forward(self, x):
                copied = torch.ops.aten._to_copy.default(x, memory_format=torch.preserve_format)
                return copied

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_to_copy(gm.graph)

        to_copy_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        self.assertEqual(len(to_copy_nodes), 1, "有 memory_format 参数不应折叠")


instantiate_parametrized_tests(TestFoldToCopyPass)


if __name__ == "__main__":
    run_tests()