import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fold_slice


class FoldSliceModel(torch.nn.Module):
    def forward(self, base, view, t1, t2, t3):
        end = 16
        slice_1 = torch.ops.aten.slice_scatter.default(base, view, 1, 0, end)
        result = view + slice_1
        slice_2 = torch.ops.aten.slice_scatter.default(t1, t2, 1, 0, 3)
        
        b = torch.ops.aten.slice.Tensor(t3, 1, 0, None)
        result_c = torch.ops.aten.add.Tensor(b, b)
        return {"result": result, "slice_2": slice_2, "result_c": result_c}


class TestFoldSlicePass(TestUtils):
    def op_calc(self, base, view, t1, t2, t3):
        end = view.shape[1]
        result = torch.slice_scatter(base, view, 1, 0, end)
        result = result + view
        data = t1.slice_scatter(t2, dim=1, start=0, end=t2.shape[1])
        
        b = t3[:, 0:]
        result_c = b + b
        return {"result": result, "data": data, "result_c": result_c}


    def test_compile_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        std_result = self.op_calc(base, view, t1, t2, t3)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(base, view, t1, t2, t3)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)
        t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = torch.tensor([[9, 9, 9], [8, 8, 8]])
        t3 = torch.randn(4, 16, 32, 64)
        model = FoldSliceModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(base, view, t1, t2, t3)
        
        # 应用优化 Pass
        fold_slice(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        std_result = model(base, view, t1, t2, t3)
        inductor_result = graph_module(base, view, t1, t2, t3)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


    @parametrize('shape', [(4, 100, 16)])
    @parametrize('dtype', ['float32'])
    def test_slice_prefix_fold(self, shape, dtype):
        """slice 前缀（0:start:1）且 start >= 原长 → 应折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.slice.Tensor(x, 1, 0, 100, 1)
                return torch.relu(s)

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_slice(gm.graph)
        gm.recompile()

        slice_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor]
        self.assertEqual(len(slice_nodes), 0, "前缀 slice 应被折叠删除")

        self.assertEqual(model(x), gm(x))


    @parametrize('shape', [(4, 100, 16)])
    @parametrize('dtype', ['float32'])
    def test_slice_full_range_no_fold(self, shape, dtype):
        """slice 全范围 pass 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.slice.Tensor(x, 1, None, None, 1)
                return s.mean()

        x = self._generate_tensor(shape, dtype)
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_slice(gm.graph)
        slice_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor]
        self.assertEqual(len(slice_nodes), 1, "全范围 slice 应折叠为 identity")


    @parametrize('shape', [(8, 20, 32)])
    def test_slice_scatter_fold(self, shape):
        """slice_scatter 场景折叠（常见于 padding + slice 组合）"""
        class M(torch.nn.Module):
            def forward(self, x, y):
                scattered = torch.ops.aten.slice_scatter.default(x, y, 1, 0, 20)
                return scattered.sum()

        x = self._generate_tensor(shape, 'float32')
        y = self._generate_tensor((8, 20, 32), 'float32')  # 假设 y shape 匹配 slice 范围
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x, y)
        fold_slice(gm.graph)
        gm.recompile()

        scatter_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice_scatter.default]
        # 如果 _fold_slice_scatter 成功折叠，则期望 0
        self.assertEqual(len(scatter_nodes), 0, "可折叠的 slice_scatter 应被移除")


    @parametrize('shape', [(16, 32)])
    def test_multi_slice_chain_fold(self, shape):
        """多层连续 slice → 应多次折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s1 = torch.ops.aten.slice.Tensor(x, 0, 0, 17, 1)
                s2 = torch.ops.aten.slice.Tensor(s1, 1, 0, 33, 1)
                s3 = torch.ops.aten.slice.Tensor(s2, 0, 0, 18, 1)
                return s3.mean()

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)

        fold_slice(gm.graph)
        gm.recompile()

        slice_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor]
        self.assertEqual(len(slice_nodes), 0, "多层可折叠 slice 应全部移除")


    @parametrize('shape', [(8, 20)])
    def test_slice_multi_users_no_fold(self, shape):
        """slice 有多个用户 → 折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.slice.Tensor(x, 0, 0, 10, 1)
                a = s * 2
                b = s + 3
                return a, b

        x = self._generate_tensor(shape, 'float32')
        model = M()
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(x)
        fold_slice(gm.graph)
        slice_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor]
        self.assertEqual(len(slice_nodes), 0, "多用户 slice 应折叠")


    @parametrize('shape', [(4, 5, 6)])
    def test_no_slice_node_no_change(self, shape):
        """无 slice/slice_scatter → pass 不修改图"""
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x) * 2.0

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        ShapeProp(gm).propagate(x)

        orig_graph = str(gm.graph)
        fold_slice(gm.graph)
        self.assertEqual(orig_graph, str(gm.graph))


    @parametrize('shape', [(2, 100)])
    def test_slice_without_shape_info_no_fold(self, shape):
        """输入无 shape 信息 → 不折叠"""
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.ops.aten.slice.Tensor(x, 0, 0, 50, 1)
                return s.sum()

        x = self._generate_tensor(shape, 'float32')
        gm = fx.symbolic_trace(M())
        # 故意移除 shape
        for node in gm.graph.nodes:
            if "val" in node.meta:
                del node.meta["val"]

        fold_slice(gm.graph)

        slice_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.aten.slice.Tensor]
        self.assertEqual(len(slice_nodes), 1, "无 shape 信息不应折叠")


instantiate_parametrized_tests(TestFoldSlicePass)


if __name__ == "__main__":
    run_tests()