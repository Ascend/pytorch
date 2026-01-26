import torch
import torch.fx as fx
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests
)
from testutils import TestUtils
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import dtype_optimal_pass


class DtypeOptimalPass(TestUtils):
    def create_gm(self, fn):
        """辅助函数: 创建 FX GraphModule (需启用 fake mode 以填充 meta['example_value'])"""
        gm = fx.symbolic_trace(fn)
        return gm

    def create_manual_gm_for_arange(self, start=0, end=2048, step=1, dtype=torch.int64):
        """手动创建 FX GM，避免常量折叠"""
        gm = fx.GraphModule({}, fx.Graph())
        arange_node = gm.graph.call_function(
            torch.arange,
            args=(end,),
            kwargs={'start': start, 'step': step, 'dtype': dtype}
        )
        gm.graph.output(arange_node)
        gm.recompile()
        return gm

    def apply_pass(self, gm):
        """应用 pass 并返回修改后的 gm"""
        dtype_optimal_pass(gm.graph)
        gm.recompile()
        return gm

    def test_static_safe_conversion_single_arg(self):
        """使用手动 GM 避免折叠"""
        gm = self.create_manual_gm_for_arange(end=2048)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int32", modified_readable)
        self.assertIn("arange", modified_readable)  # 确认无折叠

    def test_static_safe_conversion_multi_arg(self):
        """测试静态多参数，范围安全: arange(1000, 3000, 2, dtype=int64) -> int32"""
        gm = self.create_manual_gm_for_arange(start=1000, end=3000, step=2)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int32", modified_readable)

    def test_static_fallback_out_of_range(self):
        """测试静态范围超出: arange(2**31 + 1, dtype=int64) -> 保留 int64"""
        gm = self.create_manual_gm_for_arange(end=2**31 + 1)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int64", modified_readable)  # fallback

    def test_negative_step_safe(self):
        """测试负 step，范围安全: arange(2048, 0, -1, dtype=int64) -> int32"""        
        gm = self.create_manual_gm_for_arange(start=2048, end=0, step=-1)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int32", modified_readable)

    def test_zero_step_skip(self):
        """测试 zero step: 跳过转换"""
        gm = self.create_manual_gm_for_arange(start=0, end=10, step=0)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int64", modified_readable)  # 保留

    def test_non_integer_step_fallback(self):
        """测试非整数 step: fallback"""
        gm = self.create_manual_gm_for_arange(start=0, end=10, step=0.5)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.int64", modified_readable)  # fallback
    
    def test_non_int64_dtype_skip(self):
        """测试非 int64 dtype: 跳过"""
        gm = self.create_manual_gm_for_arange(end=2048, dtype=torch.float32)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("dtype = torch.float32", modified_readable)  # 未变

    def test_conversion_float32_to_int64(self):
        """测试 float32 输入的 to(int64) -> to(int32)"""
        def fn(x):
            return x.to(torch.int64)  # x 是 float32
        
        gm = self.create_gm(fn)
        # 模拟 meta['example_value'] (在 fake mode 下自动填充)
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        input_node.meta['example_value'] = torch.tensor(1.0, dtype=torch.float32)  # 模拟 float32 输入
        
        original_readable = gm.print_readable()
        self.assertIn("to(torch.int64)", original_readable)
        
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("to(torch.int32)", modified_readable)
    
    def test_conversion_bool_to_int64(self):
        """测试 bool 输入的 to(int64) -> to(int32)"""
        def fn(x):
            return x.to(torch.int64)  # x 是 bool
        
        gm = self.create_gm(fn)
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        input_node.meta['example_value'] = torch.tensor(True, dtype=torch.bool)  # 模拟 bool 输入
        
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("to(torch.int32)", modified_readable)

    def test_no_conversion_wrong_target_dtype(self):
        """测试 target_dtype 非 int64: 无转换"""
        def fn(x):
            return x.to(torch.float32)  # 非 int64
        
        gm = self.create_gm(fn)
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        input_node.meta['example_value'] = torch.tensor(1.0, dtype=torch.float32)
        
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("to(torch.float32)", modified_readable)  # 未变
    
    def test_no_conversion_wrong_target_method(self):
        """测试 target 非 'to': 无转换"""
        def fn(x):
            return x.view(1, -1)  # 非 'to'
        
        gm = self.create_gm(fn)
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        input_node.meta['example_value'] = torch.tensor(1.0, dtype=torch.float32)
        
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("view", modified_readable)  # 未变
    
    def test_kwargs_dtype_handling(self):
        """测试 kwargs 中 dtype: to(int64) -> to(int32)"""
        def fn(x):
            return x.to(dtype=torch.int64)  # 使用 kwargs
        
        gm = self.create_gm(fn)
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        input_node.meta['example_value'] = torch.tensor(1.0, dtype=torch.float32)
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("to(dtype = torch.int32)", modified_readable)
    
    def test_no_meta_example_value_skip(self):
        """测试无 meta['example_value']: 无转换 (安全 fallback)"""
        def fn(x):
            return x.to(torch.int64)
        
        gm = self.create_gm(fn)
        # 手动移除 meta 以模拟无 example_value
        input_node = next(n for n in gm.graph.nodes if n.op == 'placeholder')
        if 'example_value' in input_node.meta:
            del input_node.meta['example_value']
        
        gm = self.apply_pass(gm)
        modified_readable = gm.print_readable()
        self.assertIn("to(torch.int64)", modified_readable)  # 未变


instantiate_parametrized_tests(DtypeOptimalPass)

if __name__ == "__main__":
    run_tests()