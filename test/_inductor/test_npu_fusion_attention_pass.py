import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class FusionAttentionUnchangeModel(torch.nn.Module):
    def forward(self, primals_1, primals_2, primals_3):
        npu_fusion_attention = torch.ops.npu.npu_fusion_attention.default(primals_1, primals_2, primals_3, 8, 'BNSD', None, None, None, 0.125, 0.9)
        getitem = npu_fusion_attention[0]
        getitem_1 = npu_fusion_attention[1]
        getitem_2 = npu_fusion_attention[2]
        getitem_3 = npu_fusion_attention[3]
        getitem_4 = npu_fusion_attention[4]
        getitem_5 = npu_fusion_attention[5]
        return {"getitem": getitem, "getitem_1": getitem_1, "getitem_2": getitem_2, "getitem_3": getitem_3, "getitem_4": getitem_4, "getitem_5": getitem_5}


class TestFusionAttentionUnchangePass(TestUtils):
    def op_calc(self, primals_1, primals_2, primals_3):
        npu_fusion_attention_v3 = torch.ops.npu.npu_fusion_attention_v3.default(primals_1, primals_2, primals_3, 8, 'BNSD', None, None, None, 0.125, 0.9)
        getitem = npu_fusion_attention_v3[0]
        getitem_1 = npu_fusion_attention_v3[1]
        getitem_2 = npu_fusion_attention_v3[2]
        getitem_3 = npu_fusion_attention_v3[3]
        getitem_4 = npu_fusion_attention_v3[4]
        getitem_5 = npu_fusion_attention_v3[5]
        return {"getitem": getitem, "getitem_1": getitem_1, "getitem_2": getitem_2, "getitem_3": getitem_3, "getitem_4": getitem_4, "getitem_5": getitem_5}



    def test_compile_cases(self):
        primals_1 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")
        primals_2 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")
        primals_3 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")

        state = torch.npu.get_rng_state()
        std_result = self.op_calc(primals_1, primals_2, primals_3)
        torch.npu.set_rng_state(state)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(primals_1, primals_2, primals_3)
        self.assertIsInstance(inductor_result["getitem_4"], torch.Tensor, "The output parameter 'seed' of the npu_fusion_attention_v3 should be Tensor")
        self.assertIsInstance(inductor_result["getitem_5"], torch.Tensor, "The output parameter 'offset' of the npu_fusion_attention_v3 should be Tensor")
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)
        
        
    def test_ut_cases(self):
        primals_1 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")
        primals_2 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")
        primals_3 = torch.randn(2, 8, 16, 64, dtype=torch.float32, device="npu")
        model = FusionAttentionUnchangeModel()
        graph_module = fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(primals_1, primals_2, primals_3)
        
        # 应用优化 Pass
        from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import fusion_attention_v3_pass
        fusion_attention_v3_pass(graph_module.graph)
        graph_module.recompile()

        # 验证输出是否一致
        state = torch.npu.get_rng_state()
        std_result = model(primals_1, primals_2, primals_3)
        torch.npu.set_rng_state(state)
        inductor_result = graph_module(primals_1, primals_2, primals_3)
        self.assertIsInstance(inductor_result["getitem_4"], torch.Tensor, "After the pass, the output parameter 'seed' of the npu_fusion_attention should be Tensor")
        self.assertIsInstance(inductor_result["getitem_5"], torch.Tensor, "After the pass, the output parameter 'offset' of the npu_fusion_attention should be Tensor")
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestFusionAttentionUnchangePass)


if __name__ == "__main__":
    run_tests()