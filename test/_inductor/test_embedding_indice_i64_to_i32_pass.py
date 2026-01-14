import torch
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests
)
from testutils import TestUtils
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import embedding_indice_i64_to_i32_pass
from torch_npu._inductor.fx_passes.utils.check_op_util import check_embedding_op


@instantiate_parametrized_tests
class TestEmbeddingIndiceI64ToI32Pass(TestUtils):
    # here, we use 'self.emb_table(input_ids)'
    class EmbeddingModel_X(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_table = torch.nn.Embedding.from_pretrained(
                    torch.normal(
                        mean=0, std=0.1, size=(30522, 768)
                    )
                ).to('npu')
            
        def forward(self, input_ids):
            return self.emb_table(input_ids)
            
    @parametrize('shape', [(128, 300)])
    @parametrize('dtype', ['int64'])
    def test_compile_case(self, shape, dtype):
        input_ids = self._generate_tensor(shape, dtype)
        model = self.EmbeddingModel_X()
        model.eval()
        
        with torch.no_grad():
            compile_model = torch.compile(model, backend="inductor")
            compile_result = compile_model(input_ids)
            
            eager_result = model(input_ids)
            
            self.assertEqual(eager_result, compile_result, atol=1e-3, rtol=1e-3)
            
    # here, we use 'torch.nn.functional.embedding'
    class EmbeddingModel_Y(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            seed = 2026
            torch.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            self.emb_table = torch.nn.Embedding(30522, 768, padding_idx=0).to('npu')
            torch.nn.init.uniform_(self.emb_table.weight, a=-1.0, b=1.0)
            
        def forward(self, input_ids):
            return torch.nn.functional.embedding(input_ids, self.emb_table.weight)
    
    @parametrize('shape', [(128, 300)])
    @parametrize('dtype', ['int64'])
    def test_fx_model(self, shape, dtype):
        input_ids = self._generate_tensor(shape, dtype)
        model = self.EmbeddingModel_Y()
        model.eval()
        
        gm = torch.fx.symbolic_trace(model)
        ShapeProp(gm).propagate(input_ids)
        
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                old_meta = node.meta['tensor_meta']
                node.meta['tensor_meta'] = old_meta._replace(dtype=torch.int64)
        
        embedding_indice_i64_to_i32_pass(gm.graph)
        gm.recompile()
        
        # assert equal
        eager_result = model(input_ids)
        gm_result = gm(input_ids)
        self.assertEqual(eager_result, gm_result, atol=1e-3, rtol=1e-3)
        
        # test result
        embedding_node = None
        emb_input_is_cast = False
        found_cast = False
        
        for node in gm.graph.nodes:
            # check one of embedding-node's input-nodes is the inserted node
            if check_embedding_op(node):
                embedding_node = node
                if str(node.args[0].target) == "npu._npu_dtype_cast.default" or str(node.args[1].target) == "npu._npu_dtype_cast.default":
                    emb_input_is_cast = True
                    
            # check if cast node is inserted
            if "dtype_cast" in str(node.target) and node.kwargs.get('dtype') == torch.int32:
                found_cast = True
                
        self.assertTrue(found_cast, "cast-to-int32-node is not inserted by pass")
        self.assertIsNotNone(embedding_node)
        self.assertTrue(emb_input_is_cast, "embedding_node's input is not npu._npu_dtype_cast.default")


if __name__ == "__main__":
    run_tests()