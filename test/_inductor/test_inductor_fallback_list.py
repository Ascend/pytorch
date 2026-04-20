import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch._inductor.utils import run_and_get_code
import torch_npu
import os
os.environ["NPU_INDUCTOR_FALLBACK_LIST"] = "aten.div,aten.add.Tensor"

class TestFallback(TestUtils):
    
    def add_op(self, x, y):
            return x / y
        
    def test_add_fallback_detection(self):
        
        compiled_add = torch.compile(self.add_op, backend="inductor")
        
        x = torch.randn(4, 4, dtype=torch.float32).to("npu")
        y = torch.randn(4, 4, dtype=torch.float32).to("npu")

        _ , codes = run_and_get_code(compiled_add, x, y)

        self.assertTrue('unk_fused_div' not in codes[0])
    
    def test_add_fallback_detection_mlir(self):

        compiled_add = torch.compile(self.add_op, backend="inductor", options={"npu_backend": "mlir"})
        
        x = torch.randn(4, 4, dtype=torch.float32).to("npu")
        y = torch.randn(4, 4, dtype=torch.float32).to("npu")

        _ , codes = run_and_get_code(compiled_add, x, y)

        self.assertTrue('mlir_fused_div' not in codes[0])


if __name__ == "__main__":
    run_tests()