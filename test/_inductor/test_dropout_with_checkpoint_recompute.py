import torch
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu
    
class TestDropoutWithCheckpointRecompute(TestUtils):
    def test_dropout_with_checkpoint_recompute(self):
        device = "npu"
        
        def gn(x):
            return torch.sigmoid(torch.dropout(torch.sigmoid(x), p=0.5, train=True))
        
        def fn(x):
            return checkpoint(
                gn,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        
        x = torch.randn(4, 4, requires_grad=True, device=device)
        
        torch.manual_seed(42)
        eager_out = fn(x)
        eager_out.sum().backward()
        eager_grad = x.grad.clone()
        
        x.grad = None
        
        torch.manual_seed(42)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_out = compiled_fn(x)
        compiled_out.sum().backward()
        compiled_grad = x.grad.clone()
        
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_grad, compiled_grad)

instantiate_parametrized_tests(TestDropoutWithCheckpointRecompute)

if __name__ == "__main__":
    run_tests()