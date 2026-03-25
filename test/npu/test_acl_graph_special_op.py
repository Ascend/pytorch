import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclGraphSpecialOp(TestCase):
    @staticmethod
    def _fn_masked_assign_fwd(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Core pattern:  tensor[bool_mask] = -1
        Compiles to aten.index_put_ with boolean indices → cudagraph-unsafe.
        A relu op follows to verify that the fallback path runs normally.
        """
        x = inp.clone()
        x[mask] = -1.0          # aten.index_put_ / bool indices → skip aclgraph
        out = torch.relu(x)     # normal op: must execute correctly after fallback
        return out

    @staticmethod
    def _fn_masked_assign_fwd_bwd(inp: torch.Tensor) -> torch.Tensor:
        """
        Dynamic bool mask derived from input values, followed by a mul op.
        Returns a scalar so that .backward() can be called directly.
        The backward must produce correct gradients through the fallback path.
        """
        mask = inp > 0           # bool mask whose shape depends on runtime values
        x = inp.clone()
        x[mask] = -1.0           # aten.index_put_ with bool indices
        out = x * 2.0            # normal op following the special op
        return out.sum()

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_masked_assign_forward(self):
        """
        Verify that tensor[bool_mask] = -1 followed by relu produces the
        same result whether executed eagerly or through torch.compile.

        The compiled graph contains aten.index_put_ with bool indices, which
        triggers the cudagraph-unsafe detection in _graph_tree.py.  The graph
        is therefore skipped from ACL graph capture and runs in eager mode.
        """
        shape = (8, 16)
        npu = torch.device("npu")
        inp = torch.randn(shape, dtype=torch.float32, device=npu)
        mask = torch.randint(0, 2, shape, device=npu).bool()

        std_result = self._fn_masked_assign_fwd(inp, mask)

        compiled_fn = torch.compile(
            self._fn_masked_assign_fwd, backend="inductor",
            options={"triton.cudagraphs": True},
        )
        compiled_result = compiled_fn(inp, mask)

        self.assertEqual(std_result, compiled_result, prec=1e-3)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_masked_assign_forward_backward(self):
        """
        Verify forward output and input gradients for a function that:
          1. Builds a dynamic bool mask  (inp > 0)
          2. Assigns -1 to masked positions  (index_put_ with bool indices)
          3. Multiplies the result by 2.0   (normal op)
          4. Reduces to a scalar            (sum)

        The compiled version must:
          - Detect the boolean index_put and skip ACL graph capture.
          - Fall back to eager, preserving correct autograd behaviour.
          - Produce the same scalar output and input gradients as eager.
        """
        shape = (8, 16)
        npu = torch.device("npu")

        # Eager reference tensors
        x_ref = torch.randn(shape, dtype=torch.float32, device=npu,
                            requires_grad=True)
        # Compiled-path tensors (same initial values)
        x = x_ref.detach().clone().requires_grad_(True)

        # ---- Eager forward + backward ----
        std_out = self._fn_masked_assign_fwd_bwd(x_ref)
        std_out.backward()

        # ---- Compiled forward + backward ----
        compiled_fn = torch.compile(
            self._fn_masked_assign_fwd_bwd, backend="inductor",
            options={"triton.cudagraphs": True},
        )
        compiled_out = compiled_fn(x)
        compiled_out.backward()

        # Forward output must match
        self.assertEqual(std_out, compiled_out, prec=1e-3)
        # Input gradients must match
        self.assertEqual(x.grad, x_ref.grad, prec=1e-3)


    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npugraphs_masked_assign_forward(self):
        """
        Same as test_masked_assign_forward but using the npugraphs backend.
        Verifies that the npugraphs backend also correctly detects
        aten.index_put_ with bool indices and falls back to eager.
        """
        shape = (8, 16)
        npu = torch.device("npu")
        inp = torch.randn(shape, dtype=torch.float32, device=npu)
        mask = torch.randint(0, 2, shape, device=npu).bool()

        std_result = self._fn_masked_assign_fwd(inp, mask)

        compiled_fn = torch.compile(
            self._fn_masked_assign_fwd, backend="npugraphs",
        )
        compiled_result = compiled_fn(inp, mask)

        self.assertEqual(std_result, compiled_result, prec=1e-3)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npugraphs_masked_assign_forward_backward(self):
        """
        Same as test_masked_assign_forward_backward but using the npugraphs
        backend. Verifies forward output and input gradients are correct
        when the npugraphs backend falls back to eager due to bool index_put.
        """
        shape = (8, 16)
        npu = torch.device("npu")

        x_ref = torch.randn(shape, dtype=torch.float32, device=npu,
                            requires_grad=True)
        x = x_ref.detach().clone().requires_grad_(True)

        # ---- Eager forward + backward ----
        std_out = self._fn_masked_assign_fwd_bwd(x_ref)
        std_out.backward()

        # ---- Compiled forward + backward ----
        compiled_fn = torch.compile(
            self._fn_masked_assign_fwd_bwd, backend="npugraphs",
        )
        compiled_out = compiled_fn(x)
        compiled_out.backward()

        self.assertEqual(std_out, compiled_out, prec=1e-3)
        self.assertEqual(x.grad, x_ref.grad, prec=1e-3)



if __name__ == "__main__":
    run_tests()
