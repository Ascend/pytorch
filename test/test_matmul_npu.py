# Owner(s): ["module: linear algebra"]

import unittest
from functools import partial

import torch
from torch.testing import make_tensor
import torch_npu
import torch_npu.testing
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    tol as xtol,
    toleranceOverride,
)

from torch.testing._internal.common_utils import (
    IS_JETSON,
    parametrize,
    run_tests,
    skipIfRocmVersionLessThan,
    TEST_WITH_ROCM,
    TestCase,
)

# Protects against includes accidentally setting the default dtype
# NOTE: jit_metaprogramming_utils sets the default dtype to double!
torch.set_default_dtype(torch.float32)
if torch.get_default_dtype() is not torch.float32:
    raise RuntimeError("The torch default dtype is not float32!")


class TestMatmul(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()

    def tearDown(self):
        super(self.__class__, self).tearDown()

    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=1e-1, rtol=1e-1),
                        torch.bfloat16: xtol(atol=1e-1, rtol=1e-1),
                        torch.float32: xtol(atol=1e-1, rtol=1e-1)})
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("size", [100, 1000, 10000])
    def test_addmm(self, size: int, dtype: torch.dtype):
        #
        # Check for catastrophic cuBLAS inaccuracy by measuring the deviation between
        # results from the NPU invocation of torch.addmm and the CPU invocation
        # (which does not use NPU backend).
        #
        # Get dims
        n, m, p = (size + 1, size, size + 2)
        # Make random tensors on CPU (seed set on common_utils.py import)
        # (Not using numpy because it does not support bfloat16)
        make_arg = partial(make_tensor, dtype=dtype, device="cpu")
        m_beta = make_arg(1)
        m_input = make_arg((n, p))
        m_1 = make_arg((n, m))
        m_2 = make_arg((m, p))
        # *(B)FLOAT16 Special Handling*
        # Backend does not tensorize float16 on CPU,
        # and bloat16 may present accuracy issues,
        # so convert to float32 for these cases
        # (but keep same for other types, e.g. float32 and int*)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=torch.float32)
            m_input = m_input.to(dtype=torch.float32)
            m_1 = m_1.to(dtype=torch.float32)
            m_2 = m_2.to(dtype=torch.float32)
        # Get CPU result
        res_cpu = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # *(B)FLOAT16 Special Handling*``
        # Convert back to (b)float16
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=dtype)
            m_input = m_input.to(dtype=dtype)
            m_1 = m_1.to(dtype=dtype)
            m_2 = m_2.to(dtype=dtype)
            res_cpu = res_cpu.to(dtype=dtype)
        # Move arg tensors to NPU
        m_beta = m_beta.to("npu")
        m_input = m_input.to("npu")
        m_1 = m_1.to("npu")
        m_2 = m_2.to("npu")
        # Get NPU result
        res_npu = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # Move to CPU for comparison
        res_npu = res_npu.to("cpu")
        # Compare
        self.assertEqual(res_cpu, res_npu)

    def test_addmm_alignment(self):
        dtype = torch.half
        device = 'npu'
        # perturb X, A, or B alignment
        for idx in range(0, 3):
            for offset in range(1, 3):
                offsets = [0, 0, 0]
                offsets[idx] = offset
                x_offset, a_offset, b_offset = offsets
                A = torch.rand((5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device)
                A = A[a_offset:].reshape(5120, 2560)
                X = torch.rand((26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device)
                X = X[x_offset:].reshape(26, 1, 2560)
                B = torch.rand((5120 + b_offset), requires_grad=True, dtype=dtype, device=device)
                B = B[b_offset:].reshape(5120)
                out = torch.nn.functional.linear(X, A, B)
                self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)

    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    @toleranceOverride({torch.float32: xtol(atol=1e-5, rtol=1e-5)})
    @dtypes(*([torch.float32, torch.float16, torch.bfloat16]))
    @parametrize(
        "batch_size, N, M, P",
        [(2, 100, 100, 100),
         (2, 1000, 1000, 1000),
         (1, 10000, 1000, 10000),
         (1, 10000, 10000, 10000)],
        name_fn=lambda batch_size, N, M, P: f"{batch_size}_{N}_{M}_{P}",
    )
    def test_baddbmm_large_input(self, device, batch_size, N, M, P, dtype):
        cpu_dtype = dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            cpu_dtype = torch.float32

        M1 = torch.rand((N, M), device=device, dtype=dtype)
        M2 = torch.rand((M, P), device=device, dtype=dtype)
        A = torch.rand((N, P), device=device, dtype=dtype)

        def _convert_to_cpu(t):
            return t.to(device='cpu', dtype=cpu_dtype)
        M1_cpu, M2_cpu, A_cpu = map(_convert_to_cpu, [M1, M2, A])

        # linear
        out1_cpu = torch.nn.functional.linear(M1_cpu, M2_cpu.t(), A_cpu).to(dtype=dtype)
        out1_gpu = torch.nn.functional.linear(M1, M2.t(), A).cpu()
        self.assertEqual(out1_cpu, out1_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = torch.eye(N, device=device, dtype=dtype)
            out1_eye_gpu = torch.nn.functional.linear(M1, M2_eye.t(), torch.zeros_like(A))
            self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu())

        # baddbmm
        def _expand_to_batch(t: torch.Tensor):
            return t.expand((batch_size, ) + t.size())
        alpha, beta = 1.0, 1.0
        M1, M2, A, M1_cpu, M2_cpu, A_cpu = map(_expand_to_batch, [M1, M2, A, M1_cpu, M2_cpu, A_cpu])

        out2_cpu = torch.baddbmm(A_cpu, M1_cpu, M2_cpu, beta=beta, alpha=alpha).to(dtype=dtype)
        out2_gpu = torch.baddbmm(A, M1, M2, beta=beta, alpha=alpha).cpu()
        self.assertEqual(out2_cpu, out2_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = torch.eye(N, device=device, dtype=dtype).expand(batch_size, N, N)
            out2_eye_gpu = torch.baddbmm(torch.zeros_like(A), M1, M2_eye, beta=beta, alpha=alpha)
            self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu())

        # cross comparison
        self.assertEqual(out1_gpu, out2_gpu[0])


instantiate_device_type_tests(TestMatmul, globals(), only_for='privateuse1')


if __name__ == '__main__':
    run_tests()
