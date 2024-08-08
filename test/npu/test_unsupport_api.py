import os
import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

IS_HOSTAPI_ENABLED = os.getenv('HOSTAPI_ENABLED') == 'ON'

class TestPtaUnsupportApi(TestCase):

    def test_bitwise_left_shift_Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift.Tensor is unsupported!"):
            torch.bitwise_left_shift(torch.tensor([-1, -2, 3]).npu(), torch.tensor([1, 0, 3]).npu())

    def test_bitwise_left_shift__Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift_.Tensor is unsupported!"):
            torch.tensor([-1, -2, 3]).npu().bitwise_left_shift_(torch.tensor([1, 0, 3]).npu())

    def test_bitwise_left_shift_Tensor_out(self):
        a = torch.tensor([-1, -2, 3]).npu()
        b = torch.tensor([1, 0, 3]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift.Tensor_out is unsupported!"):
            torch.bitwise_left_shift(a, b, out=result)

    def test_bitwise_left_shift_Tensor_Scalar(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift.Tensor_Scalar is unsupported!"):
            torch.bitwise_left_shift(torch.tensor([-1, -2, 3]).npu(), 1)

    def test_bitwise_left_shift__Tensor_Scalar(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift_.Tensor_Scalar is unsupported!"):
            torch.tensor([-1, -2, 3]).npu().bitwise_left_shift_(1)

    def test_bitwise_left_shift_Tensor_Scalar_out(self):
        a = torch.tensor([-1, -2, 3]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift.Tensor_Scalar_out is unsupported!"):
            torch.bitwise_left_shift(a, 1, out=result)

    def test_bitwise_left_shift_Scalar_Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_left_shift.Scalar_Tensor is unsupported!"):
            torch.bitwise_left_shift(4, torch.tensor([-1, -2, 3]).npu())

    def test_bitwise_right_shift_Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift.Tensor is unsupported!"):
            torch.bitwise_right_shift(torch.tensor([-1, -2, 3]).npu(), torch.tensor([1, 0, 3]).npu())

    def test_bitwise_right_shift__Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift_.Tensor is unsupported!"):
            torch.tensor([-1, -2, 3]).npu().bitwise_right_shift_(torch.tensor([1, 0, 3]).npu())

    def test_bitwise_right_shift_Tensor_out(self):
        a = torch.tensor([-1, -2, 3]).npu()
        b = torch.tensor([1, 0, 3]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift.Tensor_out is unsupported!"):
            torch.bitwise_right_shift(a, b, out=result)

    def test_bitwise_right_shift_Tensor_Scalar(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift.Tensor_Scalar is unsupported!"):
            torch.bitwise_right_shift(torch.tensor([-1, -2, 3]).npu(), 1)

    def test_bitwise_right_shift__Tensor_Scalar(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift_.Tensor_Scalar is unsupported!"):
            torch.tensor([-1, -2, 3]).npu().bitwise_right_shift_(1)

    def test_bitwise_right_shift_Tensor_Scalar_out(self):
        a = torch.tensor([-1, -2, 3]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift.Tensor_Scalar_out is unsupported!"):
            torch.bitwise_right_shift(a, 1, out=result)

    def test_bitwise_right_shift_Scalar_Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "bitwise_right_shift.Scalar_Tensor is unsupported!"):
            torch.bitwise_right_shift(4, torch.tensor([-1, -2, 3]).npu())

    def test__conj_physical(self):
        with self.assertRaisesRegex(RuntimeError, "_conj_physical is unsupported!"):
            torch._conj_physical(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).npu())

    def test_conj_physical(self):
        with self.assertRaisesRegex(RuntimeError, "conj_physical is unsupported!"):
            torch.conj_physical(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).npu())

    def test_conj_physical_out(self):
        a = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "onj_physical.out is unsupported!"):
            torch.conj_physical(a, out=result)

    def test_conj_physical_(self):
        with self.assertRaisesRegex(RuntimeError, "conj_physical_ is unsupported!"):
            torch.conj_physical_(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).npu())

    def test_frexp_Tensor(self):
        with self.assertRaisesRegex(RuntimeError, "frexp.Tensor is unsupported!"):
            torch.frexp(torch.tensor([-1, -2, 3]).npu())

    def test_frexp_Tensor_out(self):
        a = torch.tensor([-1, -2, 3]).npu()
        mantissa = torch.empty_like(a)
        exponent = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "frexp.Tensor_out is unsupported!"):
            torch.frexp(a, out=(mantissa, exponent))

    def test_cholesky_out(self):
        a = torch.randn(3, 3).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "cholesky.out is unsupported!"):
            torch.cholesky(a, out=result)

    def test_cholesky(self):
        with self.assertRaisesRegex(RuntimeError, "cholesky is unsupported!"):
            torch.cholesky(torch.randn(3, 3).npu())

    def test_geqrf(self):
        with self.assertRaisesRegex(RuntimeError, "geqrf is unsupported!"):
            torch.geqrf(torch.randn(3, 3).npu())

    @unittest.skipIf(IS_HOSTAPI_ENABLED, "Hostapi support logdet.")
    def test_logdet(self):
        with self.assertRaisesRegex(RuntimeError, "logdet is unsupported!"):
            torch.logdet(torch.randn(3, 3).npu())

    def test_linalg_lu_factor_ex(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_lu_factor_ex is unsupported!"):
            torch.linalg.lu_factor_ex(torch.randn(3, 3).npu())

    def test_linalg_lu_factor_ex_out(self):
        A = torch.randn(3, 3).npu()
        LU = torch.empty_like(A)
        pivots = torch.empty(3)
        infos = torch.empty(1)
        with self.assertRaisesRegex(RuntimeError, "linalg_lu_factor_ex.out is unsupported!"):
            torch.linalg.lu_factor_ex(A, out=(LU, pivots, infos))

    def test_lu_solve_out(self):
        LU, pivots = torch.linalg.lu_factor(torch.randn(2, 3, 3))
        result = torch.empty_like(LU)
        with self.assertRaisesRegex(RuntimeError, "lu_solve.out is unsupported!"):
            torch.lu_solve(torch.randn(2, 3, 1).npu(), LU, pivots, out=result)

    def test_lu_solve(self):
        LU, pivots = torch.linalg.lu_factor(torch.randn(2, 3, 3))
        with self.assertRaisesRegex(RuntimeError, "lu_solve is unsupported!"):
            torch.lu_solve(torch.randn(2, 3, 1).npu(), LU, pivots)

    def test_lu_unpack(self):
        LU, pivots = torch.linalg.lu_factor(torch.randn(2, 3, 3))
        with self.assertRaisesRegex(RuntimeError, "lu_unpack is unsupported!"):
            torch.lu_unpack(LU.npu(), pivots)

    def test_lu_unpack_out(self):
        A = torch.randn(3, 3)
        LU, pivots = torch.linalg.lu_factor(A)
        L = torch.empty_like(A)
        U = torch.empty_like(A)
        with self.assertRaisesRegex(RuntimeError, "lu_unpack.out is unsupported!"):
            torch.lu_unpack(LU.npu(), pivots, out=(L, U, A))

    def test__det_lu_based_helper(self):
        with self.assertRaisesRegex(RuntimeError, "_det_lu_based_helper is unsupported!"):
            torch._det_lu_based_helper(torch.randn(3, 3).npu())

    def test_linalg_det(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_det is unsupported!"):
            torch.linalg.det(torch.randn(3, 3).npu())

    def test_linalg_det_out(self):
        a = torch.randn(3, 3).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "linalg_det.out is unsupported!"):
            torch.linalg.det(a, out=result)

    def test_linalg_cholesky_ex(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_cholesky_ex is unsupported!"):
            torch.linalg.cholesky_ex(torch.randn(3, 3).npu())

    def test_linalg_eig(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_eig is unsupported!"):
            torch.linalg.eig(torch.randn(3, 3).npu())

    def test_linalg_eig_out(self):
        a = torch.randn(3, 3).npu()
        eigenvalues = torch.empty_like(a)
        eigenvectors = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "linalg_eig.out is unsupported!"):
            torch.linalg.eig(a, out=(eigenvalues, eigenvectors))

    def test_linalg_eigh(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_eigh is unsupported!"):
            torch.linalg.eigh(torch.randn(3, 3).npu())

    def test_linalg_eigvalsh(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_eigvalsh is unsupported!"):
            torch.linalg.eigvalsh(torch.randn(3, 3).npu())

    def test_linalg_eigvalsh_out(self):
        a = torch.randn(3, 3).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "linalg_eigvalsh.out is unsupported!"):
            torch.linalg.eigvalsh(a, out=result)

    def test_linalg_lstsq(self):
        with self.assertRaisesRegex(RuntimeError, "linalg_lstsq is unsupported!"):
            torch.linalg.lstsq(torch.randn(1, 3, 3).npu(), torch.randn(2, 3, 3).npu())

    def test_linalg_lstsq_out(self):
        a = torch.randn(5, 3).npu()
        b = torch.randn(5, 2).npu()

        X = torch.empty(3, 2)
        residuals = torch.empty(5)
        with self.assertRaisesRegex(RuntimeError, "linalg_lstsq.out is unsupported!"):
            torch.linalg.lstsq(a, b, out=(X, residuals, a, b))

    def test_special_entr_out(self):
        a = torch.arange(-0.5, 1, 0.5).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "special_entr.out is unsupported!"):
            torch.special.entr(a, out=result)

    def test_special_entr(self):
        with self.assertRaisesRegex(RuntimeError, "special_entr is unsupported!"):
            torch.special.entr(torch.arange(-0.5, 1, 0.5).npu())

    def test_special_erfcx_out(self):
        a = torch.tensor([0, -1., 10.]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "special_erfcx.out is unsupported!"):
            torch.special.erfcx(a, out=result)

    def test_special_erfcx(self):
        with self.assertRaisesRegex(RuntimeError, "special_erfcx is unsupported!"):
            torch.special.erfcx(torch.tensor([0, -1., 10.]).npu())

    def test_special_zeta(self):
        with self.assertRaisesRegex(RuntimeError, "special_zeta is unsupported!"):
            torch.special.zeta(torch.tensor([2., 4.]).npu(), torch.tensor([2., 4.]).npu())

    def test_special_zeta_self_scalar(self):
        with self.assertRaisesRegex(RuntimeError, "special_zeta.self_scalar is unsupported!"):
            torch.special.zeta(2., torch.tensor([2., 4.]).npu())

    def test_special_zeta_other_scalar(self):
        with self.assertRaisesRegex(RuntimeError, "special_zeta.other_scalar is unsupported!"):
            torch.special.zeta(torch.tensor([2., 4.]).npu(), 1)

    def test_special_zeta_out(self):
        a = torch.tensor([2., 4.]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "special_zeta.out is unsupported!"):
            torch.special.zeta(torch.tensor([2., 4.]).npu(), torch.tensor([2., 4.]).npu(), out=result)

    def test_special_zeta_self_scalar_out(self):
        a = torch.tensor([2., 4.]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "special_zeta.self_scalar_out is unsupported!"):
            torch.special.zeta(2., torch.tensor([2., 4.]).npu(), out=result)

    def test_special_zeta_other_scalar_out(self):
        a = torch.tensor([2., 4.]).npu()
        result = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "special_zeta.other_scalar_out is unsupported!"):
            torch.special.zeta(torch.tensor([2., 4.]).npu(), 1, out=result)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_special_devcie_tensor(self):
        torch.npu.set_device(0)
        a = torch.tensor([2., 4.], device="npu:1")
        if hasattr(a.storage(), "device"):
            self.assertEqual(a.device, a.storage().device)
            self.assertEqual(a.device.index, a.storage().device.index)
        with self.assertRaisesRegex(AssertionError, "1 not less than or equal to 1e-05"):
            self.assertEqual(a.device.index, 1)

if __name__ == "__main__":
    run_tests()
