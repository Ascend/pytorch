import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSymeig(TestCase):
    def op_exec(self, input1, eigenvectorsflag):
        npu_input = input1.npu()
        en, vn = torch.symeig(npu_input, eigenvectors=eigenvectorsflag)
        if eigenvectorsflag:
            ret = torch.matmul(vn, torch.matmul(en.diag_embed(), vn.transpose(-2, -1)))
            self.assertRtolEqual(ret.cpu(), input1, prec=1e-3)
        else:
            e, v = torch.symeig(input1, eigenvectors=eigenvectorsflag)
            self.assertEqual(e, en.cpu())
            self.assertEqual(v, vn.cpu())

    def case_exec(self, input1):
        input1 = input1 + input1.transpose(-2, -1)
        self.op_exec(input1, False)
        self.op_exec(input1, True)

    @unittest.skip("skip test_symeig_null now")
    def test_symeig_null(self, device="npu"):
        a = torch.randn(0, 0)
        self.op_exec(a, False)
        self.op_exec(a, True)

    @unittest.skip("skip test_symeig_2d now")
    def test_symeig_2d(self, device="npu"):
        a = torch.randn(5, 5, dtype=torch.float32)
        self.case_exec(a)

    @unittest.skip("skip test_symeig_3d now")
    def test_symeig_3d(self, device="npu"):
        a = torch.randn(10, 5, 5, dtype=torch.float32)
        self.case_exec(a)

    @unittest.skip("skip test_symeig_4d now")
    def test_symeig_4d(self, device="npu"):
        a = torch.randn(10, 3, 5, 5, dtype=torch.float32)
        self.case_exec(a)

    @unittest.skip("skip test_symeig_5d now")
    def test_symeig_5d(self, device="npu"):
        a = torch.randn(2, 10, 3, 5, 5, dtype=torch.float32)
        self.case_exec(a)

    @unittest.skip("skip test_symeig_out now")
    def test_symeig_out(self, device="npu"):
        a = torch.randn(2, 3, 3, dtype=torch.float32)
        a = a + a.transpose(-2, -1)
        an = a.npu()
        e = torch.zeros(2, 3).npu()
        v = torch.zeros(2, 3).npu()
        out = torch.symeig(an, eigenvectors=True, out=(e, v))
        ret = torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1)))
        self.assertRtolEqual(ret.cpu(), a, prec=1e-3)


if __name__ == "__main__":
    run_tests()
